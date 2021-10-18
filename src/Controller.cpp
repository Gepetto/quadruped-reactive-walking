#include "qrw/Controller.hpp"

Controller::Controller()
    : P(Vector12::Zero()),
      D(Vector12::Zero()),
      q_des(Vector12::Zero()),
      v_des(Vector12::Zero()),
      tau_ff(Vector12::Zero()),
      FF(Vector12::Zero()),
      error(false),
      error_flag(0),
      error_value(Vector12::Zero()),
      k(0),
      k_mpc(0),
      q_filt_mpc(Vector18::Zero()),
      h_v_filt_mpc(Vector6::Zero()),
      vref_filt_mpc(Vector6::Zero()),
      o_targetFootstep(Matrix34::Zero()),
      q_wbc(Vector18::Zero()),
      dq_wbc(Vector18::Zero()),
      xgoals(Vector12::Zero()),
      hRb(Matrix3::Identity()),
      p_ref_(Vector6::Zero())
{
  /*namespace bi = boost::interprocess;
  bi::shared_memory_object::remove("SharedMemory");*/

  /*//Remove shared memory on construction and destruction
  struct shm_remove
  {
    shm_remove() { bi::shared_memory_object::remove("SharedMemory"); }
    ~shm_remove(){ bi::shared_memory_object::remove("SharedMemory"); }
  } remover;*/
}

void Controller::initialize(Params& params) {
  // Params store parameters
  params_ = &params;

  // Init robot parameters
  init_robot(params);

  // Initialization of the control blocks
  joystick.initialize(params);
  statePlanner.initialize(params);
  gait.initialize(params);
  footstepPlanner.initialize(params, gait);
  mpcWrapper.initialize(params);
  footTrajectoryGenerator.initialize(params, gait);
  estimator.initialize(params);
  wbcWrapper.initialize(params);

  filter_mpc_q.initialize(params);
  filter_mpc_v.initialize(params);
  filter_mpc_vref.initialize(params);

  // Other variables
  k_mpc = static_cast<int>(params.dt_mpc / params.dt_wbc);
  h_ref_ = params.h_ref;
  P = (Vector3(params.Kp_main.data())).replicate<4, 1>();
  D = (Vector3(params.Kd_main.data())).replicate<4, 1>();
  FF = params.Kff_main * Vector12::Ones();
}

//void Controller::compute(std::shared_ptr<odri_control_interface::Robot> robot) {
void Controller::compute(FakeRobot *robot) {
  // std::cout << "Computing Controller" << std::endl;

  // Update the reference velocity coming from the gamepad
  joystick.update_v_ref(k, params_->velID, gait.getIsStatic(), estimator.getHVWindowed().head(6));

  // Process state estimator
  estimator.run_filter(gait.getCurrentGait(),
                       footTrajectoryGenerator.getFootPosition(),
                       robot->imu->GetLinearAcceleration(),
                       robot->imu->GetGyroscope(),
                       robot->imu->GetAttitudeEuler(),
                       robot->joints->GetPositions(),
                       robot->joints->GetVelocities(),
                       Vector3::Zero(),
                       Vector3::Zero());

  // Update state vectors of the robot (q and v) + transformation matrices between world and horizontal frames
  estimator.updateState(joystick.getVRef(), gait);

  // Update gait
  gait.updateGait(k, k_mpc, joystick.getJoystickCode());

  // Quantities go through a 1st order low pass filter with fc = 15 Hz (avoid >25Hz foldback)
  q_filt_mpc.head(6) = filter_mpc_q.filter(estimator.getQUpdated().head(6), true);
  q_filt_mpc.tail(12) = estimator.getQUpdated().tail(12);
  h_v_filt_mpc = filter_mpc_v.filter(estimator.getHV().head(6), false);
  vref_filt_mpc = filter_mpc_vref.filter(estimator.getVRef().head(6), false);

  // Compute target footstep based on current and reference velocities
  o_targetFootstep = footstepPlanner.updateFootsteps((k % k_mpc == 0) && (k != 0),
                                                      static_cast<int>(k_mpc - (k % k_mpc)),
                                                      estimator.getQUpdated().head(18),
                                                      estimator.getHVWindowed().head(6),
                                                      estimator.getVRef().head(6));

  // Run state planner (outputs the reference trajectory of the base)
  statePlanner.computeReferenceStates(q_filt_mpc.head(6), h_v_filt_mpc, vref_filt_mpc, 0.0);

  // Solve MPC problem once every k_mpc iterations of the main loop
  if (k % k_mpc == 0)
  {
    mpcWrapper.solve(k, statePlanner.getReferenceStates(), footstepPlanner.getFootsteps(),
                     gait.getCurrentGait());
  }

  // Update pos, vel and acc references for feet
  footTrajectoryGenerator.update(k, o_targetFootstep);

  // Whole Body Control
  // If nothing wrong happened yet in the WBC controller
  if (!error && !joystick.getStop())
  {
    /*
    if self.gait.getIsStatic():
                hRb = np.eye(3)

            # Desired position, orientation and velocities of the base
            self.xgoals[:6, 0] = np.zeros((6,))
            if self.joystick.getL1() and self.gait.getIsStatic():
                self.p_ref[:, 0] = self.joystick.getPRef()
                # self.p_ref[3, 0] = np.clip((self.k - 2000) / 2000, 0.0, 1.0)
                self.xgoals[[3, 4], 0] = self.p_ref[[3, 4], 0]
                self.h_ref = self.p_ref[2, 0]
                hRb = pin.rpy.rpyToMatrix(0.0, 0.0, self.p_ref[5, 0])
                # print(self.joystick.getPRef())
                # print(self.p_ref[2])
            else:
                self.h_ref = self.h_ref_mem
      */

    if (gait.getIsStatic()) {hRb.setIdentity();}
    else { hRb = estimator.gethRb();}

    xgoals.head(6).setZero();
    if (joystick.getL1() && gait.getIsStatic())
    {
      p_ref_ = joystick.getPRef();
      h_ref_ = p_ref_(2, 0);
      xgoals(3, 0) = p_ref_(3, 0);
      xgoals(4, 0) = p_ref_(4, 0);
      hRb = pinocchio::rpy::rpyToMatrix(0.0, 0.0, p_ref_(5, 0));
    }
    else
    {
      h_ref_ = params_->h_ref;
    }


    // Update configuration vector for wbc
    q_wbc(3, 0) = q_filt_mpc(3, 0);  // Roll
    q_wbc(4, 0) = q_filt_mpc(4, 0);  // Pitch
    q_wbc.tail(12) = wbcWrapper.get_qdes();  // with reference angular positions of previous loop

    // Update velocity vector for wbc
    dq_wbc.head(6) = estimator.getVFilt().head(6);  //  Velocities in base frame (not horizontal frame!)
    dq_wbc.tail(12) = wbcWrapper.get_vdes();  // with reference angular velocities of previous loop

    // Desired position, orientation and velocities of the base
    xgoals.tail(6) = vref_filt_mpc;  // Velocities (in horizontal frame!)

    /*std::cout << q_wbc.transpose() << std::endl;
    std::cout << dq_wbc.transpose() << std::endl;
    std::cout << gait.getCurrentGait().row(0)  << std::endl;
    std::cout << footTrajectoryGenerator.getFootAccelerationBaseFrame(estimator.gethRb() * estimator.getoRh().transpose(),
                                                            Vector3::Zero(), Vector3::Zero()) << std::endl;
    std::cout << footTrajectoryGenerator.getFootVelocityBaseFrame(estimator.gethRb() * estimator.getoRh().transpose(),
                                                        Vector3::Zero(), Vector3::Zero()) << std::endl;
    std::cout << footTrajectoryGenerator.getFootPositionBaseFrame(estimator.gethRb() * estimator.getoRh().transpose(),
                                                        estimator.getoTh() + Vector3(0.0, 0.0, params_->h_ref)) << std::endl;
    std::cout << mpcWrapper.get_latest_result().block(12, 0, 12, 1) << std::endl;*/

    //Vector12 f_mpc = mpcWrapper.get_latest_result().block(12, 0, 12, 1);
    //std::cout << "PASS" << std::endl << mpcWrapper.get_latest_result().block(12, 0, 12, 1) << std::endl;
    /*if (k == 0)
    {
      double t = 0;
      while (t < 1.0)
      {
        std::cout << "Boop" << std::endl;
        t += 0.5;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      }
    }*/

    // Run InvKin + WBC QP
    wbcWrapper.compute(
      q_wbc, dq_wbc, mpcWrapper.get_latest_result().block(12, 0, 12, 1), gait.getCurrentGait().row(0),
      footTrajectoryGenerator.getFootPositionBaseFrame(hRb * estimator.getoRh().transpose(),
                                                        estimator.getoTh() + Vector3(0.0, 0.0, h_ref_)),
      footTrajectoryGenerator.getFootVelocityBaseFrame(hRb * estimator.getoRh().transpose(),
                                                        Vector3::Zero(), Vector3::Zero()),
      footTrajectoryGenerator.getFootAccelerationBaseFrame(hRb * estimator.getoRh().transpose(),
                                                            Vector3::Zero(), Vector3::Zero()),
      xgoals);

    // Quantities sent to the control board
    q_des = wbcWrapper.get_qdes();
    v_des = wbcWrapper.get_vdes();
    tau_ff = wbcWrapper.get_tau_ff();

    /*if (k == 0) {
      std::cout << std::fixed;
      std::cout << std::setprecision(5);
    }
    std::cout << "--- " << k << std::endl;
    std::cout << mpcWrapper.get_latest_result().block(12, 0, 12, 1).transpose() << std::endl;
    std::cout << q_des.transpose() << std::endl;
    std::cout << v_des.transpose() << std::endl;
    std::cout << tau_ff.transpose() << std::endl;
    std::cout << xgoals.transpose() << std::endl;*/
  }

  // Security check
  security_check();

  // Increment loop counter
  k++;
}

void Controller::init_robot(Params & params)
{
  // Path to the robot URDF (TODO: Automatic path)
  const std::string filename =
      std::string("/opt/openrobots/share/example-robot-data/robots/solo_description/robots/solo12.urdf");

  // Robot model
  pinocchio::Model model_;

  // Build model from urdf (base is not free flyer)
  pinocchio::urdf::buildModel(filename, pinocchio::JointModelFreeFlyer(), model_, false);

  // Construct data from model
  pinocchio::Data data_ = pinocchio::Data(model_);

  // Update all the quantities of the model
  VectorN q_tmp = VectorN::Zero(model_.nq);
  q_tmp(6, 0) = 1.0;  // Quaternion (0, 0, 0, 1)
  q_tmp.tail(12) = Vector12(params.q_init.data());

  // Initialisation of model quantities
  pinocchio::computeAllTerms(model_, data_, q_tmp, VectorN::Zero(model_.nv));
  pinocchio::centerOfMass(model_, data_, q_tmp, VectorN::Zero(model_.nv));
  pinocchio::updateFramePlacements(model_, data_);
  pinocchio::crba(model_, data_, q_tmp);

  // Initialisation of the position of footsteps
  Matrix34 fsteps_init = Matrix34::Zero();
  int indexes [4] = {static_cast<int>(model_.getFrameId("FL_FOOT")), static_cast<int>(model_.getFrameId("FR_FOOT")),
                     static_cast<int>(model_.getFrameId("HL_FOOT")), static_cast<int>(model_.getFrameId("HR_FOOT"))};
  for (int i = 0; i < 4; i++) {
    fsteps_init.col(i) = data_.oMf[indexes[i]].translation();
  }

  // Get default height
  double h_init = 0.0;
  double h_tmp = 0.0;
  for (int i = 0; i < 4; i++) {
    h_tmp = (data_.oMf[1].translation() - data_.oMf[indexes[i]].translation())(2, 0);
    if (h_tmp > h_init) { h_init = h_tmp; }
  }

  // Assumption that all feet are initially in contact on a flat ground
  fsteps_init.row(2).setZero();

  // Initialisation of the position of shoulders
  Matrix34 shoulders_init = Matrix34::Zero();
  int indexes_sh [4] = {4, 12, 20, 28}; //  Shoulder indexes
  for (int i = 0; i < 4; i++) {
    shoulders_init.col(i) = data_.oMf[indexes_sh[i]].translation();
  }

  // Saving data
  params_->h_ref = h_init;  // Reference height
  params_->mass = data_.mass[0];  // Mass

  // Inertia matrix
  Vector6 Idata = data_.Ycrb[1].inertia().data();
  Matrix3 inertia;
  inertia << Idata(0, 0), Idata(1, 0), Idata(3, 0),
             Idata(1, 0), Idata(2, 0), Idata(4, 0),
             Idata(3, 0), Idata(4, 0), Idata(5, 0);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      params_->I_mat[3 * i + j] = inertia(i ,j);
    }
  }

  // Offset between center of base and CoM
  Vector3 CoM = data_.com[0].head(3) - q_tmp.head(3);
  params_->CoM_offset[0] = CoM(0, 0);
  params_->CoM_offset[1] = CoM(1, 0);
  params_->CoM_offset[2] = CoM(2, 0);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 3; j++) {
      params_->shoulders[3 * i + j] = shoulders_init(j, i);
      params_->footsteps_init[3 * i + j] = fsteps_init(j, i);
      params_->footsteps_under_shoulders[3 * i + j] = fsteps_init(j, i);  // Use initial feet pos as reference
    }
  }
  
  /*std::cout << "DEBUG:" << std::endl;
  std::cout << params_->h_ref << std::endl;
  std::cout << params_->mass << std::endl;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      std::cout << params_->I_mat[3 * i + j] << " ";
    }
  }
  std::cout << std::endl;
  std::cout << params_->CoM_offset[0] << std::endl;
  std::cout << params_->CoM_offset[1] << std::endl;
  std::cout << params_->CoM_offset[2] << std::endl;

  std::cout << fsteps_init << std::endl;
  std::cout << shoulders_init << std::endl;
  std::cout << "-----" << std::endl;*/

}

void Controller::security_check()
{
  if (error_flag == 0 && !error)
  {
    error_flag = estimator.security_check(tau_ff);
    if (error_flag != 0)
    {
      error = true;
      switch (error_flag)
      {
        case 1:
          error_value = estimator.getQFilt().tail(12) * 180 / 3.1415;
          break;
        case 2:
          error_value = estimator.getVSecu();
          break;
        default:
          error_value = tau_ff;
      }
    }
  }

  if(joystick.getStop())
  {
    error = true;
    error_flag = -1;
  }

  // If something wrong happened in the controller we stick to a security controller
  if (error)
  {
    // Quantities sent to the control board
    P = Vector12::Zero();
    D = 0.1 * Vector12::Ones();
    q_des = Vector12::Zero();
    v_des = Vector12::Zero();
    FF = Vector12::Zero();
    tau_ff = Vector12::Zero();
  }
}
