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
      xgoals(Vector12::Zero()) {}

void Controller::initialize(Params& params) {
  // Params store parameters
  params_ = &params;

  // Initialization of the control blocks
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
  P = (Vector3(params.Kp_main.data())).replicate<4, 1>();
  D = (Vector3(params.Kd_main.data())).replicate<4, 1>();
  FF = params.Kff_main * Vector12::Ones();
}

// void Controller::compute(std::shared_ptr<odri_control_interface::Robot> robot) {
void Controller::compute(FakeRobot *robot) {
  std::cout << "Computing Controller" << std::endl;

  // Update the reference velocity coming from the gamepad
  joystick.update_v_ref(k, params_->velID);

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
      // Update configuration vector for wbc
      q_wbc(3, 0) = q_filt_mpc(3, 0);  // Roll
      q_wbc(4, 0) = q_filt_mpc(4, 0);  // Pitch
      q_wbc.tail(12) = wbcWrapper.get_qdes();  // with reference angular positions of previous loop

      // Update velocity vector for wbc
      dq_wbc.head(6) = estimator.getVFilt().head(6);  //  Velocities in base frame (not horizontal frame!)
      dq_wbc.tail(12) = wbcWrapper.get_vdes();  // with reference angular velocities of previous loop

      // Desired position, orientation and velocities of the base
      xgoals.tail(6) = vref_filt_mpc;  // Velocities (in horizontal frame!)

      // Run InvKin + WBC QP
      wbcWrapper.compute(
        q_wbc, dq_wbc, mpcWrapper.get_latest_result().block(12, 0, 12, 1), gait.getCurrentGait().row(0),
        footTrajectoryGenerator.getFootAccelerationBaseFrame(estimator.gethRb() * estimator.getoRh().transpose(),
                                                             Vector3::Zero(), Vector3::Zero()),
        footTrajectoryGenerator.getFootVelocityBaseFrame(estimator.gethRb() * estimator.getoRh().transpose(),
                                                         Vector3::Zero(), Vector3::Zero()),
        footTrajectoryGenerator.getFootPositionBaseFrame(estimator.gethRb() * estimator.getoRh().transpose(),
                                                         estimator.getoTh() + Vector3(0.0, 0.0, params_->h_ref)),
        xgoals);

      // Quantities sent to the control board
      q_des = wbcWrapper.get_qdes();
      v_des = wbcWrapper.get_vdes();
      tau_ff = wbcWrapper.get_tau_ff();
  }

  // Security check
  security_check();

  // Increment loop counter
  k++;
}

void Controller::security_check()
{
  if (error_flag == 0 && !error && !joystick.getStop())
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

  // If something wrong happened in the controller we stick to a security controller
  if (error || joystick.getStop())
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