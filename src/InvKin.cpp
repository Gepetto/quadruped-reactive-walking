#include "qrw/InvKin.hpp"

InvKin::InvKin()
    : invJ(Matrix12::Zero()),
      acc(Matrix118::Zero()),
      x_err(Matrix118::Zero()),
      dx_r(Matrix118::Zero()),
      pfeet_err(Matrix43::Zero()),
      vfeet_ref(Matrix43::Zero()),
      afeet(Matrix43::Zero()),
      posf_(Matrix43::Zero()),
      vf_(Matrix43::Zero()),
      wf_(Matrix43::Zero()),
      af_(Matrix43::Zero()),
      Jf_(Eigen::Matrix<double, 12, 18>::Zero()),
      Jf_tmp_(Eigen::Matrix<double, 6, 18>::Zero()),
      posb_(Vector3::Zero()),
      posb_ref_(Vector3::Zero()),
      posb_err_(Vector3::Zero()),
      rotb_(Matrix3::Identity()),
      rotb_ref_(Matrix3::Identity()),
      rotb_err_(Vector3::Zero()),
      vb_(Vector3::Zero()),
      vb_ref_(Vector3::Zero()),
      wb_(Vector3::Zero()),
      wb_ref_(Vector3::Zero()),
      ab_(Vector6::Zero()),
      abasis(Vector3::Zero()),
      awbasis(Vector3::Zero()),
      Jb_(Eigen::Matrix<double, 6, 18>::Zero()),
      J_(Eigen::Matrix<double, 24, 18>::Zero()),
      invJ_(Eigen::Matrix<double, 18, 24>::Zero()),
      ddq_cmd_(Vector18::Zero()),
      dq_cmd_(Vector18::Zero()),
      q_cmd_(Vector19::Zero()),
      q_step_(Vector18::Zero()) {}

void InvKin::initialize(Params& params) {
  // Params store parameters
  params_ = &params;

  // Path to the robot URDF (TODO: Automatic path)
  const std::string filename =
      std::string("/opt/openrobots/share/example-robot-data/robots/solo_description/robots/solo12.urdf");

  // Build model from urdf (base is not free flyer)
  pinocchio::urdf::buildModel(filename, pinocchio::JointModelFreeFlyer(), model_, false);

  // Construct data from model
  data_ = pinocchio::Data(model_);

  // Update all the quantities of the model
  pinocchio::computeAllTerms(model_, data_, VectorN::Zero(model_.nq), VectorN::Zero(model_.nv));

  // Get feet frame IDs
  foot_ids_[0] = static_cast<int>(model_.getFrameId("FL_FOOT"));  // from long uint to int
  foot_ids_[1] = static_cast<int>(model_.getFrameId("FR_FOOT"));
  foot_ids_[2] = static_cast<int>(model_.getFrameId("HL_FOOT"));
  foot_ids_[3] = static_cast<int>(model_.getFrameId("HR_FOOT"));

  // Get base ID
  base_id_ = static_cast<int>(model_.getFrameId("base_link"));  // from long uint to int

  // Set task gains
  Kp_base_position = Vector3(params_->Kp_base_position.data());
  Kd_base_position = Vector3(params_->Kd_base_position.data());
  Kp_base_orientation = Vector3(params_->Kp_base_orientation.data());
  Kd_base_orientation = Vector3(params_->Kd_base_orientation.data());

}

void InvKin::refreshAndCompute(Matrix14 const& contacts, Matrix43 const& pgoals, Matrix43 const& vgoals,
                               Matrix43 const& agoals) {

  /*std::cout << "pgoals:" << std::endl;
  std::cout << pgoals.row(0) << std::endl;
  std::cout << "posf_" << std::endl;
  std::cout << posf_.row(0) << std::endl;*/

  // Acceleration references for the feet tracking task
  for (int i = 0; i < 4; i++) {
    pfeet_err.row(i) = pgoals.row(i) - posf_.row(i);
    vfeet_ref.row(i) = vgoals.row(i);

    if (contacts(0, i) == 0.0)
    {
      afeet.row(i) = +params_->Kp_flyingfeet * pfeet_err.row(i) - params_->Kd_flyingfeet * (vf_.row(i) - vgoals.row(i)) +
                     agoals.row(i);
    }
    else
    {
      afeet.row(i).setZero();  // Set to 0.0 to disable position/velocity control of feet in contact
    }
    afeet.row(i) -= af_.row(i) + (wf_.row(i)).cross(vf_.row(i));  // - dJ dq
  }

  // Jacobian for the feet tracking task
  J_.block(0, 0, 12, 18) = Jf_.block(0, 0, 12, 18);

  // Acceleration references for the base orientation task
  rotb_err_ = -rotb_ref_ * pinocchio::log3(rotb_ref_.transpose() * rotb_);
  awbasis = Kp_base_orientation.cwiseProduct(rotb_err_) - Kd_base_orientation.cwiseProduct(wb_ - wb_ref_);
  awbasis -= ab_.tail(3);

  // Jacobian for the base orientation task
  J_.block(12, 0, 3, 18) = Jb_.block(3, 0, 3, 18);

  // Acceleration references for the base / feet position task
  posb_err_ = posb_ref_ - posb_;
  abasis = Kp_base_position.cwiseProduct(posb_err_) - Kd_base_position.cwiseProduct(vb_ - vb_ref_);
  abasis -= ab_.head(3) + wb_.cross(vb_);

  // Jacobian for the base / feet position task
  for (int i = 0; i < 4; i++) {
    if (contacts(0, i) == 1.0)  // Feet in contact
    {
      J_.block(15 + 3 * i, 0, 3, 18) = Jb_.block(0, 0, 3, 18) - Jf_.block(3*i, 0, 3, 18);
    }
    else  // Feet not in contact -> not used for this task
    {
      J_.block(15 + 3 * i, 0, 3, 18).setZero();
    }
  }

  // Gather all acceleration references in a single vector
  // Feet tracking task
  for (int i = 0; i < 4; i++) {
    acc.block(0, 3*i, 1, 3) = afeet.row(i);
  }
  // Base orientation task
  acc.block(0, 12, 1, 3) = awbasis.transpose();
  // Base / feet position task
  for (int i = 0; i < 4; i++) {
    acc.block(0, 15+3*i, 1, 3) = abasis.transpose() - afeet.row(i);
  }

  // Gather all task errors in a single vector
  // Feet tracking task
  for (int i = 0; i < 4; i++) {
    x_err.block(0, 3*i, 1, 3) = pfeet_err.row(i);
  }
  // Base orientation task
  x_err.block(0, 12, 1, 3) = rotb_err_.transpose();
  // Base / feet position task
  for (int i = 0; i < 4; i++) {
    x_err.block(0, 15+3*i, 1, 3) = posb_err_.transpose() - pfeet_err.row(i);
  }

  // Gather all task velocity references in a single vector
  // Feet tracking task
  for (int i = 0; i < 4; i++) {
    dx_r.block(0, 3*i, 1, 3) = vfeet_ref.row(i);
  }
  // Base orientation task
  dx_r.block(0, 12, 1, 3) = wb_ref_.transpose();
  // Base / feet position task
  for (int i = 0; i < 4; i++) {
    dx_r.block(0, 15+3*i, 1, 3) = vb_ref_.transpose() - pfeet_err.row(i);
  }

  // Jacobian inversion using damped pseudo inverse
  invJ_ = pseudoInverse(J_);

  // Store data and invert the Jacobian
  /*
  for (int i = 0; i < 4; i++) {
    acc.block(0, 3 * i, 1, 3) = afeet.row(i);
    x_err.block(0, 3 * i, 1, 3) = pfeet_err.row(i);
    dx_r.block(0, 3 * i, 1, 3) = vfeet_ref.row(i);
    invJ.block(3 * i, 3 * i, 3, 3) = Jf_.block(3 * i, 3 * i, 3, 3).inverse();
  }
  */

  // Once Jacobian has been inverted we can get command accelerations, velocities and positions
  ddq_cmd_ = invJ_ * acc.transpose();
  dq_cmd_ = invJ_ * dx_r.transpose();
  q_step_ = invJ_ * x_err.transpose();  // Not a position but a step in position

  /*std::cout << "J" << std::endl << J_ << std::endl;
  std::cout << "invJ" << std::endl << invJ_ << std::endl;
  std::cout << "acc" << std::endl << acc << std::endl;
  std::cout << "dx_r" << std::endl << dx_r << std::endl;
  std::cout << "x_err" << std::endl << x_err << std::endl;
  std::cout << "ddq_cmd" << std::endl << ddq_cmd_ << std::endl;
  std::cout << "dq_cmd" << std::endl << dq_cmd_ << std::endl;
  std::cout << "q_step" << std::endl << q_step_ << std::endl;*/
  
}

void InvKin::run_InvKin(VectorN const& q, VectorN const& dq, MatrixN const& contacts, MatrixN const& pgoals,
                        MatrixN const& vgoals, MatrixN const& agoals, MatrixN const& x_cmd) {
  // std::cout << "run invkin q: " << q << std::endl;
  
  // Update model and data of the robot
  pinocchio::forwardKinematics(model_, data_, q, dq, VectorN::Zero(model_.nv));
  pinocchio::computeJointJacobians(model_, data_);
  pinocchio::updateFramePlacements(model_, data_);

  // Get data required by IK with Pinocchio
  for (int i = 0; i < 4; i++) {
    int idx = foot_ids_[i];
    posf_.row(i) = data_.oMf[idx].translation();
    pinocchio::Motion nu = pinocchio::getFrameVelocity(model_, data_, idx, pinocchio::LOCAL_WORLD_ALIGNED);
    vf_.row(i) = nu.linear();
    wf_.row(i) = nu.angular();
    af_.row(i) = pinocchio::getFrameAcceleration(model_, data_, idx, pinocchio::LOCAL_WORLD_ALIGNED).linear();
    Jf_tmp_.setZero();  // Fill with 0s because getFrameJacobian only acts on the coeffs it changes so the
    // other coeffs keep their previous value instead of being set to 0
    pinocchio::getFrameJacobian(model_, data_, idx, pinocchio::LOCAL_WORLD_ALIGNED, Jf_tmp_);
    Jf_.block(3 * i, 0, 3, 18) = Jf_tmp_.block(0, 0, 3, 18);
  }

  // Update position and velocity of the base
  posb_ = data_.oMf[base_id_].translation();  // Position
  rotb_ = data_.oMf[base_id_].rotation();  // Orientation
  pinocchio::Motion nu = pinocchio::getFrameVelocity(model_, data_, base_id_, pinocchio::LOCAL_WORLD_ALIGNED);
  vb_ = nu.linear();  // Linear velocity
  wb_ = nu.angular();  // Angular velocity
  pinocchio::Motion acc = pinocchio::getFrameAcceleration(model_, data_, base_id_, pinocchio::LOCAL_WORLD_ALIGNED);
  ab_.head(3) = acc.linear();  // Linear acceleration
  ab_.tail(3) = acc.angular();  // Angular acceleration
  pinocchio::getFrameJacobian(model_, data_, base_id_, pinocchio::LOCAL_WORLD_ALIGNED, Jb_);

  // Update reference position and reference velocity of the base
  posb_ref_ = x_cmd.block(0, 0, 3, 1);  // Ref position
  rotb_ref_ = pinocchio::rpy::rpyToMatrix(x_cmd(3, 0), x_cmd(4, 0), x_cmd(5, 0));  // Ref orientation
  vb_ref_ = x_cmd.block(6, 0, 3, 1);  // Ref linear velocity
  wb_ref_ = x_cmd.block(9, 0, 3, 1);  // Ref angular velocity

  /*std::cout << "----" << std::endl;
  std::cout << posf_ << std::endl;
  std::cout << pgoals << std::endl;
  std::cout << Jf_ << std::endl;
  std::cout << posb_ << std::endl;
  std::cout << rotb_ << std::endl;
  std::cout << vb_ << std::endl;
  std::cout << wb_ << std::endl;
  std::cout << ab_ << std::endl;*/

  // IK output for accelerations of actuators (stored in ddq_cmd_)
  // IK output for velocities of actuators (stored in dq_cmd_)
  refreshAndCompute(contacts, pgoals, vgoals, agoals);

  // IK output for positions of actuators
  q_cmd_ = pinocchio::integrate(model_, q, q_step_);

  /*pinocchio::forwardKinematics(model_, data_, q_cmd_, dq, VectorN::Zero(model_.nv));
  pinocchio::computeJointJacobians(model_, data_);
  pinocchio::updateFramePlacements(model_, data_);
  std::cout << "pos after step" << std::endl;
  std::cout << data_.oMf[foot_ids_[0]].translation()  << std::endl;*/

  /*std::cout << "q: " << q << std::endl;
  std::cout << "q_step_: " << q_step_ << std::endl;
  std::cout << " q_cmd_: " <<  q_cmd_ << std::endl;*/
 

}
