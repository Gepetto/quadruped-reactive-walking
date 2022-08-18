#include "qrw/WbcWrapper.hpp"

#include <example-robot-data/path.hpp>
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/math/rpy.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/compute-all-terms.hpp"
#include "pinocchio/algorithm/rnea.hpp"

WbcWrapper::WbcWrapper()
    : M_(Eigen::Matrix<double, 18, 18>::Zero()),
      Jc_(Eigen::Matrix<double, 12, 6>::Zero()),
      k_since_contact_(RowVector4::Zero()),
      bdes_(Vector7::Zero()),
      qdes_(Vector12::Zero()),
      vdes_(Vector12::Zero()),
      tau_ff_(Vector12::Zero()),
      q_wbc_(Vector19::Zero()),
      dq_wbc_(Vector18::Zero()),
      ddq_cmd_(Vector18::Zero()),
      dq_cmd_(Vector18::Zero()),
      q_cmd_(Vector19::Zero()),
      f_with_delta_(Vector12::Zero()),
      ddq_with_delta_(Vector18::Zero()),
      nle_(Vector6::Zero()),
      log_feet_pos_target(Matrix34::Zero()),
      log_feet_vel_target(Matrix34::Zero()),
      log_feet_acc_target(Matrix34::Zero()),
      k_log_(0),
      enable_comp_forces_(false) {}

void WbcWrapper::initialize(Params &params) {
  // Params store parameters
  params_ = &params;

  // Set if compensation forces should be used or not
  enable_comp_forces_ = params_->enable_comp_forces;

  // Path to the robot URDF
  const std::string filename = std::string(EXAMPLE_ROBOT_DATA_MODEL_DIR "/solo_description/robots/solo12.urdf");

  // Build model from urdf (base is not free flyer)
  pinocchio::urdf::buildModel(filename, pinocchio::JointModelFreeFlyer(), model_, false);

  // Construct data from model
  data_ = pinocchio::Data(model_);

  // Update all the quantities of the model
  VectorN q_tmp = VectorN::Zero(model_.nq);
  q_tmp(6, 0) = 1.0;  // Quaternion (0, 0, 0, 1)
  pinocchio::computeAllTerms(model_, data_, q_tmp, VectorN::Zero(model_.nv));
  // TODO ADD INIT POSITION FOR ACTUATORS

  // Initialize inverse kinematic and box QP solvers
  invkin_ = new InvKin();
  invkin_->initialize(params);
  box_qp_ = new QPWBC();
  box_qp_->initialize(params);

  // Initialize quaternion
  q_wbc_(6, 0) = 1.0;

  // Initialize joint positions
  qdes_.tail(12) = Vector12(params_->q_init.data());

  // Compute the upper triangular part of the joint space inertia matrix M by using the Composite Rigid Body Algorithm
  // Result is stored in data_.M
  pinocchio::crba(model_, data_, q_wbc_);

  // Make mass matrix symetric
  data_.M.triangularView<Eigen::StrictlyLower>() = data_.M.transpose().triangularView<Eigen::StrictlyLower>();
}

void WbcWrapper::compute(VectorN const &q, VectorN const &dq, VectorN const &f_cmd, MatrixN const &contacts,
                         MatrixN const &pgoals, MatrixN const &vgoals, MatrixN const &agoals, VectorN const &xgoals) {
  if (f_cmd.rows() != 12) {
    throw std::runtime_error("f_cmd should be a vector of size 12");
  }

  //  Update nb of iterations since contact
  k_since_contact_ += contacts;                                // Increment feet in stance phase
  k_since_contact_ = k_since_contact_.cwiseProduct(contacts);  // Reset feet in swing phase

  // Store target positions, velocities and acceleration for logging purpose
  log_feet_pos_target = pgoals;
  log_feet_vel_target = vgoals;
  log_feet_acc_target = agoals;

  // Retrieve configuration data
  q_wbc_.head(3) = q.head(3);
  q_wbc_.block(3, 0, 4, 1) =
      pinocchio::SE3::Quaternion(pinocchio::rpy::rpyToMatrix(q(3, 0), q(4, 0), q(5, 0))).coeffs();  // Roll, Pitch
  q_wbc_.tail(12) = q.tail(12);                                                                     // Encoders

  // Retrieve velocity data
  dq_wbc_ = dq;

  // Compute the upper triangular part of the joint space inertia matrix M by using the Composite Rigid Body Algorithm
  // Result is stored in data_.M
  pinocchio::crba(model_, data_, q_wbc_);

  // Make mass matrix symetric
  data_.M.triangularView<Eigen::StrictlyLower>() = data_.M.transpose().triangularView<Eigen::StrictlyLower>();

  // Compute Inverse Kinematics
  invkin_->run(q_wbc_, dq_wbc_, contacts, pgoals.transpose(), vgoals.transpose(), agoals.transpose(), xgoals);
  ddq_cmd_ = invkin_->get_ddq_cmd();
  dq_cmd_ = invkin_->get_dq_cmd();
  q_cmd_ = invkin_->get_q_cmd();

  // TODO: Check if we can save time by switching MatrixXd to defined sized vector since they are
  // not called from python anymore

  // Retrieve feet jacobian  // TODO: Retrieve it in one go to avoid having Jc_ and Jc_u_
  for (int i = 0; i < 4; i++) {
    if (contacts(0, i)) {
      Jc_.block(3 * i, 0, 3, 6) = invkin_->get_Jf().block(3 * i, 0, 3, 6);
    } else {
      Jc_.block(3 * i, 0, 3, 6).setZero();
    }
  }

  Eigen::Matrix<double, 12, 12> Jc_u_ = Eigen::Matrix<double, 12, 12>::Zero();
  for (int i = 0; i < 4; i++) {
    if (contacts(0, i)) {
      Jc_u_.block(3 * i, 0, 3, 12) = invkin_->get_Jf().block(3 * i, 6, 3, 12);
    } else {
      Jc_u_.block(3 * i, 0, 3, 12).setZero();
    }
  }

  // Compute the inverse dynamics, aka the joint torques according to the current state of the system,
  // the desired joint accelerations and the external forces, using the Recursive Newton Euler Algorithm.
  // Result is stored in data_.tau
  Vector12 f_compensation;
  if (!enable_comp_forces_) {
    pinocchio::rnea(model_, data_, q_wbc_, dq_wbc_, ddq_cmd_);
    f_compensation = Vector12::Zero();
  } else {
    Vector18 ddq_test = Vector18::Zero();
    ddq_test.head(6) = ddq_cmd_.head(6);
    pinocchio::rnea(model_, data_, q_wbc_, dq_wbc_, ddq_test);
    Vector6 RNEA_without_joints = data_.tau.head(6);
    pinocchio::rnea(model_, data_, q_wbc_, dq_wbc_, VectorN::Zero(model_.nv));
    Vector6 RNEA_NLE = data_.tau.head(6);
    RNEA_NLE(2, 0) -= 9.81 * data_.mass[0];
    pinocchio::rnea(model_, data_, q_wbc_, dq_wbc_, ddq_cmd_);
    f_compensation = pseudoInverse(Jc_.transpose()) * (data_.tau.head(6) - RNEA_without_joints + RNEA_NLE);
  }

  /*std::cout << "M inertia" << std::endl;
  std::cout << data_.M << std::endl;*/
  /*std::cout << "Jc" << std::endl;
  std::cout << Jc_ << std::endl;
  std::cout << "f_cmd" << std::endl;
  std::cout << f_cmd << std::endl;
  std::cout << "rnea" << std::endl;
  std::cout << data_.tau.head(6) << std::endl;
  std::cout << "k_since" << std::endl;
  std::cout << k_since_contact_ << std::endl;*/

  // std::cout << "Force compensation " << std::endl;

  /*for (int i = 0; i < 4; i++) {
    f_compensation(3*i+2, 0) = 0.0;
  }*/
  // std::cout << f_compensation << std::endl;

  // std::cout << "agoals " << std::endl << agoals << std::endl;
  // std::cout << "ddq_cmd_bis " << std::endl << ddq_cmd_.transpose() << std::endl;

  // std::cout << "M : " << std::endl << data_.M.block(0, 0, 3, 18) << std::endl;
  // std::cout << "ddq: " << std::endl << ddq_cmd_.transpose() << std::endl;

  /*
  std::cout << "-- BEFORE QP PROBLEM --" << std::endl;
  std::cout << "M ddq_u: " << std::endl << (data_.M.block(0, 0, 3, 6) * ddq_cmd_.head(6)).transpose() << std::endl;
  std::cout << "M ddq_a: " << std::endl << (data_.M.block(0, 6, 3, 12) * ddq_cmd_.tail(12)).transpose() << std::endl;
  pinocchio::rnea(model_, data_, q_wbc_, dq_wbc_, VectorN::Zero(model_.nv));
  std::cout << "Non linear effects: " << std::endl << data_.tau.head(6).transpose() << std::endl;
  std::cout << "JcT f_cmd + f_comp: " << std::endl << (Jc_.transpose() * (f_cmd + f_compensation)).transpose() <<
  std::endl; std::cout << "JcT f_comp: " << std::endl << (Jc_.transpose() * (f_compensation)).transpose() << std::endl;
  */

  // Solve the QP problem
  pinocchio::rnea(model_, data_, q_wbc_, dq_wbc_, ddq_cmd_);
  box_qp_->run(data_.M, Jc_, ddq_cmd_, f_cmd + f_compensation, data_.tau.head(6), k_since_contact_);

  // Add to reference quantities the deltas found by the QP solver
  f_with_delta_ = f_cmd + f_compensation + box_qp_->get_f_res();
  ddq_with_delta_.head(6) = ddq_cmd_.head(6) + box_qp_->get_ddq_res();
  ddq_with_delta_.tail(12) = ddq_cmd_.tail(12);

  // DEBUG INERTIA AND NON LINEAR EFFECTS

  Vector6 left = data_.M.block(0, 0, 6, 6) * box_qp_->get_ddq_res() - Jc_.transpose() * box_qp_->get_f_res();
  Vector6 right = -data_.tau.head(6) + Jc_.transpose() * (f_cmd + f_compensation);
  Vector6 tmp_RNEA = data_.tau.head(6);

  // std::cout << "RNEA: " << std::endl << data_.tau.head(6).transpose() << std::endl;
  // std::cout << "left: " << std::endl << left.transpose() << std::endl;
  // std::cout << "right: " << std::endl << right.transpose() << std::endl;
  // std::cout << "M: " << std::endl << data_.M.block(0, 0, 6, 6) << std::endl;
  // std::cout << "JcT: " << std::endl << Jc_.transpose() << std::endl;
  // std::cout << "M: " << std::endl << data_.M.block(0, 0, 3, 18) << std::endl;
  /*
  pinocchio::rnea(model_, data_, q_wbc_, dq_wbc_, VectorN::Zero(model_.nv));
  Vector6 tmp_NLE = data_.tau.head(6);

  // std::cout << "NLE: " << std::endl << data_.tau.head(6).transpose() << std::endl;
  // std::cout << "M DDQ: " << std::endl << (tmp_RNEA - data_.tau.head(6)).transpose() << std::endl;
  // std::cout << "JcT f_cmd: " << std::endl << (Jc_.transpose() * (f_cmd + f_compensation)).transpose() << std::endl;
  // std::cout << "Gravity ?: " << std::endl <<  (data_.M * ddq_cmd_ - data_.tau.head(6)).transpose() << std::endl;

  Mddq = tmp_RNEA - data_.tau.head(6);
  NLE = data_.tau.head(6);
  JcTf = Jc_.transpose() * (f_cmd + f_compensation);
  nle_ = data_.tau.head(6);
  */

  // Compute joint torques from contact forces and desired accelerations
  pinocchio::rnea(model_, data_, q_wbc_, dq_wbc_, ddq_with_delta_);

  /*std::cout << "NLE Delta: " << std::endl << tmp_NLE.transpose() << std::endl;
  std::cout << "M DDQ Delta: " << std::endl << (data_.tau.head(6) - tmp_NLE).transpose() << std::endl;
  std::cout << "JcT f_cmd Delta: " << std::endl << (Jc_.transpose() * f_with_delta_).transpose() << std::endl;*/

  /*std::cout << "rnea delta" << std::endl;
  std::cout << data_.tau.tail(12) << std::endl;
  std::cout << "ddq del" << std::endl;
  std::cout << ddq_with_delta_ << std::endl;
  std::cout << "f del" << std::endl;
  std::cout << f_with_delta_ << std::endl;
  std::cout << "Jf" << std::endl;
  std::cout << invkin_->get_Jf().block(0, 6, 12, 12).transpose() << std::endl;*/

  // std::cout << " -- " << std::endl << invkin_->get_Jf().block(0, 6, 12, 12) << std::endl << " -- " << std::endl <<
  // Jc_u_ << std::endl;

  tau_ff_ = data_.tau.tail(12) - Jc_u_.transpose() * f_with_delta_;
  // tau_ff_ = - Jc_u_.transpose() * (f_cmd + f_compensation);

  // Retrieve desired positions and velocities
  vdes_ = invkin_->get_dq_cmd().tail(12);

  // std::cout << "GET:" << invkin_->get_q_cmd() << std::endl;

  qdes_ = invkin_->get_q_cmd().tail(12);
  bdes_ = invkin_->get_q_cmd().head(7);

  // pinocchio::rnea(model_, data_, q_wbc_, dq_wbc_, ddq_with_delta_);
  // Vector6 tau_left = data_.tau.head(6); // data_.M * ddq_with_delta_.head(6) - Jc_.transpose() * f_with_delta_; //
  // Vector6 tau_right = Jc_.transpose() * f_with_delta_;
  // std::cout << "tau_left: " << std::endl << tau_left.transpose() << std::endl;
  // std::cout << "tau_right: " << std::endl << tau_right.transpose() << std::endl;

  // Mddq_out = data_.tau.head(6) - NLE;
  // JcTf_out = Jc_.transpose() * f_with_delta_;

  /*std::cout << vdes_.transpose() << std::endl;
  std::cout << qdes_.transpose() << std::endl;*/

  /*std::cout << "----" << std::endl;
  std::cout << qdes_.transpose() << std::endl;
  std::cout << vdes_.transpose() << std::endl;
  std::cout << tau_ff_.transpose() << std::endl;*/

  // Compute joint torques from contact forces and desired accelerations
  /*Vector18 ddq_test = Vector18::Zero();
  ddq_test.head(6) = ddq_with_delta_.head(6);
  pinocchio::rnea(model_, data_, q_wbc_, dq_wbc_, ddq_test);
  std::cout << "M DDQ Delta Bis: " << std::endl << (data_.tau.head(6) - tmp_NLE).transpose() << std::endl;*/

  /*
  std::cout << "-- AFTER QP PROBLEM --" << std::endl;
  std::cout << "M ddq_u: " << std::endl << (data_.M.block(0, 0, 3, 6) * ddq_with_delta_.head(6)).transpose() <<
  std::endl; std::cout << "M ddq_a: " << std::endl << (data_.M.block(0, 6, 3, 12) *
  ddq_with_delta_.tail(12)).transpose() << std::endl; pinocchio::rnea(model_, data_, q_wbc_, dq_wbc_,
  VectorN::Zero(model_.nv)); std::cout << "Non linear effects: " << std::endl << data_.tau.head(6).transpose() <<
  std::endl; std::cout << "JcT f_cmd: " << std::endl << (Jc_.transpose() * f_with_delta_).transpose() << std::endl;

  std::cout << "LEFT " << (tmp_RNEA.head(3) + data_.M.block(0, 0, 3, 6) * box_qp_->get_ddq_res()).transpose() <<
  std::endl;
  */

  // Increment log counter
  k_log_++;
}
