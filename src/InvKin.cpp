#include "qrw/InvKin.hpp"

#include <example-robot-data/path.hpp>

#include "pinocchio/algorithm/compute-all-terms.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/math/rpy.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/spatial/explog.hpp"

InvKin::InvKin()
    : invJ(Matrix12::Zero()),
      acc(Eigen::Matrix<double, 1, 30>::Zero()),
      x_err(Eigen::Matrix<double, 1, 30>::Zero()),
      dx_r(Eigen::Matrix<double, 1, 30>::Zero()),
      pfeet_err(Matrix43::Zero()),
      vfeet_ref(Matrix43::Zero()),
      afeet(Matrix43::Zero()),
      posf_(Matrix43::Zero()),
      vf_(Matrix43::Zero()),
      wf_(Matrix43::Zero()),
      af_(Matrix43::Zero()),
      dJdq_(Matrix43::Zero()),
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
      afeet_contacts_(Matrix43::Zero()),
      Jb_(Eigen::Matrix<double, 6, 18>::Zero()),
      J_(Eigen::Matrix<double, 30, 18>::Zero()),
      invJ_(Eigen::Matrix<double, 18, 30>::Zero()),
      ddq_cmd_(Vector18::Zero()),
      dq_cmd_(Vector18::Zero()),
      q_cmd_(Vector19::Zero()),
      q_step_(Vector18::Zero()) {}

void InvKin::initialize(Params& params) {
  // Params store parameters
  params_ = &params;

  // Path to the robot URDF
  const std::string filename = std::string(EXAMPLE_ROBOT_DATA_MODEL_DIR "/solo_description/robots/solo12.urdf");

  // Build model from urdf (base is not free flyer)
  pinocchio::urdf::buildModel(filename, pinocchio::JointModelFreeFlyer(), model_, false);

  // Construct data from model
  data_ = pinocchio::Data(model_);

  // Update all the quantities of the model
  pinocchio::computeAllTerms(model_, data_, VectorN::Zero(model_.nq), VectorN::Zero(model_.nv));

  pinocchio::urdf::buildModel(filename, pinocchio::JointModelFreeFlyer(), model_dJdq_, false);
  data_dJdq_ = pinocchio::Data(model_dJdq_);
  pinocchio::computeAllTerms(model_dJdq_, data_dJdq_, VectorN::Zero(model_dJdq_.nq), VectorN::Zero(model_dJdq_.nv));

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
  w_tasks = Vector8(params_->w_tasks.data());
}

void InvKin::refreshAndCompute(RowVector4 const& contacts, Matrix43 const& pgoals, Matrix43 const& vgoals,
                               Matrix43 const& agoals) {
  std::cout << std::fixed;
  std::cout << std::setprecision(5);

  /*std::cout << "pgoals:" << std::endl;
  std::cout << pgoals << std::endl;
  std::cout << "posf_" << std::endl;
  std::cout << posf_ << std::endl;*/

  /*std::cout << "vf_" << std::endl;
  std::cout << vf_ << std::endl;*/

  /////
  // Compute tasks accelerations and Jacobians
  /////

  // Accelerations references for the base / feet position task
  for (int i = 0; i < 4; i++) {
    // Feet acceleration
    pfeet_err.row(i) = pgoals.row(i) - posf_.row(i);
    vfeet_ref.row(i) = vgoals.row(i);
    afeet.row(i) = +params_->Kp_flyingfeet * pfeet_err.row(i) +
                   params_->Kd_flyingfeet * (vfeet_ref.row(i) - vf_.row(i)) + agoals.row(i);

    // std::cout << "1: " << afeet.row(i) << std::endl;
    afeet.row(i) -= af_.row(i) + (wf_.row(i)).cross(vf_.row(i));
    // std::cout << "2: " << afeet.row(i) << std::endl;

    // Subtract base acceleration
    afeet.row(i) -= (params_->Kd_flyingfeet * (vb_ref_ - vb_) - (ab_.head(3) + wb_.cross(vb_))).transpose();
    // std::cout << "3: " << afeet.row(i) << std::endl;
    /*std::cout << vb_ref_.transpose() << std::endl;
    std::cout << vb_.transpose() << std::endl;
    std::cout << wb_.transpose() << std::endl;
    std::cout << ab_.head(3).transpose() << std::endl;
    std::cout << (vb_ref_ - vb_).transpose() << std::endl;
    std::cout << (wb_.cross(vb_)).transpose() << std::endl;*/
    // std::cout << "---" << std::endl;
  }

  // Jacobian for the base / feet position task
  for (int i = 0; i < 4; i++) {
    J_.block(6 + 3 * i, 0, 3, 18) = Jf_.block(3 * i, 0, 3, 18) - Jb_.block(0, 0, 3, 18);
  }

  // Acceleration references for the base linear velocity task
  posb_err_ = Vector3::Zero();  // No tracking in x, y, z
  abasis = Kd_base_position.cwiseProduct(vb_ref_ - vb_);
  abasis -= ab_.head(3) + wb_.cross(vb_);

  // Jacobian for the base linear velocity task
  J_.block(0, 0, 3, 18) = Jb_.block(0, 0, 3, 18);

  // Acceleration references for the base orientation task
  rotb_err_ = -rotb_ref_ * pinocchio::log3(rotb_ref_.transpose() * rotb_);
  rotb_err_(2, 0) = 0.0;  // No tracking in yaw
  awbasis = Kp_base_orientation.cwiseProduct(rotb_err_) +
            Kd_base_orientation.cwiseProduct(wb_ref_ - wb_);  // Roll, Pitch, Yaw
  awbasis -= ab_.tail(3);

  // Jacobian for the base orientation task
  J_.block(3, 0, 3, 18) = Jb_.block(3, 0, 3, 18);

  // Acceleration references for the non-moving contact task
  /*int cpt = 0;
  for (int i = 0; i < 4; i++) {
    if (contacts(0, i) == 1.0) {
      afeet_contacts_.row(cpt) = + params_->Kp_flyingfeet * pfeet_err.row(i)
                                 + params_->Kd_flyingfeet * (vfeet_ref.row(i) - vf_.row(i))
                                 + agoals.row(i);
      afeet_contacts_.row(cpt) -= af_.row(i) + (wf_.row(i)).cross(vf_.row(i));
      cpt++;
    }
  }
  for (int i = cpt; i < 4; i++) {  // Set to 0s the lines that are not used
    afeet_contacts_.row(i).setZero();
  }*/
  for (int i = 0; i < 4; i++) {
    if (contacts(0, i) == 1.0) {
      afeet_contacts_.row(i) = +params_->Kp_flyingfeet * pfeet_err.row(i) +
                               params_->Kd_flyingfeet * (vfeet_ref.row(i) - vf_.row(i)) + agoals.row(i);
      afeet_contacts_.row(i) -= af_.row(i) + (wf_.row(i)).cross(vf_.row(i));
    } else {
      afeet_contacts_.row(i).setZero();
    }
  }

  // Jacobian for the non-moving contact task
  for (int i = 0; i < 4; i++) {
    if (contacts(0, i) == 1.0) {  // Store Jacobian of feet in contact
      J_.block(18 + 3 * i, 0, 3, 18) = Jf_.block(3 * i, 0, 3, 18);
    } else {
      J_.block(18 + 3 * i, 0, 3, 18).setZero();
    }
  }
  /*cpt = 0;
  for (int i = 0; i < 4; i++) {
    if (contacts(0, i) == 1.0) {  // Store Jacobian of feet in contact
      J_.block(18 + 3 * cpt, 0, 3, 18) = Jf_.block(3*i, 0, 3, 18);
      cpt++;
    }
  }
  for (int i = cpt; i < 4; i++) {  // Set to 0s the lines that are not used
    J_.block(18 + 3 * i, 0, 3, 18).setZero();
  }*/

  /////
  // Fill acceleration reference vector
  /////

  // Feet / base tracking task
  for (int i = 0; i < 4; i++) {
    acc.block(0, 6 + 3 * i, 1, 3) = afeet.row(i);
  }
  // Base linear task
  acc.block(0, 0, 1, 3) = abasis.transpose();
  // Base angular task
  acc.block(0, 3, 1, 3) = awbasis.transpose();
  // Non-moving contact task
  for (int i = 0; i < 4; i++) {
    acc.block(0, 18 + 3 * i, 1, 3) = afeet_contacts_.row(i);
  }

  /////
  // Fill velocity reference vector
  /////

  // Feet / base tracking task
  for (int i = 0; i < 4; i++) {
    dx_r.block(0, 6 + 3 * i, 1, 3) = vfeet_ref.row(i) - vb_ref_.transpose();
  }
  // Base linear task
  dx_r.block(0, 0, 1, 3) = vb_ref_.transpose();
  // Base angular task
  dx_r.block(0, 3, 1, 3) = wb_ref_.transpose();
  // Non-moving contact task
  for (int i = 0; i < 4; i++) {
    dx_r.block(0, 18 + 3 * i, 1, 3) = vfeet_ref.row(i);
  }

  /////
  // Fill position reference vector
  /////

  // Feet / base tracking task
  for (int i = 0; i < 4; i++) {
    x_err.block(0, 6 + 3 * i, 1, 3) = pfeet_err.row(i) - posb_err_.transpose();
  }
  // Base linear task
  x_err.block(0, 0, 1, 3) = posb_err_.transpose();
  // Base angular task
  x_err.block(0, 3, 1, 3) = rotb_err_.transpose();
  // Non-moving contact task
  for (int i = 0; i < 4; i++) {
    x_err.block(0, 18 + 3 * i, 1, 3) = pfeet_err.row(i);
  }

  /////
  // Apply tasks weights
  /////

  // Product with tasks weights for Jacobian
  J_.block(6, 0, 12, 18) *= w_tasks(0, 0);
  for (int i = 0; i < 6; i++) {
    J_.row(i) *= w_tasks(1 + i, 0);
  }
  J_.block(18, 0, 12, 18) *= w_tasks(7, 0);

  // Product with tasks weights for acc references
  acc.block(6, 0, 1, 12) *= w_tasks(0, 0);
  for (int i = 0; i < 6; i++) {
    acc(0, i) *= w_tasks(1 + i, 0);
  }
  acc.tail(12) *= w_tasks(7, 0);

  // Product with tasks weights for vel references
  dx_r.block(6, 0, 1, 12) *= w_tasks(0, 0);
  for (int i = 0; i < 6; i++) {
    dx_r(0, i) *= w_tasks(1 + i, 0);
  }
  dx_r.tail(12) *= w_tasks(7, 0);

  // Product with tasks weights for pos references
  x_err.block(6, 0, 1, 12) *= w_tasks(0, 0);
  for (int i = 0; i < 6; i++) {
    x_err(0, i) *= w_tasks(1 + i, 0);
  }
  x_err.tail(12) *= w_tasks(7, 0);

  /////
  // Jacobian inversion
  /////

  // Using damped pseudo inverse
  invJ_ = pseudoInverse(J_);

  /////
  // Compute command accelerations, velocities and positions
  /////

  ddq_cmd_ = invJ_ * acc.transpose();
  dq_cmd_ = invJ_ * dx_r.transpose();
  q_step_ = invJ_ * x_err.transpose();  // Not a position but a step in position

  // std::cout << "pfeet_err" << std::endl << pfeet_err << std::endl;

  /*std::cout << "acc: " << std::endl << acc << std::endl;
  std::cout << "J_   : " << std::endl << J_ << std::endl;
  std::cout << "invJ_: " << std::endl << invJ_ << std::endl;
  std::cout << "ddq_cmd_: " << std::endl << ddq_cmd_.transpose() << std::endl;*/

  /*std::cout << "J" << std::endl << J_ << std::endl;
  std::cout << "invJ" << std::endl << invJ_ << std::endl;*/
  /*std::cout << "acc" << std::endl << acc << std::endl;
  std::cout << "dx_r" << std::endl << dx_r << std::endl;
  std::cout << "x_err" << std::endl << x_err << std::endl;*/
  /*std::cout << "ddq_cmd" << std::endl << ddq_cmd_ << std::endl;
  std::cout << "dq_cmd" << std::endl << dq_cmd_ << std::endl;
  std::cout << "q_step" << std::endl << q_step_ << std::endl;*/

  // Store data
  /*
  for (int i = 0; i < 4; i++) {
    acc.block(0, 3 * i, 1, 3) = afeet.row(i);
    x_err.block(0, 3 * i, 1, 3) = pfeet_err.row(i);
    dx_r.block(0, 3 * i, 1, 3) = vfeet_ref.row(i);
    invJ.block(3 * i, 3 * i, 3, 3) = Jf_.block(3 * i, 3 * i, 3, 3).inverse();
  }
  */
}

void InvKin::run(VectorN const& q, VectorN const& dq, MatrixN const& contacts, MatrixN const& pgoals,
                        MatrixN const& vgoals, MatrixN const& agoals, MatrixN const& x_cmd) {
  // std::cout << "run invkin q: " << q << std::endl;

  // Update model and data of the robot
  pinocchio::forwardKinematics(model_, data_, q, dq, VectorN::Zero(model_.nv));
  pinocchio::computeJointJacobians(model_, data_);
  pinocchio::computeJointJacobiansTimeVariation(model_, data_, q, dq);
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
  rotb_ = data_.oMf[base_id_].rotation();     // Orientation
  pinocchio::Motion nu = pinocchio::getFrameVelocity(model_, data_, base_id_, pinocchio::LOCAL_WORLD_ALIGNED);
  vb_ = nu.linear();   // Linear velocity
  wb_ = nu.angular();  // Angular velocity
  /*std::cout << "NU" << std::endl;
  std::cout << q.transpose() << std::endl;
  std::cout << dq.transpose() << std::endl;
  std::cout << nu.linear().transpose() << std::endl;
  std::cout << nu.angular().transpose() << std::endl;*/

  pinocchio::Motion acc = pinocchio::getFrameAcceleration(model_, data_, base_id_, pinocchio::LOCAL_WORLD_ALIGNED);
  ab_.head(3) = acc.linear();   // Linear acceleration
  ab_.tail(3) = acc.angular();  // Angular acceleration
  pinocchio::getFrameJacobian(model_, data_, base_id_, pinocchio::LOCAL_WORLD_ALIGNED, Jb_);

  // std::cout << "Jb_: " << std::endl << Jb_ << std::endl;

  // Update reference position and reference velocity of the base
  posb_ref_ = x_cmd.block(0, 0, 3, 1);                                             // Ref position
  rotb_ref_ = pinocchio::rpy::rpyToMatrix(x_cmd(3, 0), x_cmd(4, 0), x_cmd(5, 0));  // Ref orientation
  vb_ref_ = x_cmd.block(6, 0, 3, 1);                                               // Ref linear velocity
  wb_ref_ = x_cmd.block(9, 0, 3, 1);                                               // Ref angular velocity

  /*std::cout << "----" << std::endl;
  std::cout << posf_ << std::endl;
  std::cout << pgoals << std::endl;
  std::cout << Jf_ << std::endl;
  std::cout << posb_ << std::endl;
  std::cout << rotb_ << std::endl;
  std::cout << vb_ << std::endl;
  std::cout << wb_ << std::endl;
  std::cout << ab_ << std::endl;*/

  /*
  Eigen::Matrix<double, 6, 18> dJf = Eigen::Matrix<double, 6, 18>::Zero();
  std::cout << "analysis: " << std::endl;
  std::cout << pinocchio::getJointJacobianTimeVariation(model_, data_, foot_ids_[0], pinocchio::LOCAL_WORLD_ALIGNED,
  dJf) << std::endl; std::cout << "---" << std::endl; std::cout << dq.transpose() << std::endl; std::cout << "---" <<
  std::endl; std::cout << dJf * dq << std::endl; std::cout << "---" << std::endl;*/

  // Eigen::Matrix<double, 6, 18> dJf = Eigen::Matrix<double, 6, 18>::Zero();
  /*
  for (int i = 0; i < 4; i++) {
    Jf_tmp_.setZero();
    pinocchio::getFrameJacobianTimeVariation(model_, data_, foot_ids_[i], pinocchio::LOCAL_WORLD_ALIGNED, Jf_tmp_);
    dJdq_.row(i) = (Jf_tmp_.block(0, 0, 3, 18) * dq).transpose();
  }
  */
  /*std::cout << "Other: " << dJdq_.row(0) << std::endl;
  std::cout << "Other: " << dJdq_.row(1) << std::endl;
  std::cout << "Other: " << dJdq_.row(2) << std::endl;
  std::cout << "Other: " << dJdq_.row(3) << std::endl;*/

  /*
  Jf_tmp_.setZero();
  pinocchio::getFrameJacobianTimeVariation(model_, data_, base_id_, pinocchio::LOCAL_WORLD_ALIGNED, Jf_tmp_);
  */
  // std::cout << "Base dJdq: " << (Jf_tmp_ * dq).transpose() << std::endl;

  /*
  pinocchio::forwardKinematics(model_dJdq_, data_dJdq_, q, dq, VectorN::Zero(model_.nv));
  pinocchio::updateFramePlacements(model_dJdq_, data_dJdq_);
  pinocchio::rnea(model_dJdq_, data_dJdq_, q, dq, VectorN::Zero(model_dJdq_.nv));
  for (int i = 0; i < 4; i++) {
    pinocchio::Motion a = data_dJdq_.a[foot_joints_ids_[i]];
    pinocchio::Motion v = data_dJdq_.v[foot_joints_ids_[i]];
    // pinocchio::FrameVector foot = model_dJdq_.frames[foot_ids_[0]]
    pinocchio::SE3 kMf = (model_dJdq_.frames[foot_ids_[i]]).placement;
    pinocchio::SE3 wMf = data_dJdq_.oMf[foot_ids_[i]];
    // f_a = kMf.actInv(a)
    // f_v = kMf.actInv(v)
    Vector3 f_a3 = kMf.actInv(a).linear() + (kMf.actInv(v).angular()).cross(kMf.actInv(v).linear());
    Vector3 w_a3 = wMf.rotation() * f_a3;
    // std::cout << "f_a3: " << f_a3.transpose() << std::endl;
    // std::cout << "w_a3: " << w_a3.transpose() << std::endl;
    dJdq_.row(i) = w_a3.transpose();
  }
  */

  // IK output for accelerations of actuators (stored in ddq_cmd_)
  // IK output for velocities of actuators (stored in dq_cmd_)
  refreshAndCompute(contacts, pgoals, vgoals, agoals);

  // IK output for positions of actuators
  q_cmd_ = pinocchio::integrate(model_, q, q_step_);

  /*pinocchio::forwardKinematics(model_, data_, q_cmd_, dq_cmd_, ddq_cmd_);
  pinocchio::computeJointJacobians(model_, data_);
  pinocchio::updateFramePlacements(model_, data_);
  std::cout << "pos after step" << std::endl;
  std::cout << data_.oMf[foot_ids_[0]].translation()  << std::endl;
  std::cout << "vel after step" << std::endl;
  std::cout << pinocchio::getFrameVelocity(model_, data_, foot_ids_[0], pinocchio::LOCAL_WORLD_ALIGNED).linear() <<
  std::endl; std::cout << "acc after step" << std::endl; std::cout << pinocchio::getFrameAcceleration(model_, data_,
  foot_ids_[0], pinocchio::LOCAL_WORLD_ALIGNED).linear() << std::endl;*/

  /*std::cout << "q: " << q << std::endl;
  std::cout << "q_step_: " << q_step_ << std::endl;
  std::cout << " q_cmd_: " <<  q_cmd_ << std::endl;*/

  /*pinocchio::forwardKinematics(model_, data_, q, dq, ddq_cmd_);
  pinocchio::updateFramePlacements(model_, data_);
  std::cout << "Feet velocities after IK:" << std::endl;
  for (int i = 0; i < 4; i++) {
    int idx = foot_ids_[i];
    pinocchio::Motion nu = pinocchio::getFrameVelocity(model_, data_, idx, pinocchio::LOCAL_WORLD_ALIGNED);
    std::cout << nu.linear() << std::endl;
  }
  std::cout << "Feet accelerations after IK:" << std::endl;
  for (int i = 0; i < 4; i++) {
    int idx = foot_ids_[i];
    pinocchio::Motion acc = pinocchio::getFrameClassicalAcceleration(model_, data_, idx,
  pinocchio::LOCAL_WORLD_ALIGNED); std::cout << acc.linear() << std::endl;
  }*/
}
