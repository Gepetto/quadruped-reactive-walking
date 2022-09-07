#include "qrw/FootstepPlanner.hpp"

#include <example-robot-data/path.hpp>

#include "pinocchio/algorithm/compute-all-terms.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/math/rpy.hpp"
#include "pinocchio/parsers/urdf.hpp"

FootstepPlanner::FootstepPlanner()
    : gait_(NULL),
      dt(0.0),
      dt_wbc(0.0),
      h_ref(0.0),
      k_mpc_(0),
      g(9.81),
      L(0.25),
      nextFootstep_(Matrix34::Zero()),
      footsteps_(),
      previousGait_(RowVector4::Zero()),
      previousHeight_(RowVector4::Zero()),
      Rz(MatrixN::Zero(3, 3)),
      dt_cum(),
      yaws(),
      dx(),
      dy(),
      q_dxdy(Vector3::Zero()),
      RPY_(Vector3::Zero()),
      pos_feet_(Matrix34::Zero()),
      q_FK_(Vector19::Zero()) {
  // Empty
}

void FootstepPlanner::initialize(Params& params, Gait& gaitIn) {
  params_ = &params;
  dt = params.dt_mpc;
  dt_wbc = params.dt_wbc;
  h_ref = params.h_ref;
  k_mpc_ = static_cast<int>(std::round(params.dt_mpc / params.dt_wbc));
  n_steps = static_cast<int>(params.gait.rows());
  footsteps_under_shoulders_ << Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(params.footsteps_under_shoulders.data(),
                                                                              params.footsteps_under_shoulders.size());
  // Offsets to make the support polygon smaller
  double ox = 0.0;
  double oy = 0.0;
  footsteps_offset_ << -ox, -ox, ox, ox, -oy, +oy, +oy, -oy, 0.0, 0.0, 0.0, 0.0;
  currentFootstep_ << Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(params.footsteps_init.data(),
                                                                    params.footsteps_init.size());
  gait_ = &gaitIn;
  targetFootstep_ = currentFootstep_;
  o_targetFootstep_ = currentFootstep_;
  dt_cum = VectorN::Zero(params.gait.rows());
  yaws = VectorN::Zero(params.gait.rows());
  dx = VectorN::Zero(params.gait.rows());
  dy = VectorN::Zero(params.gait.rows());
  for (int i = 0; i < params.gait.rows(); i++) {
    footsteps_.push_back(Matrix34::Zero());
  }
  Rz(2, 2) = 1.0;

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

  // Get feet frame IDs
  foot_ids_[0] = static_cast<int>(model_.getFrameId("FL_FOOT"));  // from long uint to int
  foot_ids_[1] = static_cast<int>(model_.getFrameId("FR_FOOT"));
  foot_ids_[2] = static_cast<int>(model_.getFrameId("HL_FOOT"));
  foot_ids_[3] = static_cast<int>(model_.getFrameId("HR_FOOT"));
}

MatrixN FootstepPlanner::updateFootsteps(int k, VectorN const& q, Vector6 const& b_v, Vector6 const& b_vref,
                                         MatrixN const& ftgPositions) {
  if (q.rows() != 18) {
    throw std::runtime_error("q should be a vector of size 18 (pos+RPY+mot)");
  }

  /*std::cout << "Update footsteps " << std::endl;
  std::cout << gait_->getCurrentGait().row(0) << std::endl;
  std::cout << gait_->getCurrentGait().row(1) << std::endl;*/

  // Update location of feet in stance phase (for those which just entered stance phase)
  if ((k % k_mpc_ == 0 && k != 0) || gait_->isNewPhase()) {
    // updateNewContact(q);

    // Remove translation and yaw rotation to get position in local frame
    // q_FK_.head(3) << 0.0, 0.0, q(2, 0);
    // q_FK_.block(3, 0, 4, 1) = pinocchio::SE3::Quaternion(pinocchio::rpy::rpyToMatrix(q(3, 0), q(4, 0), 0.0)).coeffs();
    // q_FK_.tail(12) = q.tail(12);

    // Update model and data of the robot
    // pinocchio::forwardKinematics(model_, data_, q_FK_);
    // pinocchio::updateFramePlacements(model_, data_);

    // Retrieve gait status
    RowVector4 currentGait = gait_->getCurrentGait().row(0);

    // Rotation from world to horizontal
    double c = std::cos(q[5]);
    double s = std::sin(q[5]);
    Rz.topLeftCorner<2, 2>() << c, s, -s, c;

    // Refresh position with estimated position if foot is in stance phase
    for (int i = 0; i < 4; i++) {
      if (currentGait(0, i) - previousGait_(0, i) >
          0) {  // New contact when currentGait[i] = 1 and previousGait_[i] = 0
        currentFootstep_.col(i) = Rz * (ftgPositions.col(i).head(3) - (Vector3(q[0], q[1], 0.0) - Rz * b_vref.head(3) * dt_wbc));
        // currentFootstep_(2, i) = data_.oMf[foot_ids_[i]].translation()(2, 0);
        previousHeight_(0, i) = currentFootstep_(2, i);
        // std::cout << "= New contact at " << currentFootstep_.col(i).transpose() << " for foot " << i << std::endl;
      }
    }

    // Keep gait status in memory
    previousGait_ = currentGait;
  }

  // Feet in contact with the ground are moving in base frame (they don't move in world frame)
  double rotation_yaw = dt_wbc * b_vref(5);  // Rotation along Z for the last time step
  double c = std::cos(rotation_yaw);
  double s = std::sin(rotation_yaw);
  Rz.topLeftCorner<2, 2>() << c, s, -s, c;
  Vector2 dpos = dt_wbc * b_vref.head(2);  // Displacement along X and Y for the last time step
  for (int j = 0; j < 4; j++) {
    if (gait_->getCurrentGaitCoeff(0, j) == 1.0) {
      currentFootstep_.block(0, j, 2, 1) = Rz * (currentFootstep_.block(0, j, 2, 1) - dpos);
    }
  }

  // Compute location of footsteps
  return computeTargetFootstep(k, q.head(6), b_v, b_vref);
}

void FootstepPlanner::computeFootsteps(int k, Vector6 const& b_v, Vector6 const& b_vref) {
  for (uint i = 0; i < footsteps_.size(); i++) {
    footsteps_[i] = Matrix34::Zero();
  }
  MatrixN gait = gait_->getCurrentGait();

  // Set current position of feet for feet in stance phase
  for (int j = 0; j < 4; j++) {
    if (gait(0, j) == 1.0) {
      footsteps_[0].col(j) = currentFootstep_.col(j);
    }
  }

  // Cumulative time by adding the terms in the first column (remaining number of timesteps)
  // Get future yaw yaws compared to current position
  dt_cum(0) = dt_wbc * static_cast<double>(k_mpc_ - k % k_mpc_);
  yaws(0) = b_vref(5) * dt_cum(0);
  for (uint j = 1; j < footsteps_.size(); j++) {
    dt_cum(j) = gait.row(j).isZero() ? dt_cum(j - 1) : dt_cum(j - 1) + dt;
    yaws(j) = b_vref(5) * dt_cum(j);
  }

  // Displacement following the reference velocity compared to current position
  if (b_vref(5, 0) != 0) {
    for (uint j = 0; j < footsteps_.size(); j++) {
      dx(j) = (b_vref(0) * std::sin(b_vref(5) * dt_cum(j)) + b_vref(1) * (std::cos(b_vref(5) * dt_cum(j)) - 1.0)) /
              b_vref(5);
      dy(j) = (b_vref(1) * std::sin(b_vref(5) * dt_cum(j)) - b_vref(0) * (std::cos(b_vref(5) * dt_cum(j)) - 1.0)) /
              b_vref(5);
    }
  } else {
    for (uint j = 0; j < footsteps_.size(); j++) {
      dx(j) = b_vref(0) * dt_cum(j);
      dy(j) = b_vref(1) * dt_cum(j);
    }
  }

  // Update the footstep matrix depending on the different phases of the gait (swing & stance)
  for (int i = 1; i < gait.rows(); i++) {
    // Feet that were in stance phase and are still in stance phase do not move
    for (int j = 0; j < 4; j++) {
      if (gait(i - 1, j) > 0 && gait(i, j) > 0) {
        footsteps_[i].col(j) = footsteps_[i - 1].col(j);
      }
    }

    // Feet that were in swing phase and are now in stance phase need to be updated
    for (int j = 0; j < 4; j++) {
      if (gait(i - 1, j) == 0 && gait(i, j) > 0) {
        // Offset to the future position
        q_dxdy << dx(i - 1, 0), dy(i - 1, 0), 0.0;

        // Get future desired position of footsteps
        computeNextFootstep(i, j, b_v, b_vref);

        // Get desired position of footstep compared to current position
        double c = std::cos(yaws(i - 1));
        double s = std::sin(yaws(i - 1));
        Rz.topLeftCorner<2, 2>() << c, -s, s, c;

        footsteps_[i].col(j) = (Rz * nextFootstep_.col(j) + q_dxdy).transpose();
      }
    }
  }
}

void FootstepPlanner::computeNextFootstep(int i, int j, Vector6 const& b_v, Vector6 const& b_vref) {
  nextFootstep_ = Matrix34::Zero();
  double t_stance = gait_->getPhaseDuration(i, j);

  // Disable heuristic terms if gait is going to switch to static so that feet land at vertical of shoulders
  if (!gait_->getIsStatic()) {
    // Add symmetry term
    nextFootstep_.col(j) = t_stance * 0.5 * b_v.head(3);

    // Add feedback term
    nextFootstep_.col(j) += params_->k_feedback * (b_v.head(3) - b_vref.head(3));

    // Add centrifugal term
    Vector3 cross;
    cross << b_v(1) * b_vref(5) - b_v(2) * b_vref(4), b_v(2) * b_vref(3) - b_v(0) * b_vref(5), 0.0;
    nextFootstep_.col(j) += 0.5 * std::sqrt(h_ref / g) * cross;
  }

  // Legs have a limited length so the deviation has to be limited
  nextFootstep_(0, j) = std::min(nextFootstep_(0, j), L);
  nextFootstep_(0, j) = std::max(nextFootstep_(0, j), -L);
  nextFootstep_(1, j) = std::min(nextFootstep_(1, j), L);
  nextFootstep_(1, j) = std::max(nextFootstep_(1, j), -L);

  // Add shoulders
  nextFootstep_.col(j) += footsteps_under_shoulders_.col(j);
  nextFootstep_.col(j) += footsteps_offset_.col(j);

  // Remove Z component (working on flat ground)
  nextFootstep_.row(2) = previousHeight_;
}

void FootstepPlanner::updateTargetFootsteps() {
  for (int i = 0; i < 4; i++) {
    int index = 0;
    while (footsteps_[index](0, i) == 0.0) {
      index++;
    }
    targetFootstep_.col(i) = footsteps_[index].col(i);
  }
}

MatrixN FootstepPlanner::computeTargetFootstep(int k, Vector6 const& q, Vector6 const& b_v, Vector6 const& b_vref) {
  // Compute the desired location of footsteps over the prediction horizon
  computeFootsteps(k, b_v, b_vref);

  // Update desired location of footsteps on the ground
  updateTargetFootsteps();

  // Get o_targetFootstep_ in world frame from targetFootstep_ in horizontal frame
  RPY_ = q.tail(3);
  double c = std::cos(RPY_(2));
  double s = std::sin(RPY_(2));
  Rz.topLeftCorner<2, 2>() << c, -s, s, c;
  for (int i = 0; i < 4; i++) {
    o_targetFootstep_.block(0, i, 2, 1) = Rz.topLeftCorner<2, 2>() * targetFootstep_.block(0, i, 2, 1) + q.head(2);
    o_targetFootstep_(2, i) = targetFootstep_(2, i);
  }

  return o_targetFootstep_;
}

void FootstepPlanner::updateNewContact(Vector18 const& q) {
  // Remove translation and yaw rotation to get position in local frame
  q_FK_.head(2) = Vector2::Zero();
  q_FK_(2, 0) = q(2, 0);
  q_FK_.block(3, 0, 4, 1) = pinocchio::SE3::Quaternion(pinocchio::rpy::rpyToMatrix(q(3, 0), q(4, 0), 0.0)).coeffs();
  q_FK_.tail(12) = q.tail(12);

  // Update model and data of the robot
  pinocchio::forwardKinematics(model_, data_, q_FK_);
  pinocchio::updateFramePlacements(model_, data_);

  // Get data required by IK with Pinocchio
  for (int i = 0; i < 4; i++) {
    pos_feet_.col(i) = data_.oMf[foot_ids_[i]].translation();
  }

  // std::cout << "--- pos_feet_: " << std::endl << pos_feet_ << std::endl;
  // std::cout << "--- footsteps_:" << std::endl << footsteps_[1] << std::endl;

  // Refresh position with estimated position if foot is in stance phase
  for (int i = 0; i < 4; i++) {
    if (gait_->getCurrentGaitCoeff(0, i) == 1.0) {
      currentFootstep_.block(0, i, 2, 1) = pos_feet_.block(0, i, 2, 1);  // Get only x and y to let z = 0 for contacts
    }
  }
}

MatrixN FootstepPlanner::getFootsteps() { return vectorToMatrix(footsteps_); }
MatrixN FootstepPlanner::getTargetFootsteps() { return targetFootstep_; }

MatrixN FootstepPlanner::vectorToMatrix(std::vector<Matrix34> const& array) {
  MatrixN M = MatrixN::Zero(array.size(), 12);
  for (uint i = 0; i < array.size(); i++) {
    for (int j = 0; j < 4; j++) {
      M.row(i).segment<3>(3 * j) = array[i].col(j);
    }
  }
  return M;
}
