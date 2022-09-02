#include "qrw/StatePlanner.hpp"

StatePlanner::StatePlanner()
    : dt_(0.0), dt_wbc_(0.0), h_ref_(0.0), n_steps_(0), h_feet_mean_(0.0), RPY_(Vector3::Zero()) {
  // Empty
}

void StatePlanner::initialize(Params& params, Gait& gaitIn) {
  dt_ = params.dt_mpc;
  dt_wbc_ = params.dt_wbc;
  h_ref_ = params.h_ref;
  n_steps_ = static_cast<int>(params.gait.rows());
  referenceStates_ = MatrixN::Zero(12, 1 + n_steps_);
  dt_vector_ = VectorN::LinSpaced(n_steps_, dt_, static_cast<double>(n_steps_) * dt_);
  gait_ = &gaitIn;
}

void StatePlanner::preJumpTrajectory(int i, double t_swing, int k) {
  double g = 9.81;
  double A5 = -3 * std::pow(t_swing, 2) * g / (8 * std::pow(t_swing, 5));
  double A4 = 15 * std::pow(t_swing, 2) * g / (16 * std::pow(t_swing, 4));
  double A3 = -5 * std::pow(t_swing, 2) * g / (8 * std::pow(t_swing, 3));

  // Third phase in chronological order: jump phase, no feet in contact, ballistic motion

  // Second phase in chronological order: quickly raise the base to reach a sufficient vertical velocity
  int j = 0;
  // std::cout << "Target: " << g * t_swing / 2 << std::endl;
  // std::cout << "----" << std::endl;
  while (i >= 0 && (j * dt_ < (t_swing / 4 - 1e-5))) {
    double t_p = j * dt_;
    if (i == 0) {
      t_p -= (k % 20) * dt_wbc_ - dt_;
    }
    referenceStates_(8, 1 + i) = g * t_swing / 2 - 2.0 * g * t_p;
    referenceStates_(2, 1 + i) = h_ref_ + h_feet_mean_ + 2.0 * g * std::pow(t_p, 2) * 0.5 - g * t_swing / 2 * t_p;
    // std::cout << "t_p: " << t_p << " | Pos: " << referenceStates_(2, 1 + i) << " | Vel: " << referenceStates_(8, 1 +
    // i) << std::endl;
    j++;
    i--;
  }
  for (int a = 0; a <= j; a++) {
    referenceStates_(6, 1 + i + a) = 0.81 * 0.0 * (static_cast<double>(a) / static_cast<double>(j));
    if (i + a >= 0) {
      referenceStates_(0, 1 + i + a) = 0.81 * 0.0 * dt_ + referenceStates_(0, i + a);
    }
  }

  // First phase in chronological order: lower the base to prepare for the jump
  j = 1;
  while (i >= 0 && (j * dt_ <= t_swing)) {
    double td = t_swing - j * dt_;
    if (i == 0) {
      td += (k % 20) * dt_wbc_;
    }
    referenceStates_(2, 1 + i) =
        h_ref_ + h_feet_mean_ + A5 * std::pow(td, 5) + A4 * std::pow(td, 4) + A3 * std::pow(td, 3);
    referenceStates_(8, 1 + i) = 5 * A5 * std::pow(td, 4) + 4 * A4 * std::pow(td, 3) + 3 * A3 * std::pow(td, 2);
    j++;
    i--;
  }
}

void StatePlanner::computeReferenceStates(int k, VectorN const& q, Vector6 const& v, Vector6 const& vref,
                                          MatrixN fsteps) {
  if (q.rows() != 6) {
    throw std::runtime_error("q should be a vector of size 6");
  }
  RPY_ = q.tail(3);

  // Low pass of the mean height of feet in contact
  int cpt = 0;
  double sum = 0;
  RowVector4 cgait = gait_->getCurrentGait().row(0);
  for (int i = 0; i < 4; i++) {
    if (cgait(0, i) == 1) {
      cpt++;
      sum += fsteps(3 * i + 2, 0);
    }
  }
  if (cpt > 0) {
    h_feet_mean_ = 0.0;  // 0.99 * h_feet_mean_ + 0.005 * sum / cpt;
  }

  // Update the current state
  referenceStates_(0, 0) = 0.0;                       // In horizontal frame x = 0.0
  referenceStates_(1, 0) = 0.0;                       // In horizontal frame y = 0.0
  referenceStates_(2, 0) = q(2, 0);                   // We keep height
  referenceStates_.block(3, 0, 2, 1) = RPY_.head(2);  // We keep roll and pitch
  referenceStates_(5, 0) = 0.0;                       // In horizontal frame yaw = 0.0
  referenceStates_.block(6, 0, 3, 1) = v.head(3);
  referenceStates_.block(9, 0, 3, 1) = v.tail(3);

  for (int i = 0; i < n_steps_; i++) {
    if (vref(5) != 0) {
      referenceStates_(0, 1 + i) =
          (vref(0) * std::sin(vref(5) * dt_vector_(i)) + vref(1) * (std::cos(vref(5) * dt_vector_(i)) - 1.0)) /
          vref(5);
      referenceStates_(1, 1 + i) =
          (vref(1) * std::sin(vref(5) * dt_vector_(i)) - vref(0) * (std::cos(vref(5) * dt_vector_(i)) - 1.0)) /
          vref(5);
    } else {
      referenceStates_(0, 1 + i) = vref(0) * dt_vector_(i);
      referenceStates_(1, 1 + i) = vref(1) * dt_vector_(i);
    }
    referenceStates_(0, 1 + i) += referenceStates_(0, 0);
    referenceStates_(1, 1 + i) += referenceStates_(1, 0);

    referenceStates_(5, 1 + i) = vref(5) * dt_vector_(i);

    referenceStates_(6, 1 + i) =
        vref(0) * std::cos(referenceStates_(5, 1 + i)) - vref(1) * std::sin(referenceStates_(5, 1 + i));
    referenceStates_(7, 1 + i) =
        vref(0) * std::sin(referenceStates_(5, 1 + i)) + vref(1) * std::cos(referenceStates_(5, 1 + i));

    referenceStates_(11, 1 + i) = vref(5);
  }

  // Handle gait phases with no feet in contact with the ground

  MatrixN gait = gait_->getCurrentGait();
  for (int i = 0; i < n_steps_; i++) {
    if (gait.row(i).isZero())  // Enable for jumping
    {
      // Assumption of same duration for all feet
      double t_swing = gait_->getPhaseDuration(i);  // 0.0 for swing phase

      // Compute the reference trajectory of the CoM for time steps before the jump phase
      // preJumpTrajectory(i - 1, t_swing, k);

      // Vertical velocity at the start of the fly phase
      double g = 9.81;
      double vz0 = -g * t_swing * 0.5;

      double t_fly = t_swing - gait_->getRemainingTime(i);
      while (i < n_steps_ && gait.row(i).isZero()) {
        if (i != 0) {
          referenceStates_(2, 1 + i) = h_ref_ + h_feet_mean_ - g * 0.5 * t_fly * (t_fly - t_swing);
          referenceStates_(8, 1 + i) = g * (t_swing * 0.5 - t_fly);
          referenceStates_(0, 1 + i) = 0.81 * 0.0 * dt_ + referenceStates_(0, i);
          referenceStates_(6, 1 + i) = 0.81 * 0.0;
        } else {
          double t_p = t_fly + (k % 20) * dt_wbc_ - dt_;
          referenceStates_(2, 1 + i) = h_ref_ + h_feet_mean_ - g * 0.5 * t_p * (t_p - t_swing);
          referenceStates_(8, 1 + i) = g * (t_swing * 0.5 - t_p);
          referenceStates_(6, 1 + i) = 0.81 * 0.0;
        }
        // std::cout << t_swing << " | " << t_fly << " | " << referenceStates_(2, 1 + i) << std::endl;
        // Pitch
        // referenceStates_(4, 1 + i) = 0.5;
        // referenceStates_(10, 1 + i) = 3.0;
        t_fly += dt_;
        i++;
      }
      i--;
    } else {
      referenceStates_(2, 1 + i) = h_ref_ + h_feet_mean_;
      referenceStates_(8, 1 + i) = 0.0;
      if (false && i > 0) {  // Enable for jumping
        referenceStates_(0, 1 + i) = referenceStates_(0, i);
      }

      // Pitch
      // referenceStates_(4, 1 + i) = 0.0;
      // referenceStates_(10, 1 + i) = 0.0;
    }
  }
}
