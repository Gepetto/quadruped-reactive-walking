#include "qrw/FootTrajectoryGenerator.hpp"

// Trajectory generator functions (output reference pos, vel and acc of feet in swing phase)

FootTrajectoryGenerator::FootTrajectoryGenerator()
    : gait_(NULL),
      dt_wbc(0.0),
      k_mpc(0),
      maxHeight_(0.0),
      lockTime_(0.0),
      feet(Eigen::Matrix<double, 1, 4>::Zero()),
      t0s(Vector4::Zero()),
      t_swing(Vector4::Zero()),
      stepHeight_(Vector4::Zero()),
      targetFootstep_(Matrix34::Zero()),
      Ax(Matrix64::Zero()),
      Ay(Matrix64::Zero()),
      position_(Matrix34::Zero()),
      velocity_(Matrix34::Zero()),
      acceleration_(Matrix34::Zero()),
      jerk_(Matrix34::Zero()),
      position_base_(Matrix34::Zero()),
      velocity_base_(Matrix34::Zero()),
      acceleration_base_(Matrix34::Zero()) {}

void FootTrajectoryGenerator::initialize(Params &params, Gait &gaitIn) {
  dt_wbc = params.dt_wbc;
  k_mpc = (int)std::round(params.dt_mpc / params.dt_wbc);
  maxHeight_ = params.max_height;
  lockTime_ = params.lock_time;
  vertTime_ = params.vert_time;
  targetFootstep_ << Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(params.footsteps_init.data(),
                                                                   params.footsteps_init.size());
  position_ = targetFootstep_;
  position_base_ = targetFootstep_;
  gait_ = &gaitIn;
}

void FootTrajectoryGenerator::updateFootPosition(int const j, Vector3 const &targetFootstep) {
  double ddx0 = acceleration_(0, j);
  double ddy0 = acceleration_(1, j);
  double dx0 = velocity_(0, j);
  double dy0 = velocity_(1, j);
  double x0 = position_(0, j);
  double y0 = position_(1, j);

  double t = t0s[j] - vertTime_;
  double d = t_swing[j] - 2 * vertTime_;
  double dt = dt_wbc;

  if (t < d - lockTime_) {
    // compute polynoms coefficients for x and y
    Ax(0, j) = (ddx0 * std::pow(t, 2) - 2 * ddx0 * t * d - 6 * dx0 * t + ddx0 * std::pow(d, 2) + 6 * dx0 * d +
                12 * x0 - 12 * targetFootstep[0]) /
               (2 * std::pow((t - d), 2) *
                (std::pow(t, 3) - 3 * std::pow(t, 2) * d + 3 * t * std::pow(d, 2) - std::pow(d, 3)));
    Ax(1, j) =
        (30 * t * targetFootstep[0] - 30 * t * x0 - 30 * d * x0 + 30 * d * targetFootstep[0] -
         2 * std::pow(t, 3) * ddx0 - 3 * std::pow(d, 3) * ddx0 + 14 * std::pow(t, 2) * dx0 -
         16 * std::pow(d, 2) * dx0 + 2 * t * d * dx0 + 4 * t * std::pow(d, 2) * ddx0 + std::pow(t, 2) * d * ddx0) /
        (2 * std::pow((t - d), 2) *
         (std::pow(t, 3) - 3 * std::pow(t, 2) * d + 3 * t * std::pow(d, 2) - std::pow(d, 3)));
    Ax(2, j) = (std::pow(t, 4) * ddx0 + 3 * std::pow(d, 4) * ddx0 - 8 * std::pow(t, 3) * dx0 +
                12 * std::pow(d, 3) * dx0 + 20 * std::pow(t, 2) * x0 - 20 * std::pow(t, 2) * targetFootstep[0] +
                20 * std::pow(d, 2) * x0 - 20 * std::pow(d, 2) * targetFootstep[0] + 80 * t * d * x0 -
                80 * t * d * targetFootstep[0] + 4 * std::pow(t, 3) * d * ddx0 + 28 * t * std::pow(d, 2) * dx0 -
                32 * std::pow(t, 2) * d * dx0 - 8 * std::pow(t, 2) * std::pow(d, 2) * ddx0) /
               (2 * std::pow((t - d), 2) *
                (std::pow(t, 3) - 3 * std::pow(t, 2) * d + 3 * t * std::pow(d, 2) - std::pow(d, 3)));
    Ax(3, j) = -(std::pow(d, 5) * ddx0 + 4 * t * std::pow(d, 4) * ddx0 + 3 * std::pow(t, 4) * d * ddx0 +
                 36 * t * std::pow(d, 3) * dx0 - 24 * std::pow(t, 3) * d * dx0 + 60 * t * std::pow(d, 2) * x0 +
                 60 * std::pow(t, 2) * d * x0 - 60 * t * std::pow(d, 2) * targetFootstep[0] -
                 60 * std::pow(t, 2) * d * targetFootstep[0] - 8 * std::pow(t, 2) * std::pow(d, 3) * ddx0 -
                 12 * std::pow(t, 2) * std::pow(d, 2) * dx0) /
               (2 * (std::pow(t, 2) - 2 * t * d + std::pow(d, 2)) *
                (std::pow(t, 3) - 3 * std::pow(t, 2) * d + 3 * t * std::pow(d, 2) - std::pow(d, 3)));
    Ax(4, j) = -(2 * std::pow(d, 5) * dx0 - 2 * t * std::pow(d, 5) * ddx0 - 10 * t * std::pow(d, 4) * dx0 +
                 std::pow(t, 2) * std::pow(d, 4) * ddx0 + 4 * std::pow(t, 3) * std::pow(d, 3) * ddx0 -
                 3 * std::pow(t, 4) * std::pow(d, 2) * ddx0 - 16 * std::pow(t, 2) * std::pow(d, 3) * dx0 +
                 24 * std::pow(t, 3) * std::pow(d, 2) * dx0 - 60 * std::pow(t, 2) * std::pow(d, 2) * x0 +
                 60 * std::pow(t, 2) * std::pow(d, 2) * targetFootstep[0]) /
               (2 * std::pow((t - d), 2) *
                (std::pow(t, 3) - 3 * std::pow(t, 2) * d + 3 * t * std::pow(d, 2) - std::pow(d, 3)));
    Ax(5, j) = (2 * targetFootstep[0] * std::pow(t, 5) - ddx0 * std::pow(t, 4) * std::pow(d, 3) -
                10 * targetFootstep[0] * std::pow(t, 4) * d + 2 * ddx0 * std::pow(t, 3) * std::pow(d, 4) +
                8 * dx0 * std::pow(t, 3) * std::pow(d, 3) + 20 * targetFootstep[0] * std::pow(t, 3) * std::pow(d, 2) -
                ddx0 * std::pow(t, 2) * std::pow(d, 5) - 10 * dx0 * std::pow(t, 2) * std::pow(d, 4) -
                20 * x0 * std::pow(t, 2) * std::pow(d, 3) + 2 * dx0 * t * std::pow(d, 5) +
                10 * x0 * t * std::pow(d, 4) - 2 * x0 * std::pow(d, 5)) /
               (2 * (std::pow(t, 2) - 2 * t * d + std::pow(d, 2)) *
                (std::pow(t, 3) - 3 * std::pow(t, 2) * d + 3 * t * std::pow(d, 2) - std::pow(d, 3)));

    Ay(0, j) = (ddy0 * std::pow(t, 2) - 2 * ddy0 * t * d - 6 * dy0 * t + ddy0 * std::pow(d, 2) + 6 * dy0 * d +
                12 * y0 - 12 * targetFootstep[1]) /
               (2 * std::pow((t - d), 2) *
                (std::pow(t, 3) - 3 * std::pow(t, 2) * d + 3 * t * std::pow(d, 2) - std::pow(d, 3)));
    Ay(1, j) =
        (30 * t * targetFootstep[1] - 30 * t * y0 - 30 * d * y0 + 30 * d * targetFootstep[1] -
         2 * std::pow(t, 3) * ddy0 - 3 * std::pow(d, 3) * ddy0 + 14 * std::pow(t, 2) * dy0 -
         16 * std::pow(d, 2) * dy0 + 2 * t * d * dy0 + 4 * t * std::pow(d, 2) * ddy0 + std::pow(t, 2) * d * ddy0) /
        (2 * std::pow((t - d), 2) *
         (std::pow(t, 3) - 3 * std::pow(t, 2) * d + 3 * t * std::pow(d, 2) - std::pow(d, 3)));
    Ay(2, j) = (std::pow(t, 4) * ddy0 + 3 * std::pow(d, 4) * ddy0 - 8 * std::pow(t, 3) * dy0 +
                12 * std::pow(d, 3) * dy0 + 20 * std::pow(t, 2) * y0 - 20 * std::pow(t, 2) * targetFootstep[1] +
                20 * std::pow(d, 2) * y0 - 20 * std::pow(d, 2) * targetFootstep[1] + 80 * t * d * y0 -
                80 * t * d * targetFootstep[1] + 4 * std::pow(t, 3) * d * ddy0 + 28 * t * std::pow(d, 2) * dy0 -
                32 * std::pow(t, 2) * d * dy0 - 8 * std::pow(t, 2) * std::pow(d, 2) * ddy0) /
               (2 * std::pow((t - d), 2) *
                (std::pow(t, 3) - 3 * std::pow(t, 2) * d + 3 * t * std::pow(d, 2) - std::pow(d, 3)));
    Ay(3, j) = -(std::pow(d, 5) * ddy0 + 4 * t * std::pow(d, 4) * ddy0 + 3 * std::pow(t, 4) * d * ddy0 +
                 36 * t * std::pow(d, 3) * dy0 - 24 * std::pow(t, 3) * d * dy0 + 60 * t * std::pow(d, 2) * y0 +
                 60 * std::pow(t, 2) * d * y0 - 60 * t * std::pow(d, 2) * targetFootstep[1] -
                 60 * std::pow(t, 2) * d * targetFootstep[1] - 8 * std::pow(t, 2) * std::pow(d, 3) * ddy0 -
                 12 * std::pow(t, 2) * std::pow(d, 2) * dy0) /
               (2 * (std::pow(t, 2) - 2 * t * d + std::pow(d, 2)) *
                (std::pow(t, 3) - 3 * std::pow(t, 2) * d + 3 * t * std::pow(d, 2) - std::pow(d, 3)));
    Ay(4, j) = -(2 * std::pow(d, 5) * dy0 - 2 * t * std::pow(d, 5) * ddy0 - 10 * t * std::pow(d, 4) * dy0 +
                 std::pow(t, 2) * std::pow(d, 4) * ddy0 + 4 * std::pow(t, 3) * std::pow(d, 3) * ddy0 -
                 3 * std::pow(t, 4) * std::pow(d, 2) * ddy0 - 16 * std::pow(t, 2) * std::pow(d, 3) * dy0 +
                 24 * std::pow(t, 3) * std::pow(d, 2) * dy0 - 60 * std::pow(t, 2) * std::pow(d, 2) * y0 +
                 60 * std::pow(t, 2) * std::pow(d, 2) * targetFootstep[1]) /
               (2 * std::pow((t - d), 2) *
                (std::pow(t, 3) - 3 * std::pow(t, 2) * d + 3 * t * std::pow(d, 2) - std::pow(d, 3)));
    Ay(5, j) = (2 * targetFootstep[1] * std::pow(t, 5) - ddy0 * std::pow(t, 4) * std::pow(d, 3) -
                10 * targetFootstep[1] * std::pow(t, 4) * d + 2 * ddy0 * std::pow(t, 3) * std::pow(d, 4) +
                8 * dy0 * std::pow(t, 3) * std::pow(d, 3) + 20 * targetFootstep[1] * std::pow(t, 3) * std::pow(d, 2) -
                ddy0 * std::pow(t, 2) * std::pow(d, 5) - 10 * dy0 * std::pow(t, 2) * std::pow(d, 4) -
                20 * y0 * std::pow(t, 2) * std::pow(d, 3) + 2 * dy0 * t * std::pow(d, 5) +
                10 * y0 * t * std::pow(d, 4) - 2 * y0 * std::pow(d, 5)) /
               (2 * (std::pow(t, 2) - 2 * t * d + std::pow(d, 2)) *
                (std::pow(t, 3) - 3 * std::pow(t, 2) * d + 3 * t * std::pow(d, 2) - std::pow(d, 3)));

    targetFootstep_(0, j) = targetFootstep[0];
    targetFootstep_(1, j) = targetFootstep[1];
  }

  // Coefficients for z (deterministic)
  double Tz = t_swing[j];
  if (t0s[j] == 0) {
    stepHeight_[j] = maxHeight_ - std::max(0.0, targetFootstep_(2, j)); //* (1.0 - 0.5 * std::max(0.0, targetFootstep_(2, j)) / maxHeight_);
    // stepHeight_[j] = maxHeight_ * (1.0 - 0.5 * targetFootstep_(2, j) / maxHeight_);
  }

  Vector3 Az;  // z trajectory is split in two halfs (before/after apex point)
  if (t0s[j] < Tz * 0.5) {
    Az(0, 0) = 192.0 * stepHeight_[j] / std::pow(Tz, 5);
    Az(1, 0) = -240.0 * stepHeight_[j] / std::pow(Tz, 4);
    Az(2, 0) = 80.0 * stepHeight_[j] / std::pow(Tz, 3);
  } else {
    Az(0, 0) = -144.0 * (stepHeight_[j] + targetFootstep_(2, j)) / std::pow(Tz, 5);
    Az(1, 0) = 184.0 * (stepHeight_[j] + targetFootstep_(2, j)) / std::pow(Tz, 4);
    Az(2, 0) = -64.0 * (stepHeight_[j] + targetFootstep_(2, j)) / std::pow(Tz, 3);
  }

  // Get the next point
  double ev = t + dt;
  double evz = t0s[j] + dt;

  if (t < 0.0 || t >= d)  // Just vertical motion
  {
    position_(0, j) = x0;
    position_(1, j) = y0;
    velocity_(0, j) = 0.0;
    velocity_(1, j) = 0.0;
    acceleration_(0, j) = 0.0;
    acceleration_(1, j) = 0.0;
    jerk_(0, j) = 0.0;
    jerk_(1, j) = 0.0;
  } else {
    position_(0, j) = Ax(5, j) + Ax(4, j) * ev + Ax(3, j) * std::pow(ev, 2) + Ax(2, j) * std::pow(ev, 3) +
                      Ax(1, j) * std::pow(ev, 4) + Ax(0, j) * std::pow(ev, 5);
    position_(1, j) = Ay(5, j) + Ay(4, j) * ev + Ay(3, j) * std::pow(ev, 2) + Ay(2, j) * std::pow(ev, 3) +
                      Ay(1, j) * std::pow(ev, 4) + Ay(0, j) * std::pow(ev, 5);
    velocity_(0, j) = Ax(4, j) + 2 * Ax(3, j) * ev + 3 * Ax(2, j) * std::pow(ev, 2) + 4 * Ax(1, j) * std::pow(ev, 3) +
                      5 * Ax(0, j) * std::pow(ev, 4);
    velocity_(1, j) = Ay(4, j) + 2 * Ay(3, j) * ev + 3 * Ay(2, j) * std::pow(ev, 2) + 4 * Ay(1, j) * std::pow(ev, 3) +
                      5 * Ay(0, j) * std::pow(ev, 4);
    acceleration_(0, j) =
        2 * Ax(3, j) + 3 * 2 * Ax(2, j) * ev + 4 * 3 * Ax(1, j) * std::pow(ev, 2) + 5 * 4 * Ax(0, j) * std::pow(ev, 3);
    acceleration_(1, j) =
        2 * Ay(3, j) + 3 * 2 * Ay(2, j) * ev + 4 * 3 * Ay(1, j) * std::pow(ev, 2) + 5 * 4 * Ay(0, j) * std::pow(ev, 3);

    jerk_(0, j) = 3 * 2 * Ax(2, j) + 4 * 3 * 2 * Ax(1, j) * ev + 5 * 4 * 3 * Ax(0, j) * std::pow(ev, 2);
    jerk_(1, j) = 3 * 2 * Ay(2, j) + 4 * 3 * 2 * Ay(1, j) * ev + 5 * 4 * 3 * Ay(0, j) * std::pow(ev, 2);
  }

  if (t0s[j] >= Tz * 0.5) {
    evz -= Tz * 0.5;  // To make coefficients simpler, second half was computed as if it started at t = 0
  }
  velocity_(2, j) =
      3 * Az(2, 0) * std::pow(evz, 2) + 4 * Az(1, 0) * std::pow(evz, 3) + 5 * Az(0, 0) * std::pow(evz, 4);
  acceleration_(2, j) =
      2 * 3 * Az(2, 0) * evz + 3 * 4 * Az(1, 0) * std::pow(evz, 2) + 4 * 5 * Az(0, 0) * std::pow(evz, 3);
  jerk_(2, j) = 2 * 3 * Az(2, 0) + 3 * 4 * 2 * Az(1, 0) * evz + 4 * 5 * 3 * Az(0, 0) * std::pow(evz, 2);
  position_(2, j) =
      Az(2, 0) * std::pow(evz, 3) + Az(1, 0) * std::pow(evz, 4) + Az(0, 0) * std::pow(evz, 5) + targetFootstep_(2, j);
  if (t0s[j] >= Tz * 0.5) {
    position_(2, j) += stepHeight_[j];  // Second half starts at the apex height then goes down
  }
}

void FootTrajectoryGenerator::update(int k, MatrixN const &targetFootstep) {
  if ((k % k_mpc) == 0 || gait_->isNewPhase()) {
    // Status of feet
    feet = gait_->getCurrentGait().row(0);

    // For each foot in swing phase get remaining duration of the swing phase
    for (int i = 0; i < 4; i++) {
      if (feet(0, i) == 0) {
        t_swing[i] = gait_->getPhaseDuration(0, i);
        double value = gait_->getElapsedTime(0, i) + (k % k_mpc) * dt_wbc;
        t0s[i] = std::max(0.0, value);
      }
    }
  } else {
    // Increment of one time step for feet in swing phase
    for (int i = 0; i < 4; i++) {
      if (feet(0, i) == 0) {
        double value = t0s[i] + dt_wbc;
        t0s[i] = std::max(0.0, value);
      }
    }
  }

  for (int i = 0; i < 4; i++) {
    if (feet(0, i) == 0) {
      if (!gait_->isLate(i)) {
        updateFootPosition(i, targetFootstep.col(i));
      } else {
        double vz = -0.5 * stepHeight_[i] / (t_swing[i] * 0.5);  // Velocity at the end of second half of z trajectory
        targetFootstep_(2, i) += vz * dt_wbc;                    // Lowering the foot until contact
        position_.col(i) = targetFootstep_.col(i);
        velocity_.col(i) << 0.0, 0.0, vz;
        acceleration_.col(i).setZero();
        jerk_.col(i).setZero();
      }
    } else {
      position_(2, i) = targetFootstep(2, i);
      targetFootstep_.col(i) = position_.col(i);
      velocity_.col(i).setZero();
      acceleration_.col(i).setZero();
      jerk_.col(i).setZero();
    }
  }
  return;
}

MatrixN FootTrajectoryGenerator::getTrajectoryToTarget(int const j) {
  double ddx0 = acceleration_(0, j);
  double ddy0 = acceleration_(1, j);
  double dx0 = velocity_(0, j);
  double dy0 = velocity_(1, j);
  double x0 = position_(0, j);
  double y0 = position_(1, j);

  double t0_f = std::floor(t0s[j] / (k_mpc * dt_wbc)) * (k_mpc * dt_wbc);
  double t = t0_f - vertTime_;
  double d = t_swing[j] - 2 * vertTime_;
  double dt = k_mpc * dt_wbc;

  int N = static_cast<int>(std::round((t_swing[j] - t0_f) / dt)) + 1;
  MatrixN traj = MatrixN::Zero(3, N);

  // Coefficients for z (deterministic)
  double Tz = t_swing[j];
  Vector4 Az;
  Az(0, 0) = -stepHeight_[j] / (std::pow((Tz / 2), 3) * std::pow((Tz - Tz / 2), 3));
  Az(1, 0) = (3 * Tz * stepHeight_[j]) / (std::pow((Tz / 2), 3) * std::pow((Tz - Tz / 2), 3));
  Az(2, 0) = -(3 * std::pow(Tz, 2) * stepHeight_[j]) / (std::pow((Tz / 2), 3) * std::pow((Tz - Tz / 2), 3));
  Az(3, 0) = (std::pow(Tz, 3) * stepHeight_[j]) / (std::pow((Tz / 2), 3) * std::pow((Tz - Tz / 2), 3));

  // Get the next point
  for (int i = 0; i < N; i++) {
    double ev = t + dt * i;
    double evz = t0_f + dt * i;
    if (ev <= 0.0)  // Just vertical motion
    {
      traj(0, i) = x0;
      traj(1, i) = y0;
    } else {
      if (ev > d) {
        ev = d;
      }
      traj(0, i) = Ax(5, j) + Ax(4, j) * ev + Ax(3, j) * std::pow(ev, 2) + Ax(2, j) * std::pow(ev, 3) +
                   Ax(1, j) * std::pow(ev, 4) + Ax(0, j) * std::pow(ev, 5);
      traj(1, i) = Ay(5, j) + Ay(4, j) * ev + Ay(3, j) * std::pow(ev, 2) + Ay(2, j) * std::pow(ev, 3) +
                   Ay(1, j) * std::pow(ev, 4) + Ay(0, j) * std::pow(ev, 5);
    }
    traj(2, i) = Az(3, 0) * std::pow(evz, 3) + Az(2, 0) * std::pow(evz, 4) + Az(1, 0) * std::pow(evz, 5) +
                 Az(0, 0) * std::pow(evz, 6) + targetFootstep_(2, j);
  }

  return traj;
}