#include "qrw/StatePlanner3D.hpp"

StatePlanner3D::StatePlanner3D() : dt_(0.0), h_ref_(0.0), n_steps_(0), RPY_(Vector3::Zero()) {
  // Empty
}

void StatePlanner3D::initialize(Params& params) {
  dt_ = params.dt_mpc;
  h_ref_ = params.h_ref;
  n_steps_ = static_cast<int>(params.gait.rows());
  T_step_ = params.T_gait / 2;
  referenceStates_ = MatrixN::Zero(12, 1 + n_steps_);
  dt_vector_ = VectorN::LinSpaced(n_steps_, dt_, static_cast<double>(n_steps_) * dt_);
  heightmap_.initialize(params.environment_heightmap);
  configs = MatrixN::Zero(7,n_surface_configs);
  Rz = Matrix3::Zero();
  q_dxdy = Vector3::Zero();
}

void StatePlanner3D::computeReferenceStates(VectorN const& q, Vector6 const& v, Vector6 const& vref, int is_new_step) {
  if (q.rows() != 6) {
    throw std::runtime_error("q should be a vector of size 6");
  }
  if (is_new_step) {
    heightmap_.update_mean_surface(q(0), q(1));  // Update surface equality before new step
    rpy_map(0) = -std::atan2(heightmap_.surface_eq(1), 1.);
    rpy_map(1) = -std::atan2(heightmap_.surface_eq(0), 1.);
    compute_configurations(q,vref);
  }

  RPY_ = q.tail(3);
  double c = std::cos(RPY_(2));
  double s = std::sin(RPY_(2));
  Rz.topLeftCorner<2, 2>() << c, -s, s, c;

  // Update the current state
  referenceStates_(0, 0) = 0.0;                       // In horizontal frame x = 0.0
  referenceStates_(1, 0) = 0.0;                       // In horizontal frame y = 0.0
  referenceStates_(2, 0) = q(2, 0);                   // We keep height
  referenceStates_.block(3, 0, 2, 1) = RPY_.head(2);  // We keep roll and pitch
  referenceStates_(5, 0) = 0.0;                       // In horizontal frame yaw = 0.0
  referenceStates_.block(6, 0, 3, 1) = v.head(3);
  referenceStates_.block(9, 0, 3, 1) = v.tail(3);

  for (int i = 0; i < n_steps_; i++) {
    if (std::abs(vref(5)) >= 0.001) {
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

    referenceStates_(2, 1 + i) = h_ref_;

    referenceStates_(5, 1 + i) = vref(5) * dt_vector_(i);

    referenceStates_(6, 1 + i) =
        vref(0) * std::cos(referenceStates_(5, 1 + i)) - vref(1) * std::sin(referenceStates_(5, 1 + i));
    referenceStates_(7, 1 + i) =
        vref(0) * std::sin(referenceStates_(5, 1 + i)) + vref(1) * std::cos(referenceStates_(5, 1 + i));

    // referenceStates_(5, 1 + i) += RPY_(2);

    referenceStates_(11, 1 + i) = vref(5);

    // Update according to heightmap
    q_dxdy(0) = referenceStates_(0, i + 1);
    q_dxdy(1) = referenceStates_(1, i + 1);
    q_dxdy = Rz*q_dxdy + q.head(3); // world frame

    int idx = heightmap_.map_x(q_dxdy(0));
    int idy = heightmap_.map_y(q_dxdy(1));
    double z = heightmap_.surface_eq(0) * heightmap_.x_(idx) + heightmap_.surface_eq(1) * heightmap_.y_(idy) +
               heightmap_.surface_eq(2);
    referenceStates_(2, 1 + i) = h_ref_ + z;

    referenceStates_(3, 1 + i) = rpy_map[0] * std::cos(RPY_[2]) - rpy_map[1] * std::sin(RPY_[2]);
    referenceStates_(4, 1 + i) = rpy_map[0] * std::sin(RPY_[2]) + rpy_map[1] * std::cos(RPY_[2]);
  }

  // Update velocities according to heightmap
  for (int i = 0; i < n_steps_; i++) {
    if (i == 0) {
      referenceStates_(8, 1) = std::max(std::min((referenceStates_(2, 1) - q[2]) / dt_, v_max_z), -v_max_z);
      referenceStates_(9, 1) = std::max(std::min((referenceStates_(3, 1) - RPY_[0]) / dt_, v_max), -v_max);
      referenceStates_(10, 1) = std::max(std::min((referenceStates_(4, 1) - RPY_[1]) / dt_, v_max), -v_max);
    } else {
      referenceStates_(9, 1 + i) = 0.;
      referenceStates_(10, 1 + i) = 0.;
      referenceStates_(8, 1 + i) = (referenceStates_(2, 2) - referenceStates_(2, 1)) / dt_;
    }
  }
}

void StatePlanner3D::compute_configurations(VectorN const& q, Vector6 const& vref) {

  pinocchio::SE3::Quaternion quat_;
  for (int i = 0; i < n_surface_configs; i++) {
    Vector7 config_ = Vector7::Zero();
    // TODO : Not sure if (i+1)*T_step --> next step for MIP, not current
    double dt_config = T_step_ * i;
    // TODO : Not sure, take into account height, maybe useless since most constraints desactivated in MIP
    config_.head(3) = q.head(3);
    if (std::abs(vref(5)) >= 0.001) {
      config_(0) +=
          (vref(0) * std::sin(vref(5) * dt_config) + vref(1) * (std::cos(vref(5) * dt_config) - 1.0)) / vref(5);
      config_(1) +=
          (vref(1) * std::sin(vref(5) * dt_config) - vref(0) * (std::cos(vref(5) * dt_config) - 1.0)) / vref(5);
    } else {
      config_(0) += vref(0) * dt_config;
      config_(1) += vref(1) * dt_config;
    }

    Vector3 rpy_config = Vector3::Zero();
    rpy_config(2) = q(5) + vref(5) * dt_config;

    // Update according to heightmap
    int idx = heightmap_.map_x(config_(0));
    int idy = heightmap_.map_y(config_(1));
    config_(2) = heightmap_.surface_eq(0) * heightmap_.x_(idx) + heightmap_.surface_eq(1) * heightmap_.y_(idy) +
                heightmap_.surface_eq(2) + h_ref_;

    rpy_config(0) = rpy_map[0] * std::cos(rpy_config[2]) - rpy_map[1] * std::sin(rpy_config[2]);
    rpy_config(1) = rpy_map[0] * std::sin(rpy_config[2]) + rpy_map[1] * std::cos(rpy_config[2]);

    quat_ = pinocchio::SE3::Quaternion(pinocchio::rpy::rpyToMatrix(rpy_config));
    config_.tail(4) = quat_.coeffs();
    // configs.push_back(config_);
    configs.block(0,i,7,1) = config_;
  }
}
