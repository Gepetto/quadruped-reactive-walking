#include "qrw/Solo3D/StatePlanner3D.hpp"

#include "pinocchio/algorithm/compute-all-terms.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/math/rpy.hpp"
#include "pinocchio/parsers/urdf.hpp"

StatePlanner3D::StatePlanner3D()
    : nStates_(0),
      dt_(0.),
      referenceHeight_(0.),
      maxVelocity_(),
      rpy_(Vector3::Zero()),
      Rz_(Matrix3::Zero()),
      DxDy_(Vector3::Zero()),
      referenceStates_(),
      dtVector_(),
      heightmap_(),
      rpyMap_(Vector3::Zero()),
      fit_(Vector3::Zero()),
      nSteps_(0),
      stepDuration_(0.),
      configs_(),
      config_(Vector7::Zero()),
      rpyConfig_(Vector3::Zero()) {
  // Empty
}

void StatePlanner3D::initialize(Params& params) {
  nStates_ = static_cast<int>(params.gait.rows());
  dt_ = params.dt_mpc;
  referenceHeight_ = params.h_ref;
  maxVelocity_ = params.max_velocity;
  referenceStates_ = MatrixN::Zero(12, 1 + nStates_);
  dtVector_ =
      VectorN::LinSpaced(nStates_, dt_, static_cast<double>(nStates_) * dt_);
  heightmap_.initialize(std::getenv("SOLO3D_ENV_DIR") +
                        params.environment_heightmap);
  nSteps_ = params.number_steps;
  stepDuration_ = params.T_gait / 2;
  configs_ = MatrixN::Zero(7, params.number_steps);
}

void StatePlanner3D::updateSurface(VectorN const& q, Vector6 const& vRef) {
  fit_ = heightmap_.fitSurface_(
      q(0), q(1));  // Update surface equality before new step
  rpyMap_(0) = std::atan2(fit_(1), 1.);
  rpyMap_(1) = -std::atan2(fit_(0), 1.);

  computeConfigurations(q, vRef);
}

void StatePlanner3D::computeReferenceStates(VectorN const& q, Vector6 const& v,
                                            Vector6 const& vRef) {
  if (q.rows() != 6) {
    throw std::runtime_error(
        "StatePlanner3D::computeReferenceStates: q should be a vector of size "
        "6");
  }

  rpy_ = q.tail(3);
  double c = std::cos(rpy_(2));
  double s = std::sin(rpy_(2));
  Rz_.topLeftCorner<2, 2>() << c, -s, s, c;

  // Update the current state
  referenceStates_(0, 0) = 0.0;      // In horizontal frame x = 0.0
  referenceStates_(1, 0) = 0.0;      // In horizontal frame y = 0.0
  referenceStates_(2, 0) = q(2, 0);  // We keep height
  referenceStates_.block(3, 0, 2, 1) = rpy_.head(2);  // We keep roll and pitch
  referenceStates_(5, 0) = 0.0;  // In horizontal frame yaw = 0.0
  referenceStates_.block(6, 0, 3, 1) = v.head(3);
  referenceStates_.block(9, 0, 3, 1) = v.tail(3);

  for (int i = 0; i < nStates_; i++) {
    if (std::abs(vRef(5)) >= 0.001) {
      referenceStates_(0, 1 + i) =
          (vRef(0) * std::sin(vRef(5) * dtVector_(i)) +
           vRef(1) * (std::cos(vRef(5) * dtVector_(i)) - 1.0)) /
          vRef(5);
      referenceStates_(1, 1 + i) =
          (vRef(1) * std::sin(vRef(5) * dtVector_(i)) -
           vRef(0) * (std::cos(vRef(5) * dtVector_(i)) - 1.0)) /
          vRef(5);
    } else {
      referenceStates_(0, 1 + i) = vRef(0) * dtVector_(i);
      referenceStates_(1, 1 + i) = vRef(1) * dtVector_(i);
    }
    referenceStates_(0, 1 + i) += referenceStates_(0, 0);
    referenceStates_(1, 1 + i) += referenceStates_(1, 0);

    referenceStates_(5, 1 + i) = vRef(5) * dtVector_(i);

    referenceStates_(6, 1 + i) =
        vRef(0) * std::cos(referenceStates_(5, 1 + i)) -
        vRef(1) * std::sin(referenceStates_(5, 1 + i));
    referenceStates_(7, 1 + i) =
        vRef(0) * std::sin(referenceStates_(5, 1 + i)) +
        vRef(1) * std::cos(referenceStates_(5, 1 + i));

    referenceStates_(11, 1 + i) = vRef(5);

    // Update according to heightmap
    DxDy_(0) = referenceStates_(0, i + 1);
    DxDy_(1) = referenceStates_(1, i + 1);
    DxDy_ = Rz_ * DxDy_ + q.head(3);  // world frame

    referenceStates_(2, 1 + i) =
        fit_(0) * DxDy_(0) + fit_(1) * DxDy_(1) + fit_(2) + referenceHeight_;

    referenceStates_(3, 1 + i) = 1.0 * (rpyMap_[0] * std::cos(-rpy_[2]) -
                                        rpyMap_[1] * std::sin(-rpy_[2]));
    referenceStates_(4, 1 + i) = 1.0 * (rpyMap_[0] * std::sin(-rpy_[2]) +
                                        rpyMap_[1] * std::cos(-rpy_[2]));
  }

  // Update velocities according to heightmap
  for (int i = 0; i < nStates_; i++) {
    if (i == 0) {
      referenceStates_(8, 1 + i) = std::max(
          std::min((referenceStates_(2, 1) - q[2]) / dt_, maxVelocity_[2]),
          -maxVelocity_[2]);
      referenceStates_(9, 1 + i) = std::max(
          std::min((referenceStates_(3, 1) - rpy_[0]) / dt_, maxVelocity_[0]),
          -maxVelocity_[0]);
      referenceStates_(10, 1 + i) = std::max(
          std::min((referenceStates_(4, 1) - rpy_[1]) / dt_, maxVelocity_[1]),
          -maxVelocity_[1]);
    } else {
      referenceStates_(9, 1 + i) = 0.;
      referenceStates_(10, 1 + i) = 0.;
      referenceStates_(8, 1 + i) =
          (referenceStates_(2, 2) - referenceStates_(2, 1)) / dt_;
    }
  }
}

void StatePlanner3D::computeConfigurations(VectorN const& q,
                                           Vector6 const& vRef) {
  for (int i = 0; i < nSteps_; i++) {
    double dt_config =
        stepDuration_ * (i + 2);  // Delay of 2 phase of contact for MIP

    Vector2 dxdy;

    if (std::abs(vRef(5)) >= 0.001) {
      dxdy(0) = (vRef(0) * std::sin(vRef(5) * dt_config) +
                 vRef(1) * (std::cos(vRef(5) * dt_config) - 1.0)) /
                vRef(5);
      dxdy(1) = (vRef(1) * std::sin(vRef(5) * dt_config) -
                 vRef(0) * (std::cos(vRef(5) * dt_config) - 1.0)) /
                vRef(5);
    } else {
      dxdy(0) = vRef(0) * dt_config;
      dxdy(1) = vRef(1) * dt_config;
    }
    configs_(0, i) = std::cos(q(5)) * dxdy(0) -
                     std::sin(q(5)) * dxdy(1);  // Yaw rotation for dx
    configs_(1, i) = std::sin(q(5)) * dxdy(0) +
                     std::cos(q(5)) * dxdy(1);  // Yaw rotation for dy
    configs_.block(0, i, 2, 1) += q.head(2);    // Add initial position

    configs_(2, i) = fit_(0) * configs_(0, i) + fit_(1) * configs_(1, i) +
                     fit_(2) + referenceHeight_;
    rpyConfig_(2) = q(5) + vRef(5) * dt_config;
    rpyConfig_(0) = rpyMap_(0) * std::cos(rpyConfig_(2)) -
                    rpyMap_(1) * std::sin(rpyConfig_(2));
    rpyConfig_(1) = rpyMap_(0) * std::sin(rpyConfig_(2)) +
                    rpyMap_(1) * std::cos(rpyConfig_(2));

    configs_.block(3, i, 4, 1) =
        pinocchio::SE3::Quaternion(pinocchio::rpy::rpyToMatrix(rpyConfig_))
            .coeffs();
  }
}
