#include <example-robot-data/path.hpp>

#include "pinocchio/math/rpy.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/compute-all-terms.hpp"
#include "pinocchio/algorithm/frames.hpp"

#include "qrw/Solo3D/FootstepPlannerQP.hpp"

FootstepPlannerQP::FootstepPlannerQP()
    : gait_(NULL),
      g(9.81),
      L(0.155),
      heuristic_fb_(Vector3::Zero()),
      heuristic_(Vector3::Zero()),
      footsteps_(),
      Rz(MatrixN::Identity(3, 3)),
      Rz_tmp(MatrixN::Identity(3, 3)),
      dt_cum(),
      yaws(),
      dx(),
      dy(),
      q_dxdy(Vector3::Zero()),
      q_tmp(Vector3::Zero()),
      RPY_(Vector3::Zero()),
      pos_feet_(Matrix34::Zero()),
      q_FK_(Vector19::Zero()),
      k_mpc(0),
      feet_(),
      t0s(Vector4::Zero()),
      t_swing(Vector4::Zero()),
      weights_(VectorN::Zero(N)),
      b_voptim{Vector3::Zero()},
      delta_x{Vector3::Zero()},
      P_{MatrixN::Zero(N, N)},
      q_{VectorN::Zero(N)},
      G_{MatrixN::Zero(M, N)},
      h_{VectorN::Zero(M)},
      C_{MatrixN::Zero(N, 0)},
      d_{VectorN::Zero(0)},
      x{VectorN::Zero(N)},
      surfaceStatus_(false),
      useSL1M(true),
      surfaceIteration_(0) {
  // Empty
}

void FootstepPlannerQP::initialize(Params& params, Gait& gaitIn, Surface initialSurface_in) {
  params_ = &params;
  useSL1M = params.use_sl1m;
  dt = params.dt_mpc;
  dt_wbc = params.dt_wbc;
  h_ref = params.h_ref;
  n_steps = static_cast<int>(params.gait.rows());
  k_mpc = (int)std::round(params.dt_mpc / params.dt_wbc);
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
    b_footsteps_.push_back(Matrix34::Zero());
  }
  Rz(2, 2) = 1.0;

  // Surfaces initialization
  initialSurface_ = initialSurface_in;
  for (int foot = 0; foot < 4; foot++) {
    selectedSurfaces_.push_back(initialSurface_in);
  }

  // QP initialization
  qp.reset(N, 0, M);
  weights_.setZero(N);
  weights_ << 1000., 1000., 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
  P_.diagonal() << weights_;

  // Path to the robot URDF
  const std::string filename =
      std::string(EXAMPLE_ROBOT_DATA_MODEL_DIR "/solo_description/robots/solo12.urdf");

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

MatrixN FootstepPlannerQP::updateFootsteps(bool refresh, int k, VectorN const& q, Vector6 const& b_v,
                                           Vector6 const& b_vref) {
  if (q.rows() != 18) {
    throw std::runtime_error("q should be a vector of size 18 (base position + base RPY + joint)");
  }

  // Update location of feet in stance phase (for those which just entered stance phase)
  if (refresh && gait_->isNewPhase()) {
    updateNewContact(q);
  }

  // Compute location of footsteps
  return computeTargetFootstep(k, q.head(6), b_v, b_vref);
}

void FootstepPlannerQP::updateSurfaces(SurfaceVectorVector const& potentialSurfaces, SurfaceVector const& surfaces,
                                       bool const surfaceStatus, int const surfaceIteration) {
  surfaceStatus_ = surfaceStatus;
  surfaceIteration_ = surfaceIteration;
  surfaces_ = surfaces;
  potentialSurfaces_ = potentialSurfaces;
}

MatrixN FootstepPlannerQP::computeTargetFootstep(int k, Vector6 const& q, Vector6 const& b_v, Vector6 const& b_vref) {
  // Rotation matrix along z axis
  RPY_ = q.tail(3);
  double c = std::cos(RPY_(2));
  double s = std::sin(RPY_(2));
  Rz.topLeftCorner<2, 2>() << c, -s, s, c;

  // Current position in world frame, z = 0
  q_tmp = q.head(3);
  q_tmp(2) = 0.0;

  // ! b_vref and b_v corresponds to h_v, velocities in horizontal frame
  // b_v given in horizontal frame, b_vref given in base frame
  Vector3 RP_ = RPY_;
  RP_[2] = 0;  // Yaw taken into account later
  Vector6 h_vref;
  h_vref.head(3) = pinocchio::rpy::rpyToMatrix(RP_) * b_vref.head(3);
  h_vref.tail(3) = pinocchio::rpy::rpyToMatrix(RP_) * b_vref.tail(3);

  // Compute the desired location of footsteps over the prediction horizon
  computeFootsteps(k, b_v, h_vref);

  // Update desired location of footsteps on the ground
  updateTargetFootsteps();

  return o_targetFootstep_;
}

void FootstepPlannerQP::computeFootsteps(int k, Vector6 const& b_v, Vector6 const& b_vref) {
  for (uint i = 0; i < footsteps_.size(); i++) {
    footsteps_[i] = Matrix34::Zero();
    b_footsteps_[i] = Matrix34::Zero();
  }
  MatrixN gait = gait_->getCurrentGait();

  // Set current position of feet for feet in stance phase
  std::fill(footsteps_.begin(), footsteps_.end(), Matrix34::Zero());
  for (int j = 0; j < 4; j++) {
    if (gait(0, j) == 1.0) {
      footsteps_[0].col(j) = currentFootstep_.col(j);                               // world frame
      b_footsteps_[0].col(j) = Rz.transpose() * (currentFootstep_.col(j) - q_tmp);  // base frame
    }
  }

  // Cumulative time by adding the terms in the first column (remaining number of timesteps)
  // Get future yaw yaws compared to current position
  dt_cum(0) = dt_wbc * k;
  yaws(0) = b_vref(5) * dt_cum(0);  // base frame
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
  // Compute remaining time of the current flying phase
  update_remaining_time(k);

  // Update the footstep matrix depending on the different phases of the gait (swing & stance)
  int phase = 0;
  optimVector_.clear();
  for (int i = 1; i < gait.rows(); i++) {
    // Feet that were in stance phase and are still in stance phase do not move
    for (int j = 0; j < 4; j++) {
      if (gait(i - 1, j) * gait(i, j) > 0) {
        footsteps_[i].col(j) = footsteps_[i - 1].col(j);      // world frame
        b_footsteps_[i].col(j) = b_footsteps_[i - 1].col(j);  // base frame
      }
    }

    // Feet that were in swing phase and are now in stance phase need to be updated
    for (int j = 0; j < 4; j++) {
      if ((1 - gait(i - 1, j)) * gait(i, j) > 0) {
        // Offset to the future position
        q_dxdy << dx(i - 1, 0), dy(i - 1, 0), 0.0;  // q_dxdy from base frame

        // Get future desired position of footsteps
        computeNextFootstep(i, j, b_v, b_vref, heuristic_fb_, true);  // with feedback term
        computeNextFootstep(i, j, b_v, b_vref, heuristic_, false);    // without feeback term

        // Get desired position of footstep compared to current position
        Rz_tmp.setZero();
        double c = std::cos(yaws(i - 1));
        double s = std::sin(yaws(i - 1));
        Rz_tmp.topLeftCorner<2, 2>() << c, -s, s, c;

        // Use directly the heuristic method
        // footsteps_[i].col(j) = (Rz*Rz_tmp * heuristic_fb_ + q_tmp + Rz*q_dxdy).transpose();
        // b_footsteps_[i].col(j) = (Rz_tmp * heuristic_fb_ + q_dxdy).transpose();
        heuristic_fb_ = (Rz * Rz_tmp * heuristic_fb_ + q_tmp + Rz * q_dxdy).transpose();  // world, with feedback term
        heuristic_ = (Rz * Rz_tmp * heuristic_ + q_tmp + Rz * q_dxdy).transpose();  // world, without feedback term

        // Check if current flying phase
        bool flying_foot = false;
        for (int other_foot = 0; other_foot < (int)feet_.size(); other_foot++) {
          if (feet_[other_foot] == j) {
            flying_foot = true;
          }
        }

        if (flying_foot && phase == 0)  // feet currently in flying phase
        {
          if (t0s[j] < 10e-4 and k % k_mpc == 0)  // Beginning of flying phase
          {
            if (surfaceStatus_ && useSL1M) {
              selectedSurfaces_[j] = surfaces_[j];
            } else {
              selectedSurfaces_[j] = selectSurfaceFromPoint(heuristic_fb_, phase, j);
            }
          }
          optimData optim_data = {i, j, selectedSurfaces_[j], heuristic_, Rz_tmp};
          optimVector_.push_back(optim_data);
        } else {
          Surface sf_ = Surface();
          if (surfaceStatus_ && useSL1M) {
            sf_ = surfaces_[j];
          } else {
            sf_ = selectSurfaceFromPoint(heuristic_fb_, phase, j);
          }
          optimData optim_data = {i, j, sf_, heuristic_, Rz_tmp};
          optimVector_.push_back(optim_data);
        }
      }
    }

    if (!(gait.row(i - 1) - gait.row(i)).isZero()) {
      phase += 1;
    }
  }

  // Reset matrix Inequalities
  G_.setZero();
  h_.setZero();
  x.setZero();
  qp.reset(N, 0, M);

  // Adapting q_ vector with reference velocity
  q_(0) = -weights_(0) * b_vref(0);
  q_(1) = -weights_(1) * b_vref(1);

  // Convert problem to inequalities
  int iStart = 0;
  int foot = 0;
  for (uint id_l = 0; id_l < optimVector_.size(); id_l++) {
    iStart = surfaceInequalities(iStart, optimVector_[id_l].surface, optimVector_[id_l].constant_term, id_l,
                                 optimVector_[id_l].Rz_tmp);
    foot++;
  }
  status = qp.solve_quadprog(P_, q_, C_, d_, G_, h_, x);  // solve QP

  // Retrieve results
  b_voptim.head(2) = x.head(2);

  // Update the foostep matrix with the position optimised, for changing phase index
  for (uint id_l = 0; id_l < optimVector_.size(); id_l++) {
    int i = optimVector_[id_l].phase;
    int foot = optimVector_[id_l].foot;

    // Offset to the future position
    q_dxdy << dx(i - 1, 0), dy(i - 1, 0), 0.0;

    delta_x(0) = x(2 + 3 * id_l);
    delta_x(1) = x(2 + 3 * id_l + 1);
    delta_x(2) = x(2 + 3 * id_l + 2);

    footsteps_[i].col(foot) =
        optimVector_[id_l].constant_term - params_->k_feedback * Rz * optimVector_[id_l].Rz_tmp * b_voptim + delta_x;
    b_footsteps_[i].col(foot) = Rz.transpose() * (footsteps_[i].col(foot) - q_tmp);
  }

  // Update the next stance phase after the changing phase
  for (int i = 1; i < gait.rows(); i++) {
    // Feet that were in stance phase and are still in stance phase do not move
    for (int foot = 0; foot < 4; foot++) {
      if (gait(i - 1, foot) * gait(i, foot) > 0) {
        footsteps_[i].col(foot) = footsteps_[i - 1].col(foot);
        b_footsteps_[i].col(foot) = b_footsteps_[i - 1].col(foot);
      }
    }
  }
}

void FootstepPlannerQP::computeNextFootstep(int i, int j, Vector6 const& b_v, Vector6 const& b_vref, Vector3& footstep,
                                            bool feedback_term) {
  footstep.setZero();  // set to 0 the vector to fill

  double t_stance = gait_->getPhaseDuration(i, j);  // 1.0 for stance phase

  // Add symmetry term
  footstep = t_stance * 0.5 * b_v.head(3);

  // Add feedback term
  footstep += params_->k_feedback * b_v.head(3);
  if (feedback_term) {
    footstep += -params_->k_feedback * b_vref.head(3);
  }

  // Add centrifugal term
  Vector3 cross;
  cross << b_v(1) * b_vref(5) - b_v(2) * b_vref(4), b_v(2) * b_vref(3) - b_v(0) * b_vref(5), 0.0;
  footstep += 0.5 * std::sqrt(h_ref / g) * cross;

  // Legs have a limited length so the deviation has to be limited
  footstep(0) = std::min(footstep(0), L);
  footstep(0) = std::max(footstep(0), -L);
  footstep(1) = std::min(footstep(1), L);
  footstep(1) = std::max(footstep(1), -L);

  // Add shoulders
  Vector3 RP_ = RPY_;
  RP_[2] = 0;  // Yaw taken into account later
  footstep += pinocchio::rpy::rpyToMatrix(RP_) * footsteps_under_shoulders_.col(j);
  footstep += footsteps_offset_.col(j);

  // Remove Z component (working on flat ground)
  footstep(2) = 0.;
}

void FootstepPlannerQP::updateTargetFootsteps() {
  for (int i = 0; i < 4; i++) {
    int index = 0;
    while (footsteps_[index](0, i) == 0.0) {
      index++;
    }
    o_targetFootstep_.col(i) << footsteps_[index](0, i), footsteps_[index](1, i), footsteps_[index](2, i);
  }
}

void FootstepPlannerQP::updateNewContact(Vector18 const& q) {
  // Get position of the feet in world frame, using estimated state q
  q_FK_.head(3) = q.head(3);
  q_FK_.block(3, 0, 4, 1) =
      pinocchio::SE3::Quaternion(pinocchio::rpy::rpyToMatrix(q(3, 0), q(4, 0), q(5, 0))).coeffs();
  q_FK_.tail(12) = q.tail(12);

  // Update model and data of the robot
  pinocchio::forwardKinematics(model_, data_, q_FK_);
  pinocchio::updateFramePlacements(model_, data_);

  // Get data required by IK with Pinocchio
  for (int i = 0; i < 4; i++) {
    pos_feet_.col(i) = data_.oMf[foot_ids_[i]].translation();
  }

  // Refresh position with estimated position if foot is in stance phase
  for (int i = 0; i < 4; i++) {
    if (gait_->getCurrentGaitCoeff(0, i) == 1.0) {
      currentFootstep_.block(0, i, 2, 1) = pos_feet_.block(0, i, 2, 1);  // Get only x and y from IK
      currentFootstep_(2, i) = footsteps_[1](2, i);                      // Z from the height map
    }
  }
}

void FootstepPlannerQP::update_remaining_time(int k) {
  if ((k % k_mpc) == 0) {
    feet_.clear();
    t0s.setZero();

    // Indexes of feet in swing phase
    for (int i = 0; i < 4; i++) {
      if (gait_->getCurrentGait()(0, i) == 0) feet_.push_back(i);
    }
    // If no foot in swing phase
    if (feet_.size() == 0) return;

    // For each foot in swing phase get remaining duration of the swing phase
    for (int foot = 0; foot < (int)feet_.size(); foot++) {
      int i = feet_[foot];
      t_swing[i] = gait_->getPhaseDuration(0, feet_[foot]);
      double value = gait_->getElapsedTime(0, feet_[foot]);
      t0s[i] = std::max(0.0, value);
    }
  } else {
    // If no foot in swing phase
    if (feet_.size() == 0) return;

    // Increment of one time step for feet_ in swing phase
    for (int i = 0; i < (int)feet_.size(); i++) {
      double value = t0s[feet_[i]] + dt_wbc;
      t0s[feet_[i]] = std::max(0.0, value);
    }
  }
}

Surface FootstepPlannerQP::selectSurfaceFromPoint(Vector3 const& point, int phase, int moving_foot_index) {
  double sfHeight = 0.;
  bool surfaceFound = false;

  Surface sf = initialSurface_;

  if (surfaceIteration_ > 0) {
    SurfaceVector potentialSurfaces = potentialSurfaces_[moving_foot_index];
    for (uint i = 0; i < potentialSurfaces.size(); i++) {
      if (potentialSurfaces[i].hasPoint(point.head(2))) {
        double height = sf.getHeight(point.head(2));
        if (height > sfHeight) {
          sfHeight = height;
          sf = potentialSurfaces[i];
          surfaceFound = true;
        }
      }
    }
  }

  // The vertices has been ordered previously counter-clock wise, using qHull methods.
  // We could use hpp_fcl to compute this distance.
  Pair A = {0., 0.};
  Pair B = {0., 0.};
  Pair E = {point(0), point(1)};
  double distance = 100.0;
  double distance_tmp = 0.;
  if (not surfaceFound) {
    if (surfaceIteration_ > 0) {
      SurfaceVector potentialSurfaces = potentialSurfaces_[moving_foot_index];
      for (uint i = 0; i < potentialSurfaces.size(); i++) {
        MatrixN vertices = potentialSurfaces[i].getVertices();
        for (uint j = 0; j < vertices.rows(); j++) {
          A.F = vertices(j, 0);
          A.S = vertices(j, 1);
          B.F = vertices((j + 1) % vertices.rows(), 0);
          B.S = vertices((j + 1) % vertices.rows(), 1);
          distance_tmp = minDistance(A, B, E);
          if (distance_tmp < distance) {
            distance = distance_tmp;
            sf = potentialSurfaces[i];
          }
        }
      }
    }
  }
  return sf;
}

int FootstepPlannerQP::surfaceInequalities(int i_start, Surface const& surface, Vector3 const& next_ft, int id_l,
                                           Matrix3 Rz_tmp) {
  int n_rows = int(surface.getA().rows());
  MatrixN mat_tmp = MatrixN::Zero(n_rows, 3);
  mat_tmp = surface.getA() * Rz * Rz_tmp;
  G_.block(i_start, 0, n_rows, 2) = params_->k_feedback * mat_tmp.block(0, 0, n_rows, 2);
  G_.block(i_start, 2 + 3 * id_l, n_rows, 3) = -surface.getA();
  h_.segment(i_start, n_rows) = surface.getb() - surface.getA() * next_ft;

  return i_start + n_rows;
}

MatrixN FootstepPlannerQP::getFootsteps() { return vectorToMatrix(b_footsteps_); }
MatrixN FootstepPlannerQP::getTargetFootsteps() { return targetFootstep_; }

MatrixN FootstepPlannerQP::vectorToMatrix(std::vector<Matrix34> const& array) {
  MatrixN M = MatrixN::Zero(array.size(), 12);
  for (uint i = 0; i < array.size(); i++) {
    for (int j = 0; j < 4; j++) {
      M.row(i).segment<3>(3 * j) = array[i].col(j);
    }
  }
  return M;
}

// Function to return the minimum distance
// between a line segment AB and a point E
// https://www.geeksforgeeks.org/minimum-distance-from-a-point-to-the-line-segment-using-vectors/
double FootstepPlannerQP::minDistance(Pair const& A, Pair const& B, Pair const& E) {
  // vector AB
  Pair AB = {B.F - A.F, B.S - A.S};
  // AB.F = B.F - A.F;
  // AB.S = B.S - A.S;

  // vector BP
  Pair BE = {E.F - B.F, E.S - B.S};
  // BE.F = E.F - B.F;
  // BE.S = E.S - B.S;

  // vector AP
  Pair AE = {E.F - A.F, E.S - A.S};
  // AE.F = E.F - A.F, AE.S = E.S - A.S;

  // Variables to store dot product
  double AB_BE, AB_AE;

  // Calculating the dot product
  AB_BE = (AB.F * BE.F + AB.S * BE.S);
  AB_AE = (AB.F * AE.F + AB.S * AE.S);

  // Minimum distance from
  // point E to the line segment
  double reqAns = 0;

  // Case 1
  if (AB_BE > 0) {
    // Finding the magnitude
    double y = E.S - B.S;
    double x = E.F - B.F;
    reqAns = sqrt(x * x + y * y);
  }

  // Case 2
  else if (AB_AE < 0) {
    double y = E.S - A.S;
    double x = E.F - A.F;
    reqAns = sqrt(x * x + y * y);
  }

  // Case 3
  else {
    // Finding the perpendicular distance
    double x1 = AB.F;
    double y1 = AB.S;
    double x2 = AE.F;
    double y2 = AE.S;
    double mod = sqrt(x1 * x1 + y1 * y1);
    reqAns = abs(x1 * y2 - y1 * x2) / mod;
  }
  return reqAns;
}
