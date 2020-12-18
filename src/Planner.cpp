#include "quadruped-reactive-walking/Planner.hpp"

Planner::Planner(double dt_in, double dt_tsid_in, double T_gait_in, double T_mpc_in, int k_mpc_in, bool on_solo8_in,
                 double h_ref_in, const Eigen::MatrixXd &fsteps_in) {
  // Parameters from the main controller
  dt = dt_in;
  dt_tsid = dt_tsid_in;
  T_gait = T_gait_in;
  T_mpc = T_mpc_in;
  k_mpc = k_mpc_in;
  on_solo8 = on_solo8_in;
  h_ref = h_ref_in;

  // Position of shoulders in base frame
  shoulders << 0.1946, 0.1946, -0.1946, -0.1946, 0.14695, -0.14695, 0.14695, -0.14695, 0.0, 0.0, 0.0, 0.0;
  // Order of feet/legs: FL, FR, HL, HR

  // By default contacts are at the vertical of shoulders
  Eigen::Map<Eigen::Matrix<double, 1, 12>> v1(shoulders.data(), shoulders.size());
  o_feet_contact << v1;
  footsteps_target.block(0, 0, 2, 4) = shoulders.block(0, 0, 2, 4);

  // Predefining quantities
  n_steps = (int)std::lround(T_mpc / dt);
  dt_vector = Eigen::VectorXd::LinSpaced(n_steps, dt, T_mpc).transpose();
  R(2, 2) = 1.0;
  R_1(2, 2) = 1.0;

  // Initialize xref matrix
  xref = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(12, 1 + n_steps);

  // Create gait matrix
  // create_walk();
  create_trot();

  // Initialisation of other gait matrices based on previous gait matrix
  create_gait_f();

  // For foot trajectory generator
  goals << fsteps_in.block(0, 0, 3, 4);
  mgoals.row(0) << fsteps_in.block(0, 0, 1, 4);
  mgoals.row(3) << fsteps_in.block(1, 0, 1, 4);

  // One foot trajectory generator per leg
  for (int i = 0; i < 4; i++) {
    myTrajGen.push_back(TrajGen(max_height_feet, t_lock_before_touchdown, shoulders(0, i), shoulders(1, i)));
  }
}

Planner::Planner() {}

void Planner::Print() {
  /* To print stuff for visualisation or debug */

  std::cout << "------" << std::endl;
  std::cout << gait_p.block(0, 0, 6, 5) << std::endl;
  std::cout << "-" << std::endl;
  std::cout << gait_f.block(0, 0, 6, 5) << std::endl;
  std::cout << "-" << std::endl;
  std::cout << gait_f_des.block(0, 0, 6, 5) << std::endl;
}

int Planner::create_walk() {
  /* Create a slow walking gait, raising and moving only one foot at a time */

  // Number of timesteps in 1/4th period of gait
  int N = (int)std::lround(0.25 * T_gait / dt);

  gait_f_des = Eigen::Matrix<double, N0_gait, 5>::Zero();
  gait_f_des.block(0, 0, 4, 1) << N, N, N, N;
  fsteps.block(0, 0, 4, 1) = gait_f_des.block(0, 0, 4, 1);

  // Set stance and swing phases
  // Coefficient (i, j) is equal to 0.0 if the j-th feet is in swing phase during the i-th phase
  // Coefficient (i, j) is equal to 1.0 if the j-th feet is in stance phase during the i-th phase
  gait_f_des.block(0, 1, 1, 4) << 0.0, 1.0, 1.0, 1.0;
  gait_f_des.block(1, 1, 1, 4) << 1.0, 0.0, 1.0, 1.0;
  gait_f_des.block(2, 1, 1, 4) << 1.0, 1.0, 0.0, 1.0;
  gait_f_des.block(3, 1, 1, 4) << 1.0, 1.0, 1.0, 0.0;

  return 0;
}

int Planner::create_trot() {
  /* Create a trot gait with diagonaly opposed legs moving at the same time */

  // Number of timesteps in a half period of gait
  int N = (int)std::lround(0.5 * T_gait / dt);

  gait_f_des = Eigen::Matrix<double, N0_gait, 5>::Zero();
  gait_f_des.block(0, 0, 2, 1) << N, N;
  fsteps.block(0, 0, 2, 1) = gait_f_des.block(0, 0, 2, 1);

  // Set stance and swing phases
  // Coefficient (i, j) is equal to 0.0 if the j-th feet is in swing phase during the i-th phase
  // Coefficient (i, j) is equal to 1.0 if the j-th feet is in stance phase during the i-th phase
  gait_f_des(0, 1) = 1.0;
  gait_f_des(0, 4) = 1.0;
  gait_f_des(1, 2) = 1.0;
  gait_f_des(1, 3) = 1.0;

  return 0;
}

int Planner::create_pacing() {
  /* Create a pacing gait with legs on the same side (left or right) moving at the same time */

  // Number of timesteps in a half period of gait
  int N = (int)std::lround(0.5 * T_gait / dt);

  gait_f_des = Eigen::Matrix<double, N0_gait, 5>::Zero();
  gait_f_des.block(0, 0, 2, 1) << N, N;
  fsteps.block(0, 0, 2, 1) = gait_f_des.block(0, 0, 2, 1);

  // Set stance and swing phases
  // Coefficient (i, j) is equal to 0.0 if the j-th feet is in swing phase during the i-th phase
  // Coefficient (i, j) is equal to 1.0 if the j-th feet is in stance phase during the i-th phase
  gait_f_des(0, 1) = 1.0;
  gait_f_des(0, 3) = 1.0;
  gait_f_des(1, 2) = 1.0;
  gait_f_des(1, 4) = 1.0;

  return 0;
}

int Planner::create_bounding() {
  /* Create a bounding gait with legs on the same side (front or hind) moving at the same time */

  // Number of timesteps in a half period of gait
  int N = (int)std::lround(0.5 * T_gait / dt);

  gait_f_des = Eigen::Matrix<double, N0_gait, 5>::Zero();
  gait_f_des.block(0, 0, 2, 1) << N, N;
  fsteps.block(0, 0, 2, 1) = gait_f_des.block(0, 0, 2, 1);

  // Set stance and swing phases
  // Coefficient (i, j) is equal to 0.0 if the j-th feet is in swing phase during the i-th phase
  // Coefficient (i, j) is equal to 1.0 if the j-th feet is in stance phase during the i-th phase
  gait_f_des(0, 1) = 1.0;
  gait_f_des(0, 2) = 1.0;
  gait_f_des(1, 3) = 1.0;
  gait_f_des(1, 4) = 1.0;

  return 0;
}

int Planner::create_static() {
  /* Create a static gait with all legs in stance phase */

  // Number of timesteps in a half period of gait
  int N = (int)std::lround(T_gait / dt);

  gait_f_des = Eigen::Matrix<double, N0_gait, 5>::Zero();
  gait_f_des(0, 0) = N;
  fsteps(0, 0) = gait_f_des(0, 0);

  // Set stance and swing phases
  // Coefficient (i, j) is equal to 0.0 if the j-th feet is in swing phase during the i-th phase
  // Coefficient (i, j) is equal to 1.0 if the j-th feet is in stance phase during the i-th phase
  gait_f_des(0, 1) = 1.0;
  gait_f_des(0, 2) = 1.0;
  gait_f_des(0, 3) = 1.0;
  gait_f_des(0, 4) = 1.0;

  return 0;
}

int Planner::create_gait_f() {
  /* Initialize content of the gait matrix based on the desired gait, the gait period and
  the length of the prediciton horizon */

  double sum = 0.0;
  double offset = 0.0;
  int i = 0;
  int j = 0;

  // Fill future gait matrix
  while (sum < (T_mpc / dt)) {
    gait_f.row(j) = gait_f_des.row(i);
    sum += gait_f_des(i, 0);
    offset += gait_f_des(i, 0);
    i++;
    j++;
    if (gait_f_des(i, 0) == 0) {
      i = 0;
      offset = 0.0;
    }  // Loop back if T_mpc longer than gait duration
  }

  // Remove excess time steps
  gait_f(j - 1, 0) -= sum - (T_mpc / dt);
  offset -= sum - (T_mpc / dt);

  // Age future desired gait to take into account what has been put in the future gait matrix
  j = 1;
  while (gait_f_des(j, 0) > 0.0) {
    j++;
  }

  for (double k = 0; k < offset; k++) {
    if ((gait_f_des.block(0, 1, 1, 4)).isApprox(gait_f_des.block(j - 1, 1, 1, 4))) {
      gait_f_des(j - 1, 0) += 1.0;
    } else {
      gait_f_des.row(j) = gait_f_des.row(0);
      gait_f_des(j, 0) = 1.0;
      j++;
    }
    if (gait_f_des(0, 0) == 1.0) {
      gait_f_des.block(0, 0, N0_gait - 1, 5) = gait_f_des.block(1, 0, N0_gait - 1, 5);
      j--;
    } else {
      gait_f_des(0, 0) -= 1.0;
    }
  }

  return 0;
}

int Planner::compute_footsteps(Eigen::MatrixXd q_cur, Eigen::MatrixXd v_cur, Eigen::MatrixXd v_ref) {
  /* Compute a X by 13 matrix containing the remaining number of steps of each phase of the gait (first column)
  and the [x, y, z]^T desired position of each foot for each phase of the gait (12 other columns).
  For feet currently touching the ground the desired position is where they currently are.

  Args:
    q_cur (7x1 array): current position vector of the flying base in world frame (linear and angular stacked)
    v_cur (6x1 array): current velocity vector of the flying base in world frame (linear and angular stacked)
    v_ref (6x1 array): desired velocity vector of the flying base in world frame (linear and angular stacked)
  */

  fsteps = Eigen::Matrix<double, N0_gait, 13>::Zero();
  fsteps.col(0) = gait_f.col(0);

  // Set current position of feet for feet in stance phase
  for (int j = 0; j < 4; j++) {
    if (gait_f(0, 1 + j) == 1.0) {
      fsteps.block(0, 1 + 3 * j, 1, 3) = o_feet_contact.block(0, 3 * j, 1, 3);
    }
  }

  // Cumulative time by adding the terms in the first column (remaining number of timesteps)
  // Get future yaw angle compared to current position
  dt_cum(0, 0) = gait_f(0, 0) * dt;
  angle(0, 0) = v_ref(5, 0) * dt_cum(0, 0) + RPY(2, 0);
  for (int j = 1; j < N0_gait; j++) {
    dt_cum(j, 0) = dt_cum(j - 1, 0) + gait_f(j, 0) * dt;
    angle(j, 0) = v_ref(5, 0) * dt_cum(j, 0) + RPY(2, 0);
  }

  // Displacement following the reference velocity compared to current position
  if (v_ref(5, 0) != 0) {
    for (int j = 0; j < N0_gait; j++) {
      dx(j, 0) = (v_cur(0, 0) * std::sin(v_ref(5, 0) * dt_cum(j, 0)) +
                  v_cur(1, 0) * (std::cos(v_ref(5, 0) * dt_cum(j, 0)) - 1.0)) /
                 v_ref(5, 0);
      dy(j, 0) = (v_cur(1, 0) * std::sin(v_ref(5, 0) * dt_cum(j, 0)) -
                  v_cur(0, 0) * (std::cos(v_ref(5, 0) * dt_cum(j, 0)) - 1.0)) /
                 v_ref(5, 0);
    }
  } else {
    for (int j = 0; j < N0_gait; j++) {
      dx(j, 0) = v_cur(0, 0) * dt_cum(j, 0);
      dy(j, 0) = v_cur(1, 0) * dt_cum(j, 0);
    }
  }

  // Get current and reference velocities in base frame (rotated yaw)
  double c = std::cos(RPY(2, 0));
  double s = std::sin(RPY(2, 0));
  R_1.block(0, 0, 2, 2) << c, s, -s, c;  // already transposed here
  b_v_cur = R_1 * v_cur.block(0, 0, 3, 1);
  b_v_ref.block(0, 0, 3, 1) = R_1 * v_ref.block(0, 0, 3, 1);
  b_v_ref.block(3, 0, 3, 1) = R_1 * v_ref.block(3, 0, 3, 1);

  // Update the footstep matrix depending on the different phases of the gait (swing & stance)
  int i = 1;
  while (gait_f(i, 0) != 0) {
    // Feet that were in stance phase and are still in stance phase do not move
    for (int j = 0; j < 4; j++) {
      if (gait_f(i - 1, 1 + j) * gait_f(i, 1 + j) > 0) {
        fsteps.block(i, 1 + 3 * j, 1, 3) = fsteps.block(i - 1, 1 + 3 * j, 1, 3);
      }
    }

    // Current position without height
    q_tmp << q_cur(0, 0), q_cur(1, 0), 0.0;

    // Feet that were in swing phase and are now in stance phase need to be updated
    for (int j = 0; j < 4; j++) {
      if ((1 - gait_f(i - 1, 1 + j)) * gait_f(i, 1 + j) > 0) {
        // Offset to the future position
        q_dxdy << dx(i - 1, 0), dy(i - 1, 0), 0.0;

        // Get future desired position of footsteps
        compute_next_footstep(i, j);

        // Get desired position of footstep compared to current position
        double c = std::cos(angle(i - 1, 0));
        double s = std::sin(angle(i - 1, 0));
        R.block(0, 0, 2, 2) << c, -s, s, c;

        fsteps.block(i, 1 + 3 * j, 1, 3) = (R * next_footstep.col(j) + q_tmp + q_dxdy).transpose();
      }
    }

    i++;
  }

  return 0;
}

double Planner::get_stance_swing_duration(int i, int j, double value) {
  /* Compute the remaining and total duration of a swing phase or a stance phase based
  on the content of the gait matrix

  Args:
    i (int): considered phase (row of the gait matrix)
    j (int): considered foot (col of the gait matrix)
    value (double): 0.0 for swing phase detection, 1.0 for stance phase detection
  */

  double t_phase = gait_f(i, 0);
  int a = i;

  // Looking for the end of the swing/stance phase in gait_f
  while ((gait_f(i + 1, 0) > 0.0) && (gait_f(i + 1, 1 + j) == value)) {
    i++;
    t_phase += gait_f(i, 0);
  }
  // If we reach the end of gait_f we continue looking for the end of the swing/stance phase in gait_f_des
  if (gait_f(i + 1, 0) == 0.0) {
    int k = 0;
    while ((gait_f_des(k, 0) > 0.0) && (gait_f_des(k, 1 + j) == value)) {
      t_phase += gait_f_des(k, 0);
      k++;
    }
  }
  // We suppose that we found the end of the swing/stance phase either in gait_f or gait_f_des

  t_remaining = t_phase;

  // Looking for the beginning of the swing/stance phase in gait_f
  while ((a > 0) && (gait_f(a - 1, 1 + j) == value)) {
    a--;
    t_phase += gait_f(a, 0);
  }
  // If we reach the end of gait_f we continue looking for the beginning of the swing/stance phase in gait_p
  if (a == 0) {
    while ((gait_p(a, 0) > 0.0) && (gait_p(a, 1 + j) == value)) {
      t_phase += gait_p(a, 0);
      a++;
    }
  }
  // We suppose that we found the beginning of the swing/stance phase either in gait_f or gait_p

  return t_phase * dt;  // Take into account time step value
}

int Planner::compute_next_footstep(int i, int j) {
  /* Compute the target location on the ground of a given foot for an upcoming stance phase

  Args:
    i (int): considered phase (row of the gait matrix)
    j (int): considered foot (col of the gait matrix)
  */

  double t_stance = get_stance_swing_duration(i, j, 1.0);  // 1.0 for stance phase

  // Add symmetry term
  next_footstep.col(j) = t_stance * 0.5 * b_v_cur;

  // Add feedback term
  next_footstep.col(j) += k_feedback * (b_v_cur - b_v_ref.block(0, 0, 3, 1));

  // Add centrifugal term
  cross << b_v_cur(1, 0) * b_v_ref(5, 0) - b_v_cur(2, 0) * b_v_ref(4, 0),
      b_v_cur(2, 0) * b_v_ref(3, 0) - b_v_cur(0, 0) * b_v_ref(5, 0), 0.0;
  next_footstep.col(j) += 0.5 * std::sqrt(h_ref / g) * cross;

  // Legs have a limited length so the deviation has to be limited
  if (next_footstep(0, j) > L) {
    next_footstep(0, j) = L;
  } else if (next_footstep(0, j) < -L) {
    next_footstep(0, j) = -L;
  }

  if (next_footstep(1, j) > L) {
    next_footstep(1, j) = L;
  } else if (next_footstep(1, j) < -L) {
    next_footstep(1, j) = -L;
  }

  // Add shoulders
  next_footstep.col(j) += shoulders.col(j);

  // Remove Z component (working on flat ground)
  next_footstep.row(2) = Eigen::Matrix<double, 1, 4>::Zero();

  return 0;
}

int Planner::getRefStates(Eigen::MatrixXd q, Eigen::MatrixXd v, Eigen::MatrixXd vref, double z_average) {
  /* Compute the reference trajectory of the CoM for each time step of the
  predition horizon. The ouput is a matrix of size 12 by (N+1) with N the number
  of time steps in the gait cycle (T_gait/dt) and 12 the position, orientation,
  linear velocity and angular velocity vertically stacked. The first column contains
  the current state while the remaining N columns contains the desired future states.

  Args:
    q (7x1 array): current position vector of the flying base in world frame (linear and angular stacked)
    v (6x1 array): current velocity vector of the flying base in world frame (linear and angular stacked)
    vref (6x1 array): desired velocity vector of the flying base in world frame (linear and angular stacked)
    z_average (double): average height of feet currently in stance phase
  */

  // Update yaw and yaw velocity
  xref.block(5, 1, 1, n_steps) = vref(5, 0) * dt_vector;
  for (int i = 0; i < n_steps; i++) {
    xref(11, 1 + i) = vref(5, 0);
  }

  // Update x and y velocities taking into account the rotation of the base over the prediction horizon
  for (int i = 0; i < n_steps; i++) {
    xref(6, 1 + i) = vref(0, 0) * std::cos(xref(5, 1 + i)) - vref(1, 0) * std::sin(xref(5, 1 + i));
    xref(7, 1 + i) = vref(0, 0) * std::sin(xref(5, 1 + i)) + vref(1, 0) * std::cos(xref(5, 1 + i));
  }

  // Update x and y depending on x and y velocities (cumulative sum)
  if (vref(5, 0) != 0) {
    for (int i = 0; i < n_steps; i++) {
      xref(0, 1 + i) = (vref(0, 0) * std::sin(vref(5, 0) * dt_vector(0, i)) +
                        vref(1, 0) * (std::cos(vref(5, 0) * dt_vector(0, i)) - 1.0)) /
                       vref(5, 0);
      xref(1, 1 + i) = (vref(1, 0) * std::sin(vref(5, 0) * dt_vector(0, i)) -
                        vref(0, 0) * (std::cos(vref(5, 0) * dt_vector(0, i)) - 1.0)) /
                       vref(5, 0);
    }
  } else {
    for (int i = 0; i < n_steps; i++) {
      xref(0, 1 + i) = vref(0, 0) * dt_vector(0, i);
      xref(1, 1 + i) = vref(1, 0) * dt_vector(0, i);
    }
  }

  for (int i = 0; i < n_steps; i++) {
    xref(5, 1 + i) += RPY(2, 0);
    xref(2, 1 + i) = h_ref + z_average;
    xref(8, 1 + i) = 0.0;
  }

  // No need to update Z velocity as the reference is always 0
  // No need to update roll and roll velocity as the reference is always 0 for those
  // No need to update pitch and pitch velocity as the reference is always 0 for those

  // Update the current state
  xref.block(0, 0, 3, 1) = q.block(0, 0, 3, 1);
  xref.block(3, 0, 3, 1) = RPY;
  xref.block(6, 0, 3, 1) = v.block(0, 0, 3, 1);
  xref.block(9, 0, 3, 1) = v.block(3, 0, 3, 1);

  for (int i = 0; i < n_steps; i++) {
    xref(0, 1 + i) += xref(0, 0);
    xref(1, 1 + i) += xref(1, 0);
  }

  if (is_static) {
    Eigen::Matrix<double, 3, 1> RPY;
    Eigen::Quaterniond quat(q_static(6, 0), q_static(3, 0), q_static(4, 0), q_static(5, 0));  // w, x, y, z
    RPY << pinocchio::rpy::matrixToRpy(quat.toRotationMatrix());
    for (int i = 0; i < n_steps; i++) {
      xref.block(0, 1 + i, 3, 1) = q_static.block(0, 0, 3, 1);
      xref.block(3, 1 + i, 3, 1) = RPY;
    }
  }

  return 0;
}

int Planner::update_target_footsteps() {
  /* Update desired location of footsteps using information coming from the footsteps planner */

  for (int i = 0; i < 4; i++) {
    // Index of the first non-empty line
    int index = 0;
    while (fsteps(index, 1 + 3 * i) == 0.0) {
      index++;
    }

    // Copy fsteps
    footsteps_target.col(i) = fsteps.block(index, 1 + 3 * i, 1, 2).transpose();
  }

  return 0;
}

int Planner::update_trajectory_generator(int k, double h_estim) {
  /* Update the 3D desired position for feet in swing phase by using a 5-th order polynomial that lead them
  to the desired position on the ground (computed by the footstep planner)

  Args:
    k (int): number of time steps since the start of the simulation
    h_estim (double): estimated height of the base
  */

  if ((k % k_mpc) == 0) {
    // Indexes of feet in swing phase
    feet.clear();
    for (int i = 0; i < 4; i++) {
      if (gait_f(0, 1 + i) == 0) {
        feet.push_back(i);
      }
    }
    // If no foot in swing phase
    if (feet.size() == 0) {
      return 0;
    }

    // For each foot in swing phase get remaining duration of the swing phase
    t0s.clear();
    for (int j = 0; j < (int)feet.size(); j++) {
      int i = feet[j];

      t_swing[i] = get_stance_swing_duration(0, feet[j], 0.0);  // 0.0 for swing phase

      double value = t_swing[i] - (t_remaining * k_mpc - ((k + 1) % k_mpc)) * dt_tsid - dt_tsid;

      if (value > 0.0) {
        t0s.push_back(value);
      } else {
        t0s.push_back(0.0);
      }
    }
  } else {
    // If no foot in swing phase
    if (feet.size() == 0) {
      return 0;
    }

    // Increment of one time step for feet in swing phase
    for (int i = 0; i < (int)feet.size(); i++) {
      double value = t0s[i] + dt_tsid;
      if (value > 0.0) {
        t0s[i] = value;
      } else {
        t0s[i] = 0.0;
      }
    }
  }

  // Get position, velocity and acceleration commands for feet in swing phase
  for (int i = 0; i < (int)feet.size(); i++) {
    int i_foot = feet[i];

    // Get desired 3D position, velocity and acceleration
    if ((t0s[i] == 0.000) || (k == 0)) {
      res_gen.col(i_foot) =
          (myTrajGen[i_foot])
              .get_next_foot(mgoals(0, i_foot), 0.0, 0.0, mgoals(3, i_foot), 0.0, 0.0, footsteps_target(0, i_foot),
                             footsteps_target(1, i_foot), t0s[i], t_swing[i_foot], dt_tsid);

      mgoals.col(i_foot) << res_gen.block(0, i_foot, 6, 1);

    } else {
      res_gen.col(i_foot) =
          (myTrajGen[i_foot])
              .get_next_foot(mgoals(0, i_foot), mgoals(1, i_foot), mgoals(2, i_foot), mgoals(3, i_foot),
                             mgoals(4, i_foot), mgoals(5, i_foot), footsteps_target(0, i_foot),
                             footsteps_target(1, i_foot), t0s[i], t_swing[i_foot], dt_tsid);

      mgoals.col(i_foot) << res_gen.block(0, i_foot, 6, 1);
    }

    // Store desired position, velocity and acceleration for later call to this function
    goals.col(i_foot) << res_gen(0, i_foot), res_gen(3, i_foot), res_gen(6, i_foot);
    vgoals.col(i_foot) << res_gen(1, i_foot), res_gen(4, i_foot), res_gen(7, i_foot);
    agoals.col(i_foot) << res_gen(2, i_foot), res_gen(5, i_foot), res_gen(8, i_foot);
  }

  return 0;
}

int Planner::run_planner(int k, const Eigen::MatrixXd &q, const Eigen::MatrixXd &v, const Eigen::MatrixXd &b_vref_in,
                         double h_estim, double z_average, int joystick_code) {
  /* Run the planner for one iteration of the main control loop

  Args:
    k (int): number of time steps since the start of the simulation
    q (7x1 array): current position vector of the flying base in world frame (linear and angular stacked)
    v (6x1 array): current velocity vector of the flying base in world frame (linear and angular stacked)
    b_vref_in (6x1 array): desired velocity vector of the flying base in base frame (linear and angular stacked)
    h_estim (double): estimated height of the base
    z_average (double): average height of feet currently in stance phase
    joystick_code (int): integer to trigger events with the joystick
  */

  // Get the reference velocity in world frame (given in base frame)
  Eigen::Quaterniond quat(q(6, 0), q(3, 0), q(4, 0), q(5, 0));  // w, x, y, z
  RPY << pinocchio::rpy::matrixToRpy(quat.toRotationMatrix());
  double c = std::cos(RPY(2, 0));
  double s = std::sin(RPY(2, 0));
  R_2.block(0, 0, 2, 2) << c, -s, s, c;
  R_2(2, 2) = 1.0;
  vref_in.block(0, 0, 3, 1) = R_2 * b_vref_in.block(0, 0, 3, 1);
  vref_in.block(3, 0, 3, 1) = b_vref_in.block(3, 0, 3, 1);

  // Handle joystick events
  handle_joystick(joystick_code, q);

  // Move one step further in the gait
  if (k % k_mpc == 0) {
    roll(k);
  }

  // Compute the desired location of footsteps over the prediction horizon
  compute_footsteps(q, v, vref_in);

  // Get the reference trajectory for the MPC
  getRefStates(q, v, vref_in, z_average);

  // Update desired location of footsteps on the ground
  update_target_footsteps();

  // Update trajectory generator (3D pos, vel, acc)
  update_trajectory_generator(k, h_estim);

  return 0;
}

Eigen::MatrixXd Planner::get_xref() { return xref; }
Eigen::MatrixXd Planner::get_fsteps() { return fsteps; }
Eigen::MatrixXd Planner::get_gait() { return gait_f; }
Eigen::MatrixXd Planner::get_goals() { return goals; }
Eigen::MatrixXd Planner::get_vgoals() { return vgoals; }
Eigen::MatrixXd Planner::get_agoals() { return agoals; }

int Planner::roll(int k) {
  /* Move one step further in the gait cycle

  Decrease by 1 the number of remaining step for the current phase of the gait
  Transfer current gait phase into past gait matrix
  Insert future desired gait phase at the end of the gait matrix

  Args:
      k (int): number of WBC iterations since the start of the simulation
  */

  // Transfer current gait into past gait
  // If current gait is the same than the first line of past gait we just increment the counter
  if ((gait_f.block(0, 1, 1, 4)).isApprox(gait_p.block(0, 1, 1, 4))) {
    gait_p(0, 0) += 1.0;
  } else {  // If current gait is not the same than the first line of past gait we have to insert it
    Eigen::Matrix<double, 5, 5> tmp = gait_p.block(0, 0, N0_gait - 1, 5);
    gait_p.block(1, 0, N0_gait - 1, 5) = tmp;
    gait_p.row(0) = gait_f.row(0);
    gait_p(0, 0) = 1.0;
  }

  // Age future gait
  if (gait_f(0, 0) == 1.0) {
    gait_f.block(0, 0, N0_gait - 1, 5) = gait_f.block(1, 0, N0_gait - 1, 5);

    // Entering new contact phase, store positions of feet that are now in contact
    if (k != 0) {
      for (int i = 0; i < 4; i++) {
        if (gait_f(0, 1 + i) == 1.0) {
          o_feet_contact.block(0, 3 * i, 1, 3) = fsteps.block(1, 1 + 3 * i, 1, 3);
        }
      }
    }
  } else {
    gait_f(0, 0) -= 1.0;
  }

  // Get index of first empty line
  int i = 1;
  while (gait_f(i, 0) > 0.0) {
    i++;
  }
  // Increment last gait line or insert a new line
  if ((gait_f.block(i - 1, 1, 1, 4)).isApprox(gait_f_des.block(0, 1, 1, 4))) {
    gait_f(i - 1, 0) += 1.0;
  } else {
    gait_f.row(i) = gait_f_des.row(0);
    gait_f(i, 0) = 1.0;
  }

  // Age future desired gait
  // Get index of first empty line
  int j = 1;
  while (gait_f_des(j, 0) > 0.0) {
    j++;
  }
  // Increment last gait line or insert a new line
  if ((gait_f_des.block(0, 1, 1, 4)).isApprox(gait_f_des.block(j - 1, 1, 1, 4))) {
    gait_f_des(j - 1, 0) += 1.0;
  } else {
    gait_f_des.row(j) = gait_f_des.row(0);
    gait_f_des(j, 0) = 1.0;
  }
  if (gait_f_des(0, 0) == 1.0) {
    gait_f_des.block(0, 0, N0_gait - 1, 5) = gait_f_des.block(1, 0, N0_gait - 1, 5);
  } else {
    gait_f_des(0, 0) -= 1.0;
  }

  return 0;
}

int Planner::handle_joystick(int code, const Eigen::MatrixXd &q) {
  /* Handle the joystick code to trigger events (change of gait for instance)
  
  Args:
    code (int): integer to trigger events with the joystick
    q (7x1 array): current position vector of the flying base in world frame (linear and angular stacked)
  */

  if (code == 0) {
    return 0;
  }
  else if (code == 1) {
    create_pacing();
    is_static = false;
  }
  else if (code == 2) {
    create_bounding();
    is_static = false;
  }
  else if (code == 3) {
    create_trot();
    is_static = false;
  }
  else if (code == 4) {
    create_static();
    q_static.block(0, 0, 7, 1) = q.block(0, 0, 7, 1);
    is_static = true;
  }

  return 0;
}

// Trajectory generator functions (output reference pos, vel and acc of feet in swing phase)

TrajGen::TrajGen() {}

TrajGen::TrajGen(double h_in, double t_lock_in, double x_in, double y_in) {
  h = h_in;
  time_adaptative_disabled = t_lock_in;
  x1 = x_in;
  y1 = y_in;

  for (int i = 0; i < 6; i++) {
    lastCoeffs_x[i] = 0.0;
    lastCoeffs_y[i] = 0.0;
  }
}

Eigen::Matrix<double, 11, 1> TrajGen::get_next_foot(double x0, double dx0, double ddx0, double y0, double dy0,
                                                    double ddy0, double x1_in, double y1_in, double t0, double t1,
                                                    double dt) {
  /* Compute the reference position, velocity and acceleration of a foot in swing phase

  Args:
    x0 (double): current X position of the foot
    dx0 (double): current X velocity of the foot
    ddx0 (double): current X acceleration of the foot
    y0 (double): current Y position of the foot
    dy0 (double): current Y velocity of the foot
    ddy0 (double): current Y acceleration of the foot
    x1 (double): desired target location for X at the end of the swing phase
    y1 (double): desired target location for Y at the end of the swing phase
    t0 (double): time elapsed since the start of the swing phase
    t1 (double): duration of the swing phase
    dt (double): time step of the control
  */

  double epsilon = 0.0;
  double t2 = t1;
  double t3 = t0;
  t1 -= 2 * epsilon;
  t0 -= epsilon;

  if ((t1 - t0) > time_adaptative_disabled) {  // adaptative_mode

    // compute polynoms coefficients for x and y
    Ax5 = (ddx0 * std::pow(t0, 2) - 2 * ddx0 * t0 * t1 - 6 * dx0 * t0 + ddx0 * std::pow(t1, 2) + 6 * dx0 * t1 +
           12 * x0 - 12 * x1_in) /
          (2 * std::pow((t0 - t1), 2) *
           (std::pow(t0, 3) - 3 * std::pow(t0, 2) * t1 + 3 * t0 * std::pow(t1, 2) - std::pow(t1, 3)));
    Ax4 = (30 * t0 * x1_in - 30 * t0 * x0 - 30 * t1 * x0 + 30 * t1 * x1_in - 2 * std::pow(t0, 3) * ddx0 -
           3 * std::pow(t1, 3) * ddx0 + 14 * std::pow(t0, 2) * dx0 - 16 * std::pow(t1, 2) * dx0 + 2 * t0 * t1 * dx0 +
           4 * t0 * std::pow(t1, 2) * ddx0 + std::pow(t0, 2) * t1 * ddx0) /
          (2 * std::pow((t0 - t1), 2) *
           (std::pow(t0, 3) - 3 * std::pow(t0, 2) * t1 + 3 * t0 * std::pow(t1, 2) - std::pow(t1, 3)));
    Ax3 = (std::pow(t0, 4) * ddx0 + 3 * std::pow(t1, 4) * ddx0 - 8 * std::pow(t0, 3) * dx0 +
           12 * std::pow(t1, 3) * dx0 + 20 * std::pow(t0, 2) * x0 - 20 * std::pow(t0, 2) * x1_in +
           20 * std::pow(t1, 2) * x0 - 20 * std::pow(t1, 2) * x1_in + 80 * t0 * t1 * x0 - 80 * t0 * t1 * x1_in +
           4 * std::pow(t0, 3) * t1 * ddx0 + 28 * t0 * std::pow(t1, 2) * dx0 - 32 * std::pow(t0, 2) * t1 * dx0 -
           8 * std::pow(t0, 2) * std::pow(t1, 2) * ddx0) /
          (2 * std::pow((t0 - t1), 2) *
           (std::pow(t0, 3) - 3 * std::pow(t0, 2) * t1 + 3 * t0 * std::pow(t1, 2) - std::pow(t1, 3)));
    Ax2 = -(std::pow(t1, 5) * ddx0 + 4 * t0 * std::pow(t1, 4) * ddx0 + 3 * std::pow(t0, 4) * t1 * ddx0 +
            36 * t0 * std::pow(t1, 3) * dx0 - 24 * std::pow(t0, 3) * t1 * dx0 + 60 * t0 * std::pow(t1, 2) * x0 +
            60 * std::pow(t0, 2) * t1 * x0 - 60 * t0 * std::pow(t1, 2) * x1_in - 60 * std::pow(t0, 2) * t1 * x1_in -
            8 * std::pow(t0, 2) * std::pow(t1, 3) * ddx0 - 12 * std::pow(t0, 2) * std::pow(t1, 2) * dx0) /
          (2 * (std::pow(t0, 2) - 2 * t0 * t1 + std::pow(t1, 2)) *
           (std::pow(t0, 3) - 3 * std::pow(t0, 2) * t1 + 3 * t0 * std::pow(t1, 2) - std::pow(t1, 3)));
    Ax1 = -(2 * std::pow(t1, 5) * dx0 - 2 * t0 * std::pow(t1, 5) * ddx0 - 10 * t0 * std::pow(t1, 4) * dx0 +
            std::pow(t0, 2) * std::pow(t1, 4) * ddx0 + 4 * std::pow(t0, 3) * std::pow(t1, 3) * ddx0 -
            3 * std::pow(t0, 4) * std::pow(t1, 2) * ddx0 - 16 * std::pow(t0, 2) * std::pow(t1, 3) * dx0 +
            24 * std::pow(t0, 3) * std::pow(t1, 2) * dx0 - 60 * std::pow(t0, 2) * std::pow(t1, 2) * x0 +
            60 * std::pow(t0, 2) * std::pow(t1, 2) * x1_in) /
          (2 * std::pow((t0 - t1), 2) *
           (std::pow(t0, 3) - 3 * std::pow(t0, 2) * t1 + 3 * t0 * std::pow(t1, 2) - std::pow(t1, 3)));
    Ax0 = (2 * x1_in * std::pow(t0, 5) - ddx0 * std::pow(t0, 4) * std::pow(t1, 3) - 10 * x1_in * std::pow(t0, 4) * t1 +
           2 * ddx0 * std::pow(t0, 3) * std::pow(t1, 4) + 8 * dx0 * std::pow(t0, 3) * std::pow(t1, 3) +
           20 * x1_in * std::pow(t0, 3) * std::pow(t1, 2) - ddx0 * std::pow(t0, 2) * std::pow(t1, 5) -
           10 * dx0 * std::pow(t0, 2) * std::pow(t1, 4) - 20 * x0 * std::pow(t0, 2) * std::pow(t1, 3) +
           2 * dx0 * t0 * std::pow(t1, 5) + 10 * x0 * t0 * std::pow(t1, 4) - 2 * x0 * std::pow(t1, 5)) /
          (2 * (std::pow(t0, 2) - 2 * t0 * t1 + std::pow(t1, 2)) *
           (std::pow(t0, 3) - 3 * std::pow(t0, 2) * t1 + 3 * t0 * std::pow(t1, 2) - std::pow(t1, 3)));

    Ay5 = (ddy0 * std::pow(t0, 2) - 2 * ddy0 * t0 * t1 - 6 * dy0 * t0 + ddy0 * std::pow(t1, 2) + 6 * dy0 * t1 +
           12 * y0 - 12 * y1_in) /
          (2 * std::pow((t0 - t1), 2) *
           (std::pow(t0, 3) - 3 * std::pow(t0, 2) * t1 + 3 * t0 * std::pow(t1, 2) - std::pow(t1, 3)));
    Ay4 = (30 * t0 * y1_in - 30 * t0 * y0 - 30 * t1 * y0 + 30 * t1 * y1_in - 2 * std::pow(t0, 3) * ddy0 -
           3 * std::pow(t1, 3) * ddy0 + 14 * std::pow(t0, 2) * dy0 - 16 * std::pow(t1, 2) * dy0 + 2 * t0 * t1 * dy0 +
           4 * t0 * std::pow(t1, 2) * ddy0 + std::pow(t0, 2) * t1 * ddy0) /
          (2 * std::pow((t0 - t1), 2) *
           (std::pow(t0, 3) - 3 * std::pow(t0, 2) * t1 + 3 * t0 * std::pow(t1, 2) - std::pow(t1, 3)));
    Ay3 = (std::pow(t0, 4) * ddy0 + 3 * std::pow(t1, 4) * ddy0 - 8 * std::pow(t0, 3) * dy0 +
           12 * std::pow(t1, 3) * dy0 + 20 * std::pow(t0, 2) * y0 - 20 * std::pow(t0, 2) * y1_in +
           20 * std::pow(t1, 2) * y0 - 20 * std::pow(t1, 2) * y1_in + 80 * t0 * t1 * y0 - 80 * t0 * t1 * y1_in +
           4 * std::pow(t0, 3) * t1 * ddy0 + 28 * t0 * std::pow(t1, 2) * dy0 - 32 * std::pow(t0, 2) * t1 * dy0 -
           8 * std::pow(t0, 2) * std::pow(t1, 2) * ddy0) /
          (2 * std::pow((t0 - t1), 2) *
           (std::pow(t0, 3) - 3 * std::pow(t0, 2) * t1 + 3 * t0 * std::pow(t1, 2) - std::pow(t1, 3)));
    Ay2 = -(std::pow(t1, 5) * ddy0 + 4 * t0 * std::pow(t1, 4) * ddy0 + 3 * std::pow(t0, 4) * t1 * ddy0 +
            36 * t0 * std::pow(t1, 3) * dy0 - 24 * std::pow(t0, 3) * t1 * dy0 + 60 * t0 * std::pow(t1, 2) * y0 +
            60 * std::pow(t0, 2) * t1 * y0 - 60 * t0 * std::pow(t1, 2) * y1_in - 60 * std::pow(t0, 2) * t1 * y1_in -
            8 * std::pow(t0, 2) * std::pow(t1, 3) * ddy0 - 12 * std::pow(t0, 2) * std::pow(t1, 2) * dy0) /
          (2 * (std::pow(t0, 2) - 2 * t0 * t1 + std::pow(t1, 2)) *
           (std::pow(t0, 3) - 3 * std::pow(t0, 2) * t1 + 3 * t0 * std::pow(t1, 2) - std::pow(t1, 3)));
    Ay1 = -(2 * std::pow(t1, 5) * dy0 - 2 * t0 * std::pow(t1, 5) * ddy0 - 10 * t0 * std::pow(t1, 4) * dy0 +
            std::pow(t0, 2) * std::pow(t1, 4) * ddy0 + 4 * std::pow(t0, 3) * std::pow(t1, 3) * ddy0 -
            3 * std::pow(t0, 4) * std::pow(t1, 2) * ddy0 - 16 * std::pow(t0, 2) * std::pow(t1, 3) * dy0 +
            24 * std::pow(t0, 3) * std::pow(t1, 2) * dy0 - 60 * std::pow(t0, 2) * std::pow(t1, 2) * y0 +
            60 * std::pow(t0, 2) * std::pow(t1, 2) * y1_in) /
          (2 * std::pow((t0 - t1), 2) *
           (std::pow(t0, 3) - 3 * std::pow(t0, 2) * t1 + 3 * t0 * std::pow(t1, 2) - std::pow(t1, 3)));
    Ay0 = (2 * y1_in * std::pow(t0, 5) - ddy0 * std::pow(t0, 4) * std::pow(t1, 3) - 10 * y1_in * std::pow(t0, 4) * t1 +
           2 * ddy0 * std::pow(t0, 3) * std::pow(t1, 4) + 8 * dy0 * std::pow(t0, 3) * std::pow(t1, 3) +
           20 * y1_in * std::pow(t0, 3) * std::pow(t1, 2) - ddy0 * std::pow(t0, 2) * std::pow(t1, 5) -
           10 * dy0 * std::pow(t0, 2) * std::pow(t1, 4) - 20 * y0 * std::pow(t0, 2) * std::pow(t1, 3) +
           2 * dy0 * t0 * std::pow(t1, 5) + 10 * y0 * t0 * std::pow(t1, 4) - 2 * y0 * std::pow(t1, 5)) /
          (2 * (std::pow(t0, 2) - 2 * t0 * t1 + std::pow(t1, 2)) *
           (std::pow(t0, 3) - 3 * std::pow(t0, 2) * t1 + 3 * t0 * std::pow(t1, 2) - std::pow(t1, 3)));

    // Save coeffs
    lastCoeffs_x[0] = Ax5;
    lastCoeffs_x[1] = Ax4;
    lastCoeffs_x[2] = Ax3;
    lastCoeffs_x[3] = Ax2;
    lastCoeffs_x[4] = Ax1;
    lastCoeffs_x[5] = Ax0;
    lastCoeffs_y[0] = Ay5;
    lastCoeffs_y[1] = Ay4;
    lastCoeffs_y[2] = Ay3;
    lastCoeffs_y[3] = Ay2;
    lastCoeffs_y[4] = Ay1;
    lastCoeffs_y[5] = Ay0;
    x1 = x1_in;
    y1 = y1_in;
  } else {
    // Use last coefficients
    Ax5 = lastCoeffs_x[0];
    Ax4 = lastCoeffs_x[1];
    Ax3 = lastCoeffs_x[2];
    Ax2 = lastCoeffs_x[3];
    Ax1 = lastCoeffs_x[4];
    Ax0 = lastCoeffs_x[5];
    Ay5 = lastCoeffs_y[0];
    Ay4 = lastCoeffs_y[1];
    Ay3 = lastCoeffs_y[2];
    Ay2 = lastCoeffs_y[3];
    Ay1 = lastCoeffs_y[4];
    Ay0 = lastCoeffs_y[5];
  }

  // Coefficients for z (deterministic)
  Az6 = -h / (std::pow((t2 / 2), 3) * std::pow((t2 - t2 / 2), 3));
  Az5 = (3 * t2 * h) / (std::pow((t2 / 2), 3) * std::pow((t2 - t2 / 2), 3));
  Az4 = -(3 * std::pow(t2, 2) * h) / (std::pow((t2 / 2), 3) * std::pow((t2 - t2 / 2), 3));
  Az3 = (std::pow(t2, 3) * h) / (std::pow((t2 / 2), 3) * std::pow((t2 - t2 / 2), 3));

  // Get the next point
  double ev = t0 + dt;
  double evz = t3 + dt;

  result(6, 0) =
      Az3 * std::pow(evz, 3) + Az4 * std::pow(evz, 4) + Az5 * std::pow(evz, 5) + Az6 * std::pow(evz, 6);  // pos Z
  result(7, 0) = 3 * Az3 * std::pow(evz, 2) + 4 * Az4 * std::pow(evz, 3) + 5 * Az5 * std::pow(evz, 4) +
                 6 * Az6 * std::pow(evz, 5);  // vel Z
  result(8, 0) = 2 * 3 * Az3 * evz + 3 * 4 * Az4 * std::pow(evz, 2) + 4 * 5 * Az5 * std::pow(evz, 3) +
                 5 * 6 * Az6 * std::pow(evz, 4);  // acc Z
  result(9, 0) = x1;                              // current goal x
  result(10, 0) = y1;                             // current goal y

  if ((t3 < epsilon) || (t3 > (t2 - epsilon))) {  // Just vertical motion
    result(0, 0) = x0;
    result(1, 0) = 0.0;
    result(2, 0) = 0.0;
    result(3, 0) = y0;
    result(4, 0) = 0.0;
    result(5, 0) = 0.0;
  } else {
    // pos, vel, acc X
    result(0, 0) =
        Ax0 + Ax1 * ev + Ax2 * std::pow(ev, 2) + Ax3 * std::pow(ev, 3) + Ax4 * std::pow(ev, 4) + Ax5 * std::pow(ev, 5);
    result(1, 0) =
        Ax1 + 2 * Ax2 * ev + 3 * Ax3 * std::pow(ev, 2) + 4 * Ax4 * std::pow(ev, 3) + 5 * Ax5 * std::pow(ev, 4);
    result(2, 0) = 2 * Ax2 + 3 * 2 * Ax3 * ev + 4 * 3 * Ax4 * std::pow(ev, 2) + 5 * 4 * Ax5 * std::pow(ev, 3);

    // pos, vel, acc Y
    result(3, 0) =
        Ay0 + Ay1 * ev + Ay2 * std::pow(ev, 2) + Ay3 * std::pow(ev, 3) + Ay4 * std::pow(ev, 4) + Ay5 * std::pow(ev, 5);
    result(4, 0) =
        Ay1 + 2 * Ay2 * ev + 3 * Ay3 * std::pow(ev, 2) + 4 * Ay4 * std::pow(ev, 3) + 5 * Ay5 * std::pow(ev, 4);
    result(5, 0) = 2 * Ay2 + 3 * 2 * Ay3 * ev + 4 * 3 * Ay4 * std::pow(ev, 2) + 5 * 4 * Ay5 * std::pow(ev, 3);
  }

  return result;
}
