#include "example-adder/Planner.hpp"

Planner::Planner(double dt_in, double dt_tsid_in, int n_periods_in, double T_gait_in,
                 int k_mpc_in, bool on_solo8_in, double h_ref_in, const Eigen::MatrixXd & fsteps_in) {
  
  dt = dt_in;
  dt_tsid = dt_tsid_in;
  n_periods = n_periods_in;
  T_gait = T_gait_in;
  k_mpc = k_mpc_in;
  on_solo8 = on_solo8_in;
  h_ref = h_ref_in;

  shoulders << 0.1946, 0.1946, -0.1946, -0.1946,
               0.14695, -0.14695, 0.14695, -0.14695,
               0.0, 0.0, 0.0, 0.0;

  Eigen::Map<Eigen::Matrix<double, 1, 12>> v1(shoulders.data(), shoulders.size());
  o_feet_contact << v1;

  footsteps_target.block(0, 0, 2, 4) = shoulders.block(0, 0, 2, 4);

  n_steps = n_periods * (int)std::lround(T_gait/dt);
  dt_vector = Eigen::VectorXd::LinSpaced(n_steps, dt, T_gait).transpose();
  R(2, 2) = 1.0;
  
  // Create gait matrix
  create_trot();
  desired_gait = gait;
  new_desired_gait = gait;

  // For foot trajectory generator
  goals << fsteps_in.block(0, 0, 3, 4);
  mgoals.col(0) << fsteps_in.block(0, 0, 1, 4);
  mgoals.col(3) << fsteps_in.block(1, 0, 1, 4);

}

Planner::Planner() { }

void Planner::Print() {

    std::cout << gait << std::endl;

}

int Planner::create_trot() {

    // Number of timesteps in a half period of gait
    int N = (int)std::lround(0.5 * T_gait/dt);

    // Starting status of the gait
    // 4-stance phase, 2-stance phase, 4-stance phase, 2-stance phase
    gait = Eigen::Matrix<double, N0_gait, 5>::Zero();
    gait.block(0, 0, 2, 1) << N, N;
    fsteps.block(0, 0, 2, 1) = gait.block(0, 0, 2, 1);

    // Set stance and swing phases
    // Coefficient (i, j) is equal to 0.0 if the j-th feet is in swing phase during the i-th phase
    // Coefficient (i, j) is equal to 1.0 if the j-th feet is in stance phase during the i-th phase
    gait(0, 1) = 1.0;
    gait(0, 4) = 1.0;
    gait(1, 2) = 1.0;
    gait(1, 3) = 1.0;

    return 0;
}

int Planner::roll(int k) {

    // Retrieve the new desired gait pattern. Most of the time new_desired_gait will be equal to desired_gait since
    // we want to keep the same gait pattern. However if we want to change then the new gait pattern is temporarily
    // stored inside new_desired_gait before being stored inside desired_gait
    if (k % ((int)std::lround(T_gait/dt) * k_mpc) == 0) {
        desired_gait = new_desired_gait;
        pt_line = 0;
        pt_sum = desired_gait(0, 0);
    }

    // Index of the first empty line
    int index = 0;
    while (gait(index, 0) != 0) {index++;}

    // Create a new phase if needed or increase the last one by 1 step
    int pt = (k / k_mpc) % (int)std::lround(T_gait/dt);
    while (pt >= pt_sum) {
        pt_line += 1;
        pt_sum += desired_gait(pt_line, 0);
    }
    if ((desired_gait.block(pt_line, 1, 1, 4)).isApprox(gait.block(index-1, 1, 1, 4))) {
        gait(index-1, 0) += 1.0;
    }
    else {
        gait.block(index, 1, 1, 4) = desired_gait.block(pt_line, 1, 1, 4);
        gait(index, 0) = 1.0;
    }

    // Decrease the current phase by 1 step and delete it if it has ended
    if (gait(0, 0) > 1.0) {
        gait(0, 0) -= 1.0;
    }
    else {
        gait.block(0, 0, N0_gait-1, 5) = gait.block(1, 0, N0_gait-1, 5);
        // Last line should be empty, no need to set it to 0s

        // Store positions of feet that are now in contact
        if (k != 0) {
            for (int i=0; i<4; i++) {
                if (gait(0, 1+i) == 1.0) {
                    o_feet_contact.block(0, 3*i, 1, 3) = fsteps.block(1, 1+3*i, 1, 3);
                }
            }
        }
    }
        
    return 0;
}

int Planner::compute_footsteps(Eigen::Matrix<double, 7, 1> q_cur, Eigen::Matrix<double, 6, 1> v_cur,
                               Eigen::Matrix<double, 7, 1> v_ref) {
    
    fsteps = Eigen::Matrix<double, N0_gait, 13>::Zero();
    fsteps.col(0) = gait.col(0);

    // Set current position of feet for feet in stance phase
    for (int j=0; j<4; j++) {
        if (gait(0, 1+j) == 1.0) {
            fsteps.block(1, 1+3*j, 1, 3) = o_feet_contact.block(0, 3*j, 1, 3);
        }
    }

    // Cumulative time by adding the terms in the first column (remaining number of timesteps)
    // Get future yaw angle compared to current position
    dt_cum(0, 0) = gait(0, 0);
    angle(0, 0) = v_ref(5, 0) * dt_cum(0, 0) + RPY(2, 0);
    for (int j=1; j<N0_gait; j++) {
        dt_cum(j, 0) = dt_cum(j-1, 0) + gait(j, 0);
        angle(j, 0) = v_ref(5, 0) * dt_cum(j, 0) + RPY(2, 0);
    }

    // Displacement following the reference velocity compared to current position
    if (v_ref(5, 0) != 0) {
        for (int j=0; j<N0_gait; j++) {
            dx(j, 0) = (v_cur(0, 0) * std::sin(v_ref(5, 0) * dt_cum(j, 0))
                       + v_cur(1, 0) * (std::cos(v_ref(5, 0) * dt_cum(j, 0)) - 1.0)) / v_ref(5, 0);
            dy(j, 0) = (v_cur(1, 0) * std::sin(v_ref(5, 0) * dt_cum(j, 0))
                       - v_cur(0, 0) * (std::cos(v_ref(5, 0) * dt_cum(j, 0)) - 1.0)) / v_ref(5, 0);
        }
    }
    else {
        for (int j=0; j<N0_gait; j++) {
            dx(j, 0) = v_cur(0, 0) * dt_cum(j, 0);
            dy(j, 0) = v_cur(1, 0) * dt_cum(j, 0);
        }
    }

    // Update the footstep matrix depending on the different phases of the gait (swing & stance)
    int i = 1;
    while (gait(i, 0) != 0) {

        // Feet that were in stance phase and are still in stance phase do not move
        for (int j=0; j<4; j++) {
            if (gait(i-1, j) * gait(i, j) > 0) {
                fsteps.block(i, 1+3*j, 1, 3) = fsteps.block(i-1, 1+3*j, 1, 3);
            }
        }

        // Current position without height
        q_tmp << q_cur(0, 0), q_cur(1, 0), 0.0;

        // Feet that were in swing phase and are now in stance phase need to be updated
        for (int j=0; j<4; j++) {
            if ((1 - gait(i-1, j)) * gait(i, j) > 0) {

                // Offset to the future position
                q_dxdy << dx(i-1, 0), dy(i-1, 0), 0.0;

                // Get future desired position of footsteps
                // TODO: compute_next_footstep(i, q_cur, v_cur, v_ref);

                // Get desired position of footstep compared to current position
                double c = std::cos(angle(i-1, 0));
                double s = std::sin(angle(i-1, 0));
                R.block(0, 0, 2, 2) << c, -s, s, c;
                
                fsteps.block(i, 1+3*j, 1, 3) = (R * next_footstep.col(j) + q_tmp + q_dxdy).transpose();
            }
        }

        i++;
    }

    return 0;
}