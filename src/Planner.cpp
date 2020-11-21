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

  xref = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(12, 1 + n_steps);

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

    // std::cout << gait << std::endl;
    // std::cout << next_footstep << std::endl;
    std::cout << fsteps.block(0, 0, 5, 13) << std::endl;
    std::cout << "xref:" << std::endl;
    std::cout << xref.block(0, 0, 12, 5) << std::endl;


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

int Planner::compute_footsteps(Eigen::MatrixXd q_cur, Eigen::MatrixXd v_cur, Eigen::MatrixXd v_ref) {
    
    fsteps = Eigen::Matrix<double, N0_gait, 13>::Zero();
    fsteps.col(0) = gait.col(0);

    // Set current position of feet for feet in stance phase
    for (int j=0; j<4; j++) {
        if (gait(0, 1+j) == 1.0) {
            fsteps.block(0, 1+3*j, 1, 3) = o_feet_contact.block(0, 3*j, 1, 3);
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

    // Get current and reference velocities in base frame (rotated yaw)
    // Velocity along Z component is set to 0.0 with R_1
    double c = std::cos(RPY(2, 0));
    double s = std::sin(RPY(2, 0));
    R_1.block(0, 0, 2, 2) << c, s, -s, c; // already transposed here
    b_v_cur = R_1 * v_cur.block(0, 0, 3, 1);
    b_v_ref = R_1 * v_ref.block(0, 0, 3, 1);

    // Update the footstep matrix depending on the different phases of the gait (swing & stance)
    int i = 1;
    while (gait(i, 0) != 0) {

        // Feet that were in stance phase and are still in stance phase do not move
        for (int j=0; j<4; j++) {
            if (gait(i-1, 1+j) * gait(i, 1+j) > 0) {
                fsteps.block(i, 1+3*j, 1, 3) = fsteps.block(i-1, 1+3*j, 1, 3);
            }
        }

        // Current position without height
        q_tmp << q_cur(0, 0), q_cur(1, 0), 0.0;

        // Feet that were in swing phase and are now in stance phase need to be updated
        for (int j=0; j<4; j++) {
            if ((1 - gait(i-1, 1+j)) * gait(i, 1+j) > 0) {

                // Offset to the future position
                q_dxdy << dx(i-1, 0), dy(i-1, 0), 0.0;

                // Get future desired position of footsteps
                compute_next_footstep(j);

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

int Planner::compute_next_footstep(int j) {

    double t_stance = 0.5 * T_gait;

    // Order of feet: FL, FR, HL, HR

    // self.next_footstep = np.zeros((3, 4))

    // Add symmetry term
    next_footstep.col(j) = t_stance * 0.5 * b_v_cur;

    // Add feedback term
    next_footstep.col(j) += k_feedback * (b_v_cur - b_v_ref);

    // Add centrifugal term
    cross << b_v_cur(1, 0) * b_v_ref(2, 0) - b_v_cur(2, 0) * b_v_ref(1, 0),
                b_v_cur(2, 0) * b_v_ref(0, 0) - b_v_cur(0, 0) * b_v_ref(2, 0),
                0.0; 
    next_footstep.col(j) += 0.5 * std::sqrt(h_ref/g) * cross;

    // Legs have a limited length so the deviation has to be limited
    if (next_footstep(0, j) > L) {next_footstep(0, j) = L;}
    else if (next_footstep(0, j) < -L) {next_footstep(0, j) = -L;}
    
    if (next_footstep(1, j) > L) {next_footstep(1, j) = L;}
    else if (next_footstep(1, j) < -L) {next_footstep(1, j) = -L;}

    // Add shoulders
    next_footstep.col(j) += shoulders.col(j);

    return 0;
}

int Planner::getRefStates(Eigen::MatrixXd q, Eigen::MatrixXd v, Eigen::MatrixXd vref, double z_average) {

    std::cout << q.transpose() << std::endl;
    std::cout << v.transpose() << std::endl;
    std::cout << vref.transpose() << std::endl;

    // Update yaw and yaw velocity
    xref.block(5, 1, 1, n_steps) = vref(5, 0) * dt_vector;
    for (int i=0; i<n_steps; i++) {
        xref(11, 1+i) = vref(5, 0);
    }

    // Update x and y velocities taking into account the rotation of the base over the prediction horizon
    for (int i=0; i<n_steps; i++) {
       xref(6, 1+i) = vref(0, 0) * std::cos(xref(5, 1+i)) - vref(1, 0) * std::sin(xref(5, 1+i));
       xref(7, 1+i) = vref(0, 0) * std::sin(xref(5, 1+i)) + vref(1, 0) * std::cos(xref(5, 1+i));
    }
 
    // Update x and y depending on x and y velocities (cumulative sum)
    if (vref(5, 0) != 0) {
        for (int i=0; i<n_steps; i++) {
            xref(0, 1+i) = (vref(0, 0) * std::sin(vref(5, 0) * dt_vector(0, i))
                            + vref(1, 0) * (std::cos(vref(5, 0) * dt_vector(0, i)) - 1.0)) / vref(5, 0);
            xref(1, 1+i) = (vref(1, 0) * std::sin(vref(5, 0) * dt_vector(0, i))
                            - vref(0, 0) * (std::cos(vref(5, 0) * dt_vector(0, i)) - 1.0)) / vref(5, 0);
        }
    }
    else {
        for (int i=0; i<n_steps; i++) {
            xref(0, 1+i) = vref(0, 0) * dt_vector(0, i);
            xref(1, 1+i) = vref(1, 0) * dt_vector(0, i);
        }
    }

    for (int i=0; i<n_steps; i++) {
        xref(5, 1+i) += RPY(2, 0);
        xref(2, 1+i) = h_ref + z_average;
        xref(8, 1+i) = 0.0;
    }
    
    // No need to update Z velocity as the reference is always 0
    // No need to update roll and roll velocity as the reference is always 0 for those
    // No need to update pitch and pitch velocity as the reference is always 0 for those

    // Update the current state
    xref.block(0, 0, 3, 1) = q.block(0, 0, 3, 1);
    xref.block(3, 0, 3, 1) = RPY;
    xref.block(6, 0, 3, 1) = v.block(0, 0, 3, 1);
    xref.block(9, 0, 3, 1) = v.block(3, 0, 3, 1);

    for (int i=0; i<n_steps; i++) {
        xref(0, 1+i) += xref(0, 0);
        xref(1, 1+i) += xref(1, 0);
    }

    if (is_static) {
        for (int i=0; i<n_steps; i++) {
            xref.block(0, 1+i, 3, 1) = q_static.block(0, 0, 3, 1);
        }
        xref.block(3, 1, 3, n_steps) = Eigen::Matrix<double, 3, Eigen::Dynamic>::Zero(3, n_steps); // (utils_mpc.quaternionToRPY(self.q_static[3:7, 0])).reshape((3, 1))
    }

    return 0;
}

int Planner::update_target_footsteps() {

    for (int i=0; i<4; i++) {
        // Index of the first non-empty line
        int index = 0;
        while (fsteps(index, 1+3*i) == 0.0) {index++;}

        // Copy fsteps
        footsteps_target.col(i) = fsteps.block(index, 1+3*i, 1, 2).transpose();
    }

    return 0;
}

int Planner::run_planner(int k, const Eigen::MatrixXd &q, const Eigen::MatrixXd &v,
                         const Eigen::MatrixXd &b_vref_in, double h_estim, double z_average) {

    // Get the reference velocity in world frame (given in base frame)
    double c = std::cos(RPY(2, 0));
    double s = std::sin(RPY(2, 0));
    R_2.block(0, 0, 2, 2) << c, -s, s, c;
    R_2(2, 2) = 1.0;
    vref_in.block(0, 0, 3, 1) = R_2 * b_vref_in.block(0, 0, 3, 1);
    vref_in.block(3, 0, 3, 1) = b_vref_in.block(3, 0, 3, 1);

    // Move one step further in the gait
    if (k % k_mpc == 0) {roll(k);}

    // Compute the desired location of footsteps over the prediction horizon
    compute_footsteps(q, v, vref_in);

    // Get the reference trajectory for the MPC
    getRefStates(q, v, vref_in, z_average);

    // Update desired location of footsteps on the ground
    update_target_footsteps();

    // Update trajectory generator (3D pos, vel, acc)
    // update_trajectory_generator(k, h_estim, q);

    return 0;
}

Eigen::MatrixXd Planner::get_xref() {return xref;}
Eigen::MatrixXd Planner::get_fsteps() {return fsteps;}