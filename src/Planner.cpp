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
  R_1(2, 2) = 1.0;

  xref = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(12, 1 + n_steps);

  // Create gait matrix
  create_trot();
  desired_gait = gait;
  new_desired_gait = gait;

  // For foot trajectory generator
  goals << fsteps_in.block(0, 0, 3, 4);
  mgoals.row(0) << fsteps_in.block(0, 0, 1, 4);
  mgoals.row(3) << fsteps_in.block(1, 0, 1, 4);

  constraints.init_vel = curves::point3_t(0.0, 0.0, 0.0);
  constraints.init_acc = curves::point3_t(0.0, 0.0, 0.0);
  constraints.end_vel = curves::point3_t(0.0, 0.0, 0.0);
  constraints.end_acc = curves::point3_t(0.0, 0.0, 0.0);

  std::vector<curves::point3_t> params;
  params.push_back(curves::point3_t(0.0, 0.0, 0.0));
  params.push_back(curves::point3_t(0.0, 0.0, 0.0));
  params.push_back(curves::point3_t(0.0, 0.0, 0.0));
  for (int i=0; i<4; i++) {
    pr_feet.push_back(params);
    T_min.push_back(curves::bezier_t::num_t(0.0));
    T_max.push_back(curves::bezier_t::num_t(0.5 * T_gait));
    c_feet.push_back(curves::bezier_t(pr_feet[i].begin(), pr_feet[i].end(), constraints, T_min[i], T_max[i]));
  }

  for (int i=0; i<4; i++) {
    myTrajGen.push_back(TrajGen(max_height_feet, t_lock_before_touchdown, shoulders(0, i), shoulders(1, i)));
  }
 

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
    dt_cum(0, 0) = gait(0, 0) * dt;
    angle(0, 0) = v_ref(5, 0) * dt_cum(0, 0) + RPY(2, 0);
    for (int j=1; j<N0_gait; j++) {
        dt_cum(j, 0) = dt_cum(j-1, 0) + gait(j, 0) * dt;
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

    /*std::cout << "vcur: " << v_cur(0, 0) << " " << v_cur(1, 0) << std::endl;
    std::cout << "vref: " << v_ref(5, 0) << std::endl;
    std::cout << "dt_cum: " << dt_cum << std::endl;*/

    // Get current and reference velocities in base frame (rotated yaw)
    double c = std::cos(RPY(2, 0));
    double s = std::sin(RPY(2, 0));
    R_1.block(0, 0, 2, 2) << c, s, -s, c; // already transposed here
    b_v_cur = R_1 * v_cur.block(0, 0, 3, 1);
    b_v_ref.block(0, 0, 3, 1) = R_1 * v_ref.block(0, 0, 3, 1);
    b_v_ref.block(3, 0, 3, 1) = R_1 * v_ref.block(3, 0, 3, 1);

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

                /*std::cout << "#### " << i << " " << j << std::endl;
                std::cout << R << std::endl;
                std::cout << next_footstep.col(j) << std::endl;
                std::cout << q_tmp << std::endl;
                std::cout << q_dxdy << std::endl;*/
                
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

    //std::cout << "Computing " << j << std::endl;

    // Add symmetry term
    next_footstep.col(j) = t_stance * 0.5 * b_v_cur;

    //std::cout << next_footstep.col(j).transpose() << std::endl;

    // Add feedback term
    next_footstep.col(j) += k_feedback * (b_v_cur - b_v_ref.block(0, 0, 3, 1));

    /*std::cout << next_footstep.col(j).transpose() << std::endl;

    std::cout << "b_v_cur: " << b_v_cur(0,0) << b_v_cur(1,0) << b_v_cur(2,0) << std::endl;
    std::cout << "b_v_ref: " << b_v_ref(3,0) << b_v_ref(4,0) << b_v_ref(5,0) << std::endl;
    std::cout << h_ref << " " << g << std::endl;*/

    // Add centrifugal term
    cross << b_v_cur(1, 0) * b_v_ref(5, 0) - b_v_cur(2, 0) * b_v_ref(4, 0),
                b_v_cur(2, 0) * b_v_ref(3, 0) - b_v_cur(0, 0) * b_v_ref(5, 0),
                0.0; 
    next_footstep.col(j) += 0.5 * std::sqrt(h_ref/g) * cross;

    //std::cout << next_footstep.col(j).transpose() << std::endl;

    // Legs have a limited length so the deviation has to be limited
    if (next_footstep(0, j) > L) {next_footstep(0, j) = L;}
    else if (next_footstep(0, j) < -L) {next_footstep(0, j) = -L;}
    
    if (next_footstep(1, j) > L) {next_footstep(1, j) = L;}
    else if (next_footstep(1, j) < -L) {next_footstep(1, j) = -L;}

    // Add shoulders
    next_footstep.col(j) += shoulders.col(j);

    // Remove Z component (working on flat ground)
    next_footstep.row(2) = Eigen::Matrix<double, 1, 4>::Zero();

    //std::cout << next_footstep.col(j).transpose() << std::endl;

    return 0;
}

int Planner::getRefStates(Eigen::MatrixXd q, Eigen::MatrixXd v, Eigen::MatrixXd vref, double z_average) {

    /*std::cout << q.transpose() << std::endl;
    std::cout << v.transpose() << std::endl;
    std::cout << vref.transpose() << std::endl;*/

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

int Planner::update_trajectory_generator(int k, double h_estim) {

    int looping = n_periods * (int)std::lround(T_gait/dt_tsid);
    int k_loop = k % looping;

    if ((k_loop % k_mpc) == 0) {

        // Indexes of feet in swing phase
        feet.clear();
        for (int i=0; i<4; i++) {
            if (gait(0, 1+i) == 0) {feet.push_back(i);}
        }
        // If no foot in swing phase
        if (feet.size() == 0) {
            return 0;
        }

        // i_end_gait should point to the latest non-zero line
        if (gait(i_end_gait, 0) == 0) {
            i_end_gait -= 1;
            while (gait(i_end_gait, 0) == 0) {
                i_end_gait -= 1;
            }
        }
        else {
            while (gait(i_end_gait+1, 0) != 0) {
                i_end_gait += 1;
            }
        }

        // For each foot in swing phase get remaining duration of the swing phase
        t0s.clear();
        for (int j=0; j<feet.size(); j++) {
            int i = feet[j];

            // Compute total duration of current swing phase
            int i_iter = 1;
            t_swing[i] = gait(0, 0);
            while (gait(i_iter, 1+i) == 0) {
                t_swing[i] += gait(i_iter, 0);
                i_iter++;
            }

            double remaining_iterations = t_swing[i] * k_mpc - ((k_loop+1) % k_mpc);

            i_iter = i_end_gait;
            while (gait(i_iter, 1+i) == 0) {
                t_swing[i] += gait(i_iter, 0);
                i_iter--;
            }
            t_swing[i] *= dt_tsid * k_mpc;

            // TODO: Fix that
            t_swing[i] = 0.5 * T_gait;

            double value = t_swing[i] - remaining_iterations * dt_tsid - dt_tsid;
            if (value > 0.0) {t0s.push_back(value);}
            else {t0s.push_back(0.0);}
        }
    }
    else {
        // If no foot in swing phase
        if (feet.size() == 0) {
            return 0;
        }

        for (int i=0; i<feet.size(); i++) {
            double value = t0s[i] + dt_tsid;
            if (value > 0.0) {t0s[i] = value;}
            else {t0s[i] = 0.0;}
        }
    }

    // Get position, velocity and acceleration commands for feet in swing phase
    for (int i=0; i<feet.size(); i++) {
        int i_foot = feet[i];

        
        // c_feet[i_foot] = curves::bezier_t(pr_feet[i].begin(), pr_feet[i].end(), constraints, T_min[i], T_max[i]));

        // Get desired 3D position, velocity and acceleration
        if ((t0s[i] == 0.000) || (k == 0)) {
            /*std::cout << "PASS 1 ";
            std::cout << mgoals(0, i_foot) << " " << mgoals(3, i_foot) << std::endl;
            std::cout << footsteps_target(0, i_foot) << " " << footsteps_target(1, i_foot) << std::endl;*/
            res_gen.col(i_foot) = (myTrajGen[i_foot]).get_next_foot(
                mgoals(0, i_foot), 0.0, 0.0,
                mgoals(3, i_foot), 0.0, 0.0,
                footsteps_target(0, i_foot), footsteps_target(1, i_foot), t0s[i], t_swing[i_foot], dt_tsid);

            mgoals.col(i_foot) << res_gen.block(0, i_foot, 6, 1);

            /*pr_feet[i_foot][0] = curves::point3_t(mgoals(0, i_foot), mgoals(3, i_foot), 0.0);
            pr_feet[i_foot][2] = curves::point3_t(footsteps_target(0, i_foot), footsteps_target(1, i_foot), 0.0);
            pr_feet[i_foot][1] = curves::point3_t(((pr_feet[i_foot][0])(0) + (pr_feet[i_foot][2])(0)) * 0.5,
                                                  ((pr_feet[i_foot][0])(1) + (pr_feet[i_foot][2])(1)) * 0.5,
                                                   max_height_feet);
            T_min[i_foot] = k * dt_tsid;
            T_max[i_foot] = T_min[i_foot] + t_swing[i_foot];*/
        }
        else {
            //std::cout << "PASS 2 ";

            res_gen.col(i_foot) = (myTrajGen[i_foot]).get_next_foot(
                mgoals(0, i_foot), mgoals(1, i_foot), mgoals(2, i_foot),
                mgoals(3, i_foot), mgoals(4, i_foot), mgoals(5, i_foot),
                footsteps_target(0, i_foot), footsteps_target(1, i_foot), t0s[i], t_swing[i_foot], dt_tsid);

            mgoals.col(i_foot) << res_gen.block(0, i_foot, 6, 1);

            /*T_min[i_foot] = k * dt_tsid;
            pr_feet[i_foot][0] = curves::point3_t(mgoals(0, i_foot), mgoals(3, i_foot), 0.0);
            pr_feet[i_foot][2] = curves::point3_t(footsteps_target(0, i_foot), footsteps_target(1, i_foot), 0.0);
            pr_feet[i_foot][1] = curves::point3_t(((pr_feet[i_foot][0])(0) + (pr_feet[i_foot][2])(0)) * 0.5,
                                                  ((pr_feet[i_foot][0])(1) + (pr_feet[i_foot][2])(1)) * 0.5,
                                                   max_height_feet);
            T_max[i_foot] = T_min[i_foot] + t_swing[i_foot];*/
        }

        /*std::cout << "---- " << std::endl;
        std::cout << "Processing feet " << i_foot << std::endl;
        std::cout << "Tmin/max: " <<  T_min[i_foot] << " / " << T_max[i_foot] << std::endl;
        std::cout << "t_swing: " << t_swing[0] << " / " << t_swing[1] << " / " << t_swing[2] << " / " << t_swing[3] << std::endl;
        std::cout << "start: " << (pr_feet[i_foot][0])(0) << " / " << (pr_feet[i_foot][0])(1) << " / "<< (pr_feet[i_foot][0])(2) << std::endl;
        std::cout << "end  : " << (pr_feet[i_foot][2])(0) << " / " << (pr_feet[i_foot][2])(1) << " / "<< (pr_feet[i_foot][2])(2) << std::endl;
        std::cout << "mgoals:" << mgoals(0, i_foot) << " / " << mgoals(3, i_foot) << std::endl;
        std::cout << "target:" << footsteps_target(0, i_foot) << " / " << footsteps_target(1, i_foot) << std::endl;


        c_feet[i_foot] = curves::bezier_t(pr_feet[i_foot].begin(), pr_feet[i_foot].end(), constraints, T_min[i_foot], T_max[i_foot]);
        // Store desired position, velocity and acceleration for later call to this function
        goals.col(i_foot) << (c_feet[i_foot])((k+1) * dt_tsid - T_min[i_foot]);
        vgoals.col(i_foot) << (c_feet[i_foot]).derivate((k+1) * dt_tsid - T_min[i_foot], 1);
        agoals.col(i_foot) << (c_feet[i_foot]).derivate((k+1) * dt_tsid - T_min[i_foot], 2);
        mgoals.col(i_foot) << goals(0, i_foot), vgoals(0, i_foot), agoals(0, i_foot), goals(1, i_foot), vgoals(1, i_foot), agoals(1, i_foot);*/
    
        /*std::cout << "---- " << std::endl;
        std::cout << "Processing feet " << i_foot << std::endl; 
        std::cout << "t0 / tswing " << t0s[i] << " / " << t_swing[i_foot] << std::endl;
        std::cout << "res_gen:";
        for (int o=0; o<11; o++) {
            std::cout << res_gen(o, i_foot) << " / ";
        }
        std::cout << std::endl;
        std::cout << "target:" << footsteps_target(0, i_foot) << " / " << footsteps_target(1, i_foot) << std::endl;
        */
       
        // Store desired position, velocity and acceleration for later call to this function
        goals.col(i_foot) << res_gen(0, i_foot), res_gen(3, i_foot), res_gen(6, i_foot);
        vgoals.col(i_foot) << res_gen(1, i_foot), res_gen(4, i_foot), res_gen(7, i_foot);
        agoals.col(i_foot) << res_gen(2, i_foot), res_gen(5, i_foot), res_gen(8, i_foot);
    
    }
    
    return 0;
}

int Planner::run_planner(int k, const Eigen::MatrixXd &q, const Eigen::MatrixXd &v,
                         const Eigen::MatrixXd &b_vref_in, double h_estim, double z_average) {

    // Get the reference velocity in world frame (given in base frame)
    Eigen::Quaterniond quat(q(6, 0), q(3, 0), q(4, 0), q(5, 0)); // w, x, y, z
    RPY << pinocchio::rpy::matrixToRpy(quat.toRotationMatrix());
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
    update_trajectory_generator(k, h_estim);

    return 0;
}

Eigen::MatrixXd Planner::get_xref() {return xref;}
Eigen::MatrixXd Planner::get_fsteps() {return fsteps;}
Eigen::MatrixXd Planner::get_gait() {return gait;}
Eigen::MatrixXd Planner::get_goals() {return goals;}
Eigen::MatrixXd Planner::get_vgoals() {return vgoals;}
Eigen::MatrixXd Planner::get_agoals() {return agoals;}

int Planner::roll_exp(int k) {

    // Transfer current gait into past gait
    // If current gait is the same than the first line of past gait we just increment the counter
    if ((gait_c.block(0, 1, 1, 4)).isApprox(gait_p.block(0, 1, 1, 4))) {
        gait_p(0, 0) += 1.0;
    }
    else {  // If current gait is not the same than the first line of past gait we have to insert it
        gait_p.block(1, 0, N0_gait - 1, 5) = gait_p.block(0, 0, N0_gait - 1, 5);
        gait_p.row(0) = gait_c.row(0);
    }

    // Transfert future gait into current gait
    gait_c.row(0) = gait_f.row(0);
    gait_c(0, 0) = 1.0;

    // Age future gait
    if (gait_f(0, 0) == 1.0) {
        gait_f.block(0, 0, N0_gait - 1, 5) = gait_f.block(1, 0, N0_gait - 1, 5);

        // Entering new contact phase, store positions of feet that are now in contact
        if (k != 0) {
            for (int i=0; i<4; i++) {
                if (gait_c(0, 1+i) == 1.0) {
                    o_feet_contact.block(0, 3*i, 1, 3) = fsteps.block(1, 1+3*i, 1, 3);
                }
            }
        }
    }
    else
    {
        gait_f(0, 0) -= 1.0;
    }

    int i = 1;
    while (gait_f(i, 0) > 0.0) {i++;}
    if ((gait_f.block(i-1, 1, 1, 4)).isApprox(gait_f_des.block(0, 1, 1, 4))) {
        gait_f(i-1, 0) += 1.0;
    }
    else
    {
        gait_f.row(i) = gait_f_des.row(0);
        gait_f(i, 0) = 1.0;
    }

    // Age future desired gait
    if (gait_f_des(0, 0) == 1.0) {
        gait_f_des.block(0, 0, N0_gait - 1, 5) = gait_f_des.block(1, 0, N0_gait - 1, 5);
    }
    else
    {
        gait_f_des(0, 0) -= 1.0;
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

    for (int i=0; i<6; i++) {
        lastCoeffs_x[i] = 0.0;
        lastCoeffs_y[i] = 0.0;
    }

}

Eigen::Matrix<double, 11, 1> TrajGen::get_next_foot(double x0, double dx0, double ddx0, double y0, double dy0, double ddy0,
                                        double x1_in, double y1_in, double t0, double t1, double dt) {

    double epsilon = 0.0;
    double t2 = t1;
    double t3 = t0;
    t1 -= 2*epsilon;
    t0 -= epsilon;
    
    if ((t1 - t0) > time_adaptative_disabled) { // adaptative_mode

        //std::cout << "PASS HERE " << x0 << " " << y0 << std::endl; 

        // compute polynoms coefficients for x and y
        Ax5 = (ddx0*std::pow(t0,2) - 2*ddx0*t0*t1 - 6*dx0*t0 + ddx0*std::pow(t1,2) + 6*dx0*t1 + 12 *
                x0 - 12*x1_in)/(2*std::pow((t0 - t1),2)*(std::pow(t0,3) - 3*std::pow(t0,2)*t1 + 3*t0*std::pow(t1,2) - std::pow(t1,3)));
        Ax4 = (30*t0*x1_in - 30*t0*x0 - 30*t1*x0 + 30*t1*x1_in - 2*std::pow(t0,3)*ddx0 - 3*std::pow(t1,3)*ddx0 + 14*std::pow(t0,2)*dx0 - 16*std::pow(t1,2)*dx0 +
                2*t0*t1*dx0 + 4*t0*std::pow(t1,2)*ddx0 + std::pow(t0,2)*t1*ddx0)/(2*std::pow((t0 - t1),2)*(std::pow(t0,3) - 3*std::pow(t0,2)*t1 + 3*t0*std::pow(t1,2) - std::pow(t1,3)));
        Ax3 = (std::pow(t0,4)*ddx0 + 3*std::pow(t1,4)*ddx0 - 8*std::pow(t0,3)*dx0 + 12*std::pow(t1,3)*dx0 + 20*std::pow(t0,2)*x0 - 20*std::pow(t0,2)*x1_in + 20*std::pow(t1,2)*x0 - 20*std::pow(t1,2)*x1_in + 80*t0*t1*x0 - 80*t0 *
                t1*x1_in + 4*std::pow(t0,3)*t1*ddx0 + 28*t0*std::pow(t1,2)*dx0 - 32*std::pow(t0,2)*t1*dx0 - 8*std::pow(t0,2)*std::pow(t1,2)*ddx0)/(2*std::pow((t0 - t1),2)*(std::pow(t0,3) - 3*std::pow(t0,2)*t1 + 3*t0*std::pow(t1,2) - std::pow(t1,3)));
        Ax2 = -(std::pow(t1,5)*ddx0 + 4*t0*std::pow(t1,4)*ddx0 + 3*std::pow(t0,4)*t1*ddx0 + 36*t0*std::pow(t1,3)*dx0 - 24*std::pow(t0,3)*t1*dx0 + 60*t0*std::pow(t1,2)*x0 + 60*std::pow(t0,2)*t1*x0 - 60*t0*std::pow(t1 ,
                2)*x1_in - 60*std::pow(t0,2)*t1*x1_in - 8*std::pow(t0,2)*std::pow(t1,3)*ddx0 - 12*std::pow(t0,2)*std::pow(t1,2)*dx0)/(2*(std::pow(t0,2) - 2*t0*t1 + std::pow(t1,2))*(std::pow(t0,3) - 3*std::pow(t0,2)*t1 + 3*t0*std::pow(t1,2) - std::pow(t1,3)));
        Ax1 = -(2*std::pow(t1,5)*dx0 - 2*t0*std::pow(t1,5)*ddx0 - 10*t0*std::pow(t1,4)*dx0 + std::pow(t0,2)*std::pow(t1,4)*ddx0 + 4*std::pow(t0,3)*std::pow(t1,3)*ddx0 - 3*std::pow(t0,4)*std::pow(t1,2)*ddx0 - 16*std::pow(t0,2) *
                std::pow(t1,3)*dx0 + 24*std::pow(t0,3)*std::pow(t1,2)*dx0 - 60*std::pow(t0,2)*std::pow(t1,2)*x0 + 60*std::pow(t0,2)*std::pow(t1,2)*x1_in)/(2*std::pow((t0 - t1),2)*(std::pow(t0,3) - 3*std::pow(t0,2)*t1 + 3*t0*std::pow(t1,2) - std::pow(t1,3)));
        Ax0 = (2*x1_in*std::pow(t0,5) - ddx0*std::pow(t0,4)*std::pow(t1,3) - 10*x1_in*std::pow(t0,4)*t1 + 2*ddx0*std::pow(t0,3)*std::pow(t1,4) + 8*dx0*std::pow(t0,3)*std::pow(t1,3) + 20*x1_in*std::pow(t0,3)*std::pow(t1,2) - ddx0*std::pow(t0,2)*std::pow(t1,5) - 10*dx0*std::pow(t0 ,
                2)*std::pow(t1,4) - 20*x0*std::pow(t0,2)*std::pow(t1,3) + 2*dx0*t0*std::pow(t1,5) + 10*x0*t0*std::pow(t1,4) - 2*x0*std::pow(t1,5))/(2*(std::pow(t0,2) - 2*t0*t1 + std::pow(t1,2))*(std::pow(t0,3) - 3*std::pow(t0,2)*t1 + 3*t0*std::pow(t1,2) - std::pow(t1,3)));

        Ay5 = (ddy0*std::pow(t0,2) - 2*ddy0*t0*t1 - 6*dy0*t0 + ddy0*std::pow(t1,2) + 6*dy0*t1 + 12 *
                y0 - 12*y1_in)/(2*std::pow((t0 - t1),2)*(std::pow(t0,3) - 3*std::pow(t0,2)*t1 + 3*t0*std::pow(t1,2) - std::pow(t1,3)));
        Ay4 = (30*t0*y1_in - 30*t0*y0 - 30*t1*y0 + 30*t1*y1_in - 2*std::pow(t0,3)*ddy0 - 3*std::pow(t1,3)*ddy0 + 14*std::pow(t0,2)*dy0 - 16*std::pow(t1,2)*dy0 +
                2*t0*t1*dy0 + 4*t0*std::pow(t1,2)*ddy0 + std::pow(t0,2)*t1*ddy0)/(2*std::pow((t0 - t1),2)*(std::pow(t0,3) - 3*std::pow(t0,2)*t1 + 3*t0*std::pow(t1,2) - std::pow(t1,3)));
        Ay3 = (std::pow(t0,4)*ddy0 + 3*std::pow(t1,4)*ddy0 - 8*std::pow(t0,3)*dy0 + 12*std::pow(t1,3)*dy0 + 20*std::pow(t0,2)*y0 - 20*std::pow(t0,2)*y1_in + 20*std::pow(t1,2)*y0 - 20*std::pow(t1,2)*y1_in + 80*t0*t1*y0 - 80*t0 *
                t1*y1_in + 4*std::pow(t0,3)*t1*ddy0 + 28*t0*std::pow(t1,2)*dy0 - 32*std::pow(t0,2)*t1*dy0 - 8*std::pow(t0,2)*std::pow(t1,2)*ddy0)/(2*std::pow((t0 - t1),2)*(std::pow(t0,3) - 3*std::pow(t0,2)*t1 + 3*t0*std::pow(t1,2) - std::pow(t1,3)));
        Ay2 = -(std::pow(t1,5)*ddy0 + 4*t0*std::pow(t1,4)*ddy0 + 3*std::pow(t0,4)*t1*ddy0 + 36*t0*std::pow(t1,3)*dy0 - 24*std::pow(t0,3)*t1*dy0 + 60*t0*std::pow(t1,2)*y0 + 60*std::pow(t0,2)*t1*y0 - 60*t0*std::pow(t1 ,
                2)*y1_in - 60*std::pow(t0,2)*t1*y1_in - 8*std::pow(t0,2)*std::pow(t1,3)*ddy0 - 12*std::pow(t0,2)*std::pow(t1,2)*dy0)/(2*(std::pow(t0,2) - 2*t0*t1 + std::pow(t1,2))*(std::pow(t0,3) - 3*std::pow(t0,2)*t1 + 3*t0*std::pow(t1,2) - std::pow(t1,3)));
        Ay1 = -(2*std::pow(t1,5)*dy0 - 2*t0*std::pow(t1,5)*ddy0 - 10*t0*std::pow(t1,4)*dy0 + std::pow(t0,2)*std::pow(t1,4)*ddy0 + 4*std::pow(t0,3)*std::pow(t1,3)*ddy0 - 3*std::pow(t0,4)*std::pow(t1,2)*ddy0 - 16*std::pow(t0,2) *
                std::pow(t1,3)*dy0 + 24*std::pow(t0,3)*std::pow(t1,2)*dy0 - 60*std::pow(t0,2)*std::pow(t1,2)*y0 + 60*std::pow(t0,2)*std::pow(t1,2)*y1_in)/(2*std::pow((t0 - t1),2)*(std::pow(t0,3) - 3*std::pow(t0,2)*t1 + 3*t0*std::pow(t1,2) - std::pow(t1,3)));
        Ay0 = (2*y1_in*std::pow(t0,5) - ddy0*std::pow(t0,4)*std::pow(t1,3) - 10*y1_in*std::pow(t0,4)*t1 + 2*ddy0*std::pow(t0,3)*std::pow(t1,4) + 8*dy0*std::pow(t0,3)*std::pow(t1,3) + 20*y1_in*std::pow(t0,3)*std::pow(t1,2) - ddy0*std::pow(t0,2)*std::pow(t1,5) - 10*dy0*std::pow(t0 ,
                2)*std::pow(t1,4) - 20*y0*std::pow(t0,2)*std::pow(t1,3) + 2*dy0*t0*std::pow(t1,5) + 10*y0*t0*std::pow(t1,4) - 2*y0*std::pow(t1,5))/(2*(std::pow(t0,2) - 2*t0*t1 + std::pow(t1,2))*(std::pow(t0,3) - 3*std::pow(t0,2)*t1 + 3*t0*std::pow(t1,2) - std::pow(t1,3)));

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
    }
    else {
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
    Az6 = -h/(std::pow((t2/2),3)*std::pow((t2 - t2/2),3));
    Az5 = (3*t2*h)/(std::pow((t2/2),3)*std::pow((t2 - t2/2),3));
    Az4 = -(3*std::pow(t2,2)*h)/(std::pow((t2/2),3)*std::pow((t2 - t2/2),3));
    Az3 = (std::pow(t2,3)*h)/(std::pow((t2/2),3)*std::pow((t2 - t2/2),3));

    // Get the next point
    double ev = t0+dt;
    double evz = t3+dt;

    result(6, 0) = Az3*std::pow(evz,3) + Az4*std::pow(evz,4) + Az5*std::pow(evz,5) + Az6*std::pow(evz,6); // pos Z
    result(7, 0) = 3*Az3*std::pow(evz,2) + 4*Az4*std::pow(evz,3) + 5*Az5*std::pow(evz,4) + 6*Az6*std::pow(evz,5);  // vel Z
    result(8, 0) = 2*3*Az3*evz + 3*4*Az4*std::pow(evz,2) + 4*5*Az5*std::pow(evz,3) + 5*6*Az6*std::pow(evz,4);  // acc Z
    result(9, 0) = x1; // current goal x
    result(10, 0) = y1; // current goal y

    
    /*std::cout << "X1/Y1: " << x1 << " / " << y1 << std::endl;
    for (int o=0; o<6; o++) {
        std::cout << lastCoeffs_x[o] << " / ";
    }
    std::cout << std::endl;
    for (int o=0; o<6; o++) {
        std::cout << lastCoeffs_y[o] << " / ";
    }
    std::cout << std::endl;*/

    if ((t3 < epsilon) || (t3 > (t2-epsilon))) { // Just vertical motion
        // std::cout << "LOCKED" << std::endl;
        result(0, 0) = x0;
        result(1, 0) = 0.0;
        result(2, 0) = 0.0;
        result(3, 0) = y0;
        result(4, 0) = 0.0;
        result(5, 0) = 0.0;
    }
    else {
        // std::cout << "NOT LOCKED" << std::endl;
        // pos, vel, acc X
        result(0, 0) = Ax0 + Ax1*ev + Ax2*std::pow(ev,2) + Ax3*std::pow(ev,3) + Ax4*std::pow(ev,4) + Ax5*std::pow(ev,5);
        result(1, 0) = Ax1 + 2*Ax2*ev + 3*Ax3*std::pow(ev,2) + 4*Ax4*std::pow(ev,3) + 5*Ax5*std::pow(ev,4);
        result(2, 0) = 2*Ax2 + 3*2*Ax3*ev + 4*3*Ax4*std::pow(ev,2) + 5*4*Ax5*std::pow(ev,3);

        // pos, vel, acc Y
        result(3, 0) = Ay0 + Ay1*ev + Ay2*std::pow(ev,2) + Ay3*std::pow(ev,3) + Ay4*std::pow(ev,4) + Ay5*std::pow(ev,5);
        result(4, 0) = Ay1 + 2*Ay2*ev + 3*Ay3*std::pow(ev,2) + 4*Ay4*std::pow(ev,3) + 5*Ay5*std::pow(ev,4);
        result(5, 0) = 2*Ay2 + 3*2*Ay3*ev + 4*3*Ay4*std::pow(ev,2) + 5*4*Ay5*std::pow(ev,3);
    }

    return result;

}
