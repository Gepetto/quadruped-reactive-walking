#include "quadruped-reactive-walking/InvKin.hpp"

InvKin::InvKin(double dt_in) {

  // Parameters from the main controller
  dt = dt_in;

  // Reference position of feet
  feet_position_ref << 0.1946, 0.1946, -0.1946, -0.1946, 0.14695, -0.14695, 0.14695, -0.14695, 0.0191028, 0.0191028, 0.0191028, 0.0191028;
}

InvKin::InvKin() {}

Eigen::Matrix<double, 1, 3> InvKin::cross3(Eigen::Matrix<double, 1, 3> left, Eigen::Matrix<double, 1, 3> right) {
    Eigen::Matrix<double, 1, 3> res;
    res << left(0, 1) * right(0, 2) - left(0, 2) * right(0, 1),
           left(0, 2) * right(0, 0) - left(0, 0) * right(0, 2),
           left(0, 0) * right(0, 1) - left(0, 1) * right(0, 0);
    return res;
}


Eigen::MatrixXd InvKin::refreshAndCompute(const Eigen::MatrixXd &x_cmd, const Eigen::MatrixXd &contacts,
                                          const Eigen::MatrixXd &goals, const Eigen::MatrixXd &vgoals, const Eigen::MatrixXd &agoals,
                                          const Eigen::MatrixXd &posf, const Eigen::MatrixXd &vf, const Eigen::MatrixXd &wf, const Eigen::MatrixXd &af,
                                          const Eigen::MatrixXd &Jf, const Eigen::MatrixXd &posb, const Eigen::MatrixXd &rotb, const Eigen::MatrixXd &vb,
                                          const Eigen::MatrixXd &ab, const Eigen::MatrixXd &Jb) {

    // Update contact status of the feet
    flag_in_contact.block(0, 0, 1, 4) = contacts.block(0, 0, 1, 4);

    // Update position, velocity and acceleration references for the feet
    for (int i = 0; i < 4; i++) {
        feet_position_ref.block(i, 0, 1, 3) = goals.block(0, i, 3, 1).transpose();
        feet_velocity_ref.block(i, 0, 1, 3) = vgoals.block(0, i, 3, 1).transpose();
        feet_acceleration_ref.block(i, 0, 1, 3) = agoals.block(0, i, 3, 1).transpose();
    }

    // Update position and velocity reference for the base
    base_position_ref = x_cmd.block(0, 0, 1, 3);
    base_orientation_ref = pinocchio::rpy::rpyToMatrix(x_cmd(0, 3), x_cmd(0, 4), x_cmd(0, 5));
    base_linearvelocity_ref = x_cmd.block(0, 6, 1, 3);
    base_angularvelocity_ref = x_cmd.block(0, 9, 1, 3);

    /* std::cout << base_position_ref << std::endl;
    std::cout << base_orientation_ref << std::endl;
    std::cout << base_linearvelocity_ref << std::endl;
    std::cout << base_angularvelocity_ref << std::endl;
    std::cout << "--" << std::endl; */

    return computeInvKin(posf, vf, wf, af, Jf, posb, rotb, vb, ab, Jb);
}

Eigen::MatrixXd InvKin::computeInvKin(const Eigen::MatrixXd &posf, const Eigen::MatrixXd &vf, const Eigen::MatrixXd &wf, const Eigen::MatrixXd &af,
                                      const Eigen::MatrixXd &Jf, const Eigen::MatrixXd &posb, const Eigen::MatrixXd &rotb, const Eigen::MatrixXd &vb, const Eigen::MatrixXd &ab,
                                      const Eigen::MatrixXd &Jb) {

    // Process feet
    for (int i = 0; i < 4; i++) {

        pfeet_err.row(i) = feet_position_ref.row(i) - posf.row(i);
        vfeet_ref.row(i) = feet_velocity_ref.row(i);

        afeet.row(i) = + Kp_flyingfeet * pfeet_err.row(i) - Kd_flyingfeet * (vf.row(i)-feet_velocity_ref.row(i)) + feet_acceleration_ref.row(i);
        if (flag_in_contact(0, i)) {
            afeet.row(i) *= 1.0; // Set to 0.0 to disable position/velocity control of feet in contact
        }
        afeet.row(i) -= af.row(i) + cross3(wf.row(i), vf.row(i)); // Drift
    }
    J.block(6, 0, 12, 18) = Jf.block(0, 0, 12, 18);

    // Process base position
    e_basispos = base_position_ref - posb;
    abasis = Kp_base_position * e_basispos - Kd_base_position * (vb.block(0, 0, 1, 3) - base_linearvelocity_ref);
    abasis -= ab.block(0, 0, 1, 3) + cross3(vb.block(0, 3, 1, 3), vb.block(0, 0, 1, 3)); // Drift
    x_ref.block(0, 0, 3, 1) = base_position_ref;
    x.block(0, 0, 3, 1) = posb;
    dx_ref.block(0, 0, 3, 1) = base_linearvelocity_ref;
    x_ref.block(0, 0, 3, 1) = vb.block(0, 0, 1, 3);

    // Process base orientation
    e_basisrot = -base_orientation_ref * pinocchio::log3(base_orientation_ref.transpose() * rotb);
    awbasis = Kp_base_orientation * e_basisrot - Kd_base_orientation * (vb.block(0, 3, 1, 3) - base_angularvelocity_ref);
    awbasis -= ab.block(0, 3, 1, 3);

    x_ref.block(3, 0, 3, 1) = Eigen::Matrix<double, 1, 3>::Zero();
    x.block(3, 0, 3, 1) = Eigen::Matrix<double, 1, 3>::Zero();
    dx_ref.block(3, 0, 3, 1) = base_angularvelocity_ref;
    x_ref.block(3, 0, 3, 1) = vb.block(0, 3, 1, 3);

    J.block(0, 0, 6, 18) = Jb.block(0, 0, 6, 18); // Position and orientation

    acc.block(0, 0, 1, 3) = abasis;
    acc.block(0, 3, 1, 3) = awbasis;
    for (int i = 0; i < 4; i++) {
        acc.block(0, 6+3*i, 1, 3) = afeet.row(i);
    }

    x_err.block(0, 0, 1, 3) = e_basispos;
    x_err.block(0, 3, 1, 3) = e_basisrot;
    for (int i = 0; i < 4; i++) {
        x_err.block(0, 6+3*i, 1, 3) = pfeet_err.row(i);
    }

    dx_r.block(0, 0, 1, 3) = base_linearvelocity_ref;
    dx_r.block(0, 3, 1, 3) = base_angularvelocity_ref;
    for (int i = 0; i < 4; i++) {
        dx_r.block(0, 6+3*i, 1, 3) = vfeet_ref.row(i);
    }

    // std::cout << "J" << std::endl << J << std::endl;

    invJ = pseudoInverse(J);
    
    // std::cout << "invJ" << std::endl << invJ << std::endl;
    // std::cout << "acc" << std::endl << acc << std::endl;

    ddq = invJ * acc.transpose();
    q_step = invJ * x_err.transpose(); // Need to be pin.integrate in Python to get q_cmd
    dq_cmd = invJ * dx_r.transpose();

    // std::cout << "q_step" << std::endl << q_step << std::endl;
    // std::cout << "dq_cmd" << std::endl << dq_cmd << std::endl;

    return ddq;
}

Eigen::MatrixXd InvKin::get_q_step() { return q_step; }
Eigen::MatrixXd InvKin::get_dq_cmd() { return dq_cmd; }