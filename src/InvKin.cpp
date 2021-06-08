#include "qrw/InvKin.hpp"

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


Eigen::MatrixXd InvKin::refreshAndCompute(const Eigen::MatrixXd &contacts,
                                          const Eigen::MatrixXd &goals, const Eigen::MatrixXd &vgoals, const Eigen::MatrixXd &agoals,
                                          const Eigen::MatrixXd &posf, const Eigen::MatrixXd &vf, const Eigen::MatrixXd &wf,
                                          const Eigen::MatrixXd &af, const Eigen::MatrixXd &Jf) {

    // Update contact status of the feet
    flag_in_contact.block(0, 0, 1, 4) = contacts.block(0, 0, 1, 4);

    // Update position, velocity and acceleration references for the feet
    for (int i = 0; i < 4; i++) {
        feet_position_ref.block(i, 0, 1, 3) = goals.block(0, i, 3, 1).transpose();
        feet_velocity_ref.block(i, 0, 1, 3) = vgoals.block(0, i, 3, 1).transpose();
        feet_acceleration_ref.block(i, 0, 1, 3) = agoals.block(0, i, 3, 1).transpose();
    }

    // Process feet
    for (int i = 0; i < 4; i++) {

        pfeet_err.row(i) = feet_position_ref.row(i) - posf.row(i);
        vfeet_ref.row(i) = feet_velocity_ref.row(i);

        afeet.row(i) = + Kp_flyingfeet * pfeet_err.row(i) - Kd_flyingfeet * (vf.row(i)-feet_velocity_ref.row(i)) + feet_acceleration_ref.row(i);
        if (flag_in_contact(0, i)) {
            afeet.row(i) *= 0.0; // Set to 0.0 to disable position/velocity control of feet in contact
        }
        afeet.row(i) -= af.row(i) + cross3(wf.row(i), vf.row(i)); // Drift
    }

    // Store data and invert the Jacobian
    for (int i = 0; i < 4; i++) {
        acc.block(0, 3*i, 1, 3) = afeet.row(i);
        x_err.block(0, 3*i, 1, 3) = pfeet_err.row(i);
        dx_r.block(0, 3*i, 1, 3) = vfeet_ref.row(i);
        invJ.block(3*i, 3*i, 3, 3) = Jf.block(3*i, 3*i, 3, 3).inverse();
    }

    // Once Jacobian has been inverted we can get command accelerations, velocities and positions
    ddq = invJ * acc.transpose();
    dq_cmd = invJ * dx_r.transpose();
    q_step = invJ * x_err.transpose(); // Not a position but a step in position

    /*
    std::cout << "J" << std::endl << Jf << std::endl;
    std::cout << "invJ" << std::endl << invJ << std::endl;
    std::cout << "acc" << std::endl << acc << std::endl;
    std::cout << "q_step" << std::endl << q_step << std::endl;
    std::cout << "dq_cmd" << std::endl << dq_cmd << std::endl;
    */

    return ddq;
}

Eigen::MatrixXd InvKin::get_q_step() { return q_step; }
Eigen::MatrixXd InvKin::get_dq_cmd() { return dq_cmd; }