#include <cstdlib>
#include <iostream>
#include <boost/smart_ptr/shared_ptr.hpp>

#include <Eigen/Core>
#include "quadruped-reactive-walking/gepadd.hpp"
#include "quadruped-reactive-walking/MPC.hpp"
#include "other/st_to_cc.hpp"
#include "quadruped-reactive-walking/Planner.hpp"
#include "pinocchio/math/rpy.hpp"

int main(int argc, char** argv) {
  if (argc == 3) {
    int arg_a = std::atoi(argv[1]), arg_b = std::atoi(argv[2]);
    std::cout << "The sum of " << arg_a << " and " << arg_b << " is: ";
    std::cout << gepetto::example::add(arg_a, arg_b) << std::endl;

    std::cout << "-- Test quaternion conversion --" << std::endl;
    std::cout << "Initial Roll Pitch Yaw : 0.1 0.2 0.3" << std::endl;
    std::cout << "RPY -> Quaternion -> Matrix -> RPY" << std::endl;
    Eigen::Quaterniond quat = Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ()) *
                              Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitY()) *
                              Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitX());
    std::cout << pinocchio::rpy::matrixToRpy(quat.toRotationMatrix()) << std::endl;

    std::cout << "-- Test Planner --" << std::endl;

    double dt_in = 0.02;              // Time step of the MPC
    double dt_wbc_in = 0.002;         // Time step of the WBC
    double T_gait_in = 0.64;          // Period of the gait
    double T_mpc_in = 0.40;           // Prediction horizon of the mpc
    int k_mpc_in = 10;                // dt_in / dt_wbc_in
    bool on_solo8_in = false;         // not used (lock shoulder joints)
    double h_ref_in = 0.21;           // target height for the base
    Eigen::MatrixXd fsteps_in(3, 4);  // Initial position of footsteps
    fsteps_in << 0.1946, 0.1946, -0.1946, -0.1946, 0.14695, -0.14695, 0.14695, -0.14695, 0.0, 0.0, 0.0, 0.0;
    Planner planner(dt_in, dt_wbc_in, T_gait_in, T_mpc_in, k_mpc_in, on_solo8_in, h_ref_in, fsteps_in);
    std::cout << "Initialization Planner OK " << std::endl;

    Eigen::MatrixXd q = Eigen::MatrixXd::Zero(7, 1);  // Position/Orientation of the base
    q << 0.0, 0.0, 0.21, 0.0, 0.0, 0.0, 1.0;
    Eigen::MatrixXd v = Eigen::MatrixXd::Zero(6, 1);  // Velocity of the base
    v << 0.02, 0.0, 0.0, 0.0, 0.0, 0.0;
    Eigen::MatrixXd b_vref_in = Eigen::MatrixXd::Zero(6, 1);  // Reference velocity of the base
    b_vref_in << 0.02, 0.0, 0.0, 0.0, 0.0, 0.05;

    std::cout << "#### " << std::endl;
    planner.Print();  // Initial state of the planner

    // Running the planner and displaying the state once every ten iterations
    for (int k = 0; k < 100; k++) {
      planner.run_planner(k, q, v, b_vref_in, 0.21, 0.0);
      if (k % 10 == 0) {
        std::cout << "#### " << k << std::endl;
        planner.Print();
      }
    }

    return EXIT_SUCCESS;
  } else {
    std::cerr << "This program needs 2 integers" << std::endl;
    return EXIT_FAILURE;
  }
}
