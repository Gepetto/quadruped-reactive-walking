#include <cstdlib>
#include <iostream>
#include <boost/smart_ptr/shared_ptr.hpp>

#include <Eigen/Core>
#include "qrw/gepadd.hpp"
#include "qrw/MPC.hpp"
#include "other/st_to_cc.hpp"
#include "pinocchio/math/rpy.hpp"
#include "qrw/Gait.hpp"
#include "qrw/Params.hpp"

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

    /*std::cout << "-- Test Planner --" << std::endl;

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
    }*/

    /*EiquadprogFast qp;
    qp.reset(16, 0, 16);
    Eigen::MatrixXd Q_qp = Eigen::MatrixXd::Zero(16,16);
    Eigen::VectorXd C_qp = Eigen::VectorXd::Zero(16);
    Eigen::MatrixXd Aeq = Eigen::MatrixXd::Zero(0, 16);
    Eigen::VectorXd Beq = Eigen::VectorXd::Zero(0);
    Eigen::MatrixXd Aineq = Eigen::MatrixXd::Zero(16, 16);
    Eigen::VectorXd Bineq = Eigen::VectorXd::Zero(16);
    Eigen::VectorXd x_qp = Eigen::VectorXd::Zero(16);

    Q_qp <<
     20.3068,    4.19814,    13.5679,   -2.54082,          0,          0,          0,          0,          0, 0, 0, 0,
    1.84671,   -11.9805,   -3.77491,   -17.6021, 4.19814,    10.1805,   -4.47586,    1.50651,          0,          0,
    0,          0,          0,          0,          0,          0,  0.0678086, -0.0493709,   -3.58335,   -3.70053,
     13.5679,   -4.47586,    16.9657,   -1.07805,          0,          0,          0,          0,          0, 0, 0, 0,
    1.54046,   -10.3655,  -0.113754,   -12.0197, -2.54082,    1.50651,   -1.07805,    2.96928,          0,          0,
    0,          0,          0,          0,          0,          0,  -0.238438,    1.56562,  0.0778063,    1.88187, 0,
    0,          0,          0,       2.62,          1,          1,      -0.62,          0,          0,          0, 0,
    0,          0,          0,          0, 0,          0,          0,          0,          1,       2.62,      -0.62,
    1,          0,          0,          0,          0,          0,          0,          0,          0, 0,          0,
    0,          0,          1,      -0.62,       2.62,          1,          0,          0,          0,          0, 0,
    0,          0,          0, 0,          0,          0,          0,      -0.62,          1,          1,       2.62,
    0,          0,          0,          0,          0,          0,          0,          0, 0,          0,          0,
    0,          0,          0,          0,          0,       2.62,          1,          1,      -0.62,          0, 0,
    0,          0, 0,          0,          0,          0,          0,          0,          0,          0,          1,
    2.62,      -0.62,          1,          0,          0,          0,          0, 0,          0,          0, 0, 0, 0,
    0,          0,          1,      -0.62,       2.62,          1,          0,          0,          0,          0, 0,
    0,          0,          0,          0,          0,          0,          0,      -0.62,          1,          1,
    2.62,          0,          0,          0,          0, 1.84671,  0.0678086,    1.54046,  -0.238438,          0, 0,
    0,          0,          0,          0,          0,          0,    2.96016,   -1.05119,    1.49914,    -2.5122,
    -11.9805, -0.0493709,   -10.3655,    1.56562,          0,          0,          0,          0,          0, 0, 0, 0,
    -1.05119,    17.0288,   -4.49649,    13.5835, -3.77491,   -3.58335,  -0.113754,  0.0778063,          0,          0,
    0,          0,          0,          0,          0,          0,    1.49914,   -4.49649,    10.2498,    4.25414,
    -17.6021,   -3.70053,   -12.0197,    1.88187,          0,          0,          0,          0,          0, 0, 0, 0,
    -2.5122,    13.5835,    4.25414,    20.3499;

    C_qp <<
        -12.97,
      -12.7995,
      -12.8596,
      -12.6892,
    -2.6356e-07,
    -2.6356e-07,
    -2.6356e-07,
    -2.6356e-07,
    -2.6356e-07,
    -2.6356e-07,
    -2.6356e-07,
    -2.6356e-07,
      -12.7546,
      -12.4232,
      -12.7665,
      -12.4351;

    for (int i = 0; i < 16; i++) {
        Aineq(i, i) = 1.;
    }

    std::cout << "Matrices:" << std::endl;
    std::cout << Q_qp << std::endl << "--" << std::endl;
    std::cout << C_qp << std::endl << "--" << std::endl;
    std::cout << Aeq << std::endl << "--" << std::endl;
    std::cout << Beq << std::endl << "--" << std::endl;
    std::cout << Aineq << std::endl << "--" << std::endl;
    std::cout << Bineq << std::endl << "--" << std::endl;

    qp.solve_quadprog(Q_qp, C_qp, Aeq, Beq, Aineq, Bineq, x_qp);

    Eigen::VectorXd dx = Eigen::VectorXd::Zero(16);
    dx(0) = 0.01;
    dx(1) = 0.01;
    dx(2) = 0.01;
    dx(3) = 0.01;
    dx(12) = 0.01;
    dx(13) = 0.01;
    dx(14) = 0.01;
    dx(15) = 0.01;

    std::cout << "Cost for sol   : " << 0.5 * x_qp.transpose() * Q_qp * x_qp + x_qp.transpose() * C_qp << std::endl;
    std::cout << "Cost for sol-dx: " << 0.5 * (x_qp-dx).transpose() * Q_qp * (x_qp-dx) + (x_qp-dx).transpose() * C_qp
    << std::endl; std::cout << "Cost for sol+dx: " << 0.5 * (x_qp+dx).transpose() * Q_qp * (x_qp+dx) +
    (x_qp+dx).transpose() * C_qp << std::endl;*/

    /*Eigen::Matrix<double, 20, 4> desiredGait_;
    int N = 5;
    Eigen::Matrix<double, 1, 4> sequence;
    sequence << 0, 1, 1, 1;
    desiredGait_.block(0, 0, N, 4) = sequence.colwise().replicate(N);
    sequence << 1, 0, 1, 1;
    desiredGait_.block(N, 0, N, 4) = sequence.colwise().replicate(N);
    sequence << 1, 1, 0, 1;
    desiredGait_.block(2*N, 0, N, 4) = sequence.colwise().replicate(N);
    sequence << 1, 1, 1, 0;
    desiredGait_.block(3*N, 0, N, 4) = sequence.colwise().replicate(N);

    std::cout << desiredGait_ << std::endl;
    std::cout << "##" << std::endl;
    */
    /*Gait gait = Gait();
    gait.initialize(0.02, 0.32, 0.16);

    std::cout << gait.getPastGait() << std::endl << "##" << std::endl;
    std::cout << gait.getCurrentGait() << std::endl << "##" << std::endl;
    std::cout << gait.getDesiredGait() << std::endl << "##" << std::endl;

    for (int k = 0; k < 10; k++)
    {
      gait.updateGait(k, 1, VectorN::Zero(19), 0);
      std::cout << "## " << k << " ##" << std::endl;
      std::cout << gait.getPastGait() << std::endl << "##" << std::endl;
      std::cout << gait.getCurrentGait() << std::endl << "##" << std::endl;
      std::cout << gait.getDesiredGait() << std::endl << "##" << std::endl;
    }*/

    // std::cout << yaml_control_interface::RobotFromYamlFile(CONFIG_SOLO12_YAML) << std::endl;
    Params params = Params();
    std::cout << params.interface << std::endl;
    std::cout << params.SIMULATION << std::endl;
    std::cout << params.LOGGING << std::endl;
    std::cout << params.PLOTTING << std::endl;

    return EXIT_SUCCESS;
  } else {
    std::cerr << "This program needs 2 integers" << std::endl;
    return EXIT_FAILURE;
  }
}
