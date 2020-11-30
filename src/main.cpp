#include <cstdlib>
#include <iostream>
#include <boost/smart_ptr/shared_ptr.hpp>

#include <Eigen/Core>
#include "example-adder/gepadd.hpp"
#include "example-adder/MPC.hpp"
#include "other/st_to_cc.hpp"
#include "example-adder/Planner.hpp"
#include "pinocchio/math/rpy.hpp"

int main(int argc, char** argv) {
  if (argc == 3) {
    int arg_a = std::atoi(argv[1]), arg_b = std::atoi(argv[2]);
    std::cout << "The sum of " << arg_a << " and " << arg_b << " is: ";
    std::cout << gepetto::example::add(arg_a, arg_b) << std::endl;

    /*Eigen::MatrixXd test_fsteps = Eigen::MatrixXd::Zero(20, 13);
    test_fsteps.row(0) << 15, 0.19, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.19, -0.15, 0.0;
    test_fsteps.row(1) << 1, 0.19, 0.15, 0.0, 0.19, -0.15, 0.0, -0.19, 0.15, 0.0, -0.19, -0.15, 0.0;
    test_fsteps.row(2) << 15, 0.0, 0.0, 0.0, 0.19, -0.15, 0.0, -0.19, 0.15, 0.0, 0.0, 0.0, 0.0;
    test_fsteps.row(3) << 1, 0.19, 0.15, 0.0, 0.19, -0.15, 0.0, -0.19, 0.15, 0.0, -0.19, -0.15, 0.0;

    Eigen::Matrix<double, 12, Eigen::Dynamic> test_xref = Eigen::Matrix<double, 12, Eigen::Dynamic>::Zero(12, 33);
    test_xref.row(2) = 0.17 * Eigen::Matrix<double, 1, Eigen::Dynamic>::Ones(1, 33);

    MPC test(0.02f, 32, 0.64f);
    test.run(0, test_xref, test_fsteps);*/
    /*double * result;
    result = test.get_latest_result();

    test_fsteps(0, 0) = 14;
    test_fsteps.row(4) = test_fsteps.row(0);
    test_fsteps(4, 0) = 1;

    test.run(1, test_xref, test_fsteps);

    std::cout << test.gethref() << std::endl;*/
    /*std::cout << test.getA() << std::endl;
    std::cout << test.getML() << std::endl;
    std::cout << test.getDay() << std::endl;
    std::cout << test.getMonth() << std::endl;
    std::cout << test.getYear() << std::endl;*/

    /*Eigen::MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    std::cout << m << std::endl;

    Eigen::MatrixXf M1(3,6);    // Column-major storage
    M1 << 1, 2, 3,  4,  5,  6,
          7, 8, 9, 10, 11, 12,
          13, 14, 15, 16, 17, 18;

    Eigen::Map<Eigen::MatrixXf> M2(M1.data(), M1.size(), 1);
    std::cout << "M2:" << std::endl << M2 << std::endl;*/

    std::cout << "Test quaternion" << std::endl;
    Eigen::MatrixXd qt = Eigen::MatrixXd::Zero(4, 1);
    qt(3, 0) = 1.0;
    qt << 0.0342708, 0.1060205, 0.1534393, 0.9818562;
    Eigen::Quaterniond quat(qt(3, 0), qt(0, 0), qt(1, 0), qt(2, 0));
    std::cout << quat.toRotationMatrix() << std::endl;
    std::cout << pinocchio::rpy::matrixToRpy(quat.toRotationMatrix()) << std::endl;

    std::cout << pinocchio::rpy::rpyToMatrix(0.1, 0.2, 0.3) << std::endl;

    quat = Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitY()) *
           Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitX());
    std::cout << pinocchio::rpy::matrixToRpy(quat.toRotationMatrix()) << std::endl;

    std::cout << "Init Starting " << std::endl;
    double dt_in = 0.02;
    double dt_tsid_in = 0.002;
    double T_gait_in = 0.64;
    double T_mpc_in = 0.40;
    int k_mpc_in = 10;
    bool on_solo8_in = false;
    double h_ref_in = 0.21;
    Eigen::MatrixXd fsteps_in(3, 4);
    fsteps_in << 0.1946, 0.1946, -0.1946, -0.1946, 0.14695, -0.14695, 0.14695, -0.14695, 0.0, 0.0, 0.0, 0.0;

    Planner planner(dt_in, dt_tsid_in, T_gait_in, T_mpc_in, k_mpc_in, on_solo8_in, h_ref_in, fsteps_in);
    std::cout << "Init OK " << std::endl;

    Eigen::MatrixXd q = Eigen::MatrixXd::Zero(7, 1);
    q << 0.0, 0.0, 0.21, 0.0, 0.0, 0.0, 1.0;
    Eigen::MatrixXd v = Eigen::MatrixXd::Zero(6, 1);
    v << 0.02, 0.0, 0.0, 0.0, 0.0, 0.0;
    Eigen::MatrixXd b_vref_in = Eigen::MatrixXd::Zero(6, 1);
    b_vref_in << 0.02, 0.0, 0.0, 0.0, 0.0, 0.05;

    /*Eigen::Matrix<double, 6, 5> foo = Eigen::Matrix<double, 6, 5>::Zero();
    foo.row(0) << 8.0, 1.0, 0.0, 0.0, 1.0;
    std::cout << "#### " << std::endl << foo << std::endl;
    foo.block<5, 5>(1, 0) = foo.block<5, 5>(0, 0);
    std::cout << "#### " << std::endl << foo << std::endl;
    foo.row(0) << 1.0, 0.0, 1.0, 1.0, 0.0;
    std::cout << "#### " << std::endl << foo << std::endl;*/

    std::cout << "#### " << std::endl;
    planner.Print();

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
