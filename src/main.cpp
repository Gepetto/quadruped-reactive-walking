#include <cstdlib>
#include <iostream>
#include <boost/smart_ptr/shared_ptr.hpp>

#include <Eigen/Core>
#include "example-adder/gepadd.hpp"
#include "example-adder/MPC.hpp"
#include "other/st_to_cc.hpp"
#include "example-adder/Planner.hpp"

#include "pinocchio/math/rpy.hpp"
#include "curves/fwd.h"
#include "curves/bezier_curve.h"
// #include "curves/splines.h"
#include "curves/helpers/effector_spline.h"

/*namespace curves {
typedef exact_cubic<double, double, true, Eigen::Matrix<double, 1, 1> > exact_cubic_one;
typedef exact_cubic_t::spline_constraints spline_constraints_t;

typedef std::pair<double, pointX_t> Waypoint;
typedef std::vector<Waypoint> T_Waypoint;
typedef Eigen::Matrix<double, 1, 1> point_one;
typedef std::pair<double, point_one> WaypointOne;
typedef std::vector<WaypointOne> T_WaypointOne;
typedef std::pair<pointX_t, pointX_t> pair_point_tangent_t;
typedef std::vector<pair_point_tangent_t, Eigen::aligned_allocator<pair_point_tangent_t> > t_pair_point_tangent_t;

const double margin = 1e-3;
bool QuasiEqual(const double a, const double b) { return std::fabs(a - b) < margin; }
bool QuasiEqual(const point3_t a, const point3_t b) {
  bool equal = true;
  for (size_t i = 0; i < 3; ++i) {
    equal = equal && QuasiEqual(a[i], b[i]);
  }
  return equal;
}
}  // End namespace curves*/

using namespace curves;

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

    /*typedef std::pair<double, Eigen::Vector3d> Waypoint;
    typedef std::vector<Waypoint> T_Waypoint;

    // loading helper class namespace
    using namespace curves::helpers;

    // Create waypoints
    T_Waypoint waypoints;
    waypoints.push_back(std::make_pair(0., Eigen::Vector3d(0,0,0)));
    waypoints.push_back(std::make_pair(1., Eigen::Vector3d(0.5,0.5,0.5)));
    waypoints.push_back(std::make_pair(2., Eigen::Vector3d(1,1,0)));

    exact_cubic_t* eff_traj = effector_spline(waypoints.begin(),waypoints.end());

    // evaluate spline
    std::cout << (*eff_traj)(0.) << std::endl; // (0,0,0)
    std::cout << (*eff_traj)(2.) << std::endl; // (1,1,0)*/

    point3_t a(1, 2, 0);
    point3_t b(2, 3, 0.5);  // * (1000/312.5));
    point3_t c(3, 4, 1);
    std::cout << a(0) << " " << a(1) << " " << a(2) << std::endl;
    bezier_t::curve_constraints_t constraints(3);
    constraints.init_vel = point3_t(-1, -1, 0);
    constraints.init_acc = point3_t(-1, -2, 0);
    constraints.end_vel = point3_t(-10, -10, 0);
    constraints.end_acc = point3_t(-20, -20, 0);
    std::vector<point3_t> params;
    params.push_back(a);
    params.push_back(b);
    params.push_back(c);
    bezier_t::num_t T_min = 0.0;
    bezier_t::num_t T_max = 30.0;
    bezier_t cf(params.begin(), params.end(), constraints, T_min, T_max);

    std::cout << "Point a" << std::endl;
    std::cout << a << std::endl;
    std::cout << cf(T_min) << std::endl;
    std::cout << "Point c" << std::endl;
    std::cout << c << std::endl;
    std::cout << cf(T_max) << std::endl;
    std::cout << "Init vel" << std::endl;
    std::cout << constraints.init_vel << std::endl;
    std::cout << cf.derivate(T_min, 1) << std::endl;
    std::cout << "End vel" << std::endl;
    std::cout << constraints.end_vel << std::endl;
    std::cout << cf.derivate(T_max, 1) << std::endl;
    std::cout << "Init acc" << std::endl;
    std::cout << constraints.init_acc << std::endl;
    std::cout << cf.derivate(T_min, 2) << std::endl;
    std::cout << "End acc" << std::endl;
    std::cout << constraints.end_acc << std::endl;
    std::cout << cf.derivate(T_max, 2) << std::endl;

    /*// Create waypoints
    // curves::helpers::T_Waypoint waypoints;
    //typedef std::pair<double, Eigen::Vector3d> Waypoint;
    typedef std::vector<point3_t> myWaypoint;
    myWaypoint waypoints;
    waypoints.push_back(std::make_pair(0., Eigen::Vector3d(0,0,0)));
    waypoints.push_back(std::make_pair(1., Eigen::Vector3d(0.5,0.5,0.5)));
    waypoints.push_back(std::make_pair(2., Eigen::Vector3d(0,1,0)));

    // Create constraints
    curves::exact_cubic3_t::spline_constraints constraints(3);
    constraints.init_vel = point3_t(0, 0, 0);
    constraints.init_acc = point3_t(-1, -2, 0);
    constraints.end_vel = point3_t(-10, -10, 0);
    constraints.end_acc = point3_t(-20, -20, 0);

    // Create spline with constraints
    curves::exact_cubic3_t end_eff(waypoints.begin(), waypoints.end(), constraints);*/

    std::cout << cf(15.0)[2] << std::endl;

    const float N_test = 100.0;
    for (float i = 0; i < (N_test + 1); i++) {
      if (std::abs(i - 60) < 0.1) {
        params[2] = point3_t(3, 4, 1);
        cf = bezier_t(params.begin(), params.end(), constraints, T_min, T_max);
      }

      std::cout << (cf(i * 30.0 / N_test))[2] << ", ";
    }
    std::cout << std::endl;

    /*// Constraints
    bezier_t::curve_constraints_t pc_constraints0(3);
    pc_constraints0.init_vel = point3_t(-1, -1, 0);
    pc_constraints0.init_acc = point3_t(-1, -2, 0);
    pc_constraints0.end_vel = point3_t(-10, -10, 0);
    // pc_constraints0.end_acc = point3_t(-20, -20, 0);
    bezier_t::curve_constraints_t pc_constraints1(3);
    pc_constraints1.init_vel = point3_t(-1, -1, 0);
    //pc_constraints0.init_acc = point3_t(-1, -2, 0);
    pc_constraints1.end_vel = point3_t(-10, -10, 0);
    pc_constraints1.end_acc = point3_t(-20, -20, 0);

    // Piecewises
    point3_t a0(0.0, 0.0, 0.0);
    point3_t b0(0.5, 0.5, 1.0);
    point3_t c0(1.0, 1.0, 0.0);
    std::vector<point3_t> params0;
    std::vector<point3_t> params1;
    params0.push_back(a0);  // bezier between [0,1]
    params0.push_back(b0);
    params1.push_back(b0);  // bezier between [1,2]
    params1.push_back(c0);
    boost::shared_ptr<bezier_t> bc0_ptr(new bezier_t(params0.begin(), params0.end(), pc_constraints0, 0., 1.));
    boost::shared_ptr<bezier_t> bc1_ptr(new bezier_t(params1.begin(), params1.end(), pc_constraints1, 1., 2.));
    piecewise_t pc_C0(bc0_ptr);
    pc_C0.add_curve_ptr(bc1_ptr);

    for (float i=0; i<(N_test+1); i++) {
      std::cout << (pc_C0(i * 2.0 / N_test))[2] << ", ";
    }*/

    /*ComparePoints(a, cf(T_min), errMsg0, error);
    ComparePoints(c, cf(T_max), errMsg0, error);
    ComparePoints(constraints.init_vel, cf.derivate(T_min, 1), errMsg0, error);
    ComparePoints(constraints.end_vel, cf.derivate(T_max, 1), errMsg0, error);
    ComparePoints(constraints.init_acc, cf.derivate(T_min, 2), errMsg0, error);
    ComparePoints(constraints.end_vel, cf.derivate(T_max, 1), errMsg0, error);
    ComparePoints(constraints.end_acc, cf.derivate(T_max, 2), errMsg0, error);*/

    return EXIT_SUCCESS;
  } else {
    std::cerr << "This program needs 2 integers" << std::endl;
    return EXIT_FAILURE;
  }
}
