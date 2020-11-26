#ifndef PLANNER_H_INCLUDED
#define PLANNER_H_INCLUDED

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "pinocchio/math/rpy.hpp"
#include "curves/fwd.h"
#include "curves/bezier_curve.h"
// #include "curves/helpers/effector_spline.h"

typedef Eigen::MatrixXd matXd;

const int N0_gait = 20;

class TrajGen {
 private:
  double lastCoeffs_x[6];
  double lastCoeffs_y[6];
  double h = 0.05;
  double time_adaptative_disabled = 0.2;
  double x1 = 0.0;
  double y1 = 0.0;
  Eigen::Matrix<double, 11, 1> result = Eigen::Matrix<double, 11, 1>::Zero();

  // Coefficients
  double Ax5 = 0.0, Ax4 = 0.0, Ax3 = 0.0, Ax2 = 0.0, Ax1 = 0.0, Ax0 = 0.0, Ay5 = 0.0, Ay4 = 0.0, Ay3 = 0.0, Ay2 = 0.0,
         Ay1 = 0.0, Ay0 = 0.0, Az6 = 0.0, Az5 = 0.0, Az4 = 0.0, Az3 = 0.0;

 public:
  TrajGen();
  TrajGen(double h_in, double t_lock_in, double x_in, double y_in);
  Eigen::Matrix<double, 11, 1> get_next_foot(double x0, double dx0, double ddx0, double y0, double dy0, double ddy0,
                                             double x1_in, double y1_in, double t0, double t1, double dt);
};

class Planner {
 private:
  // Inputs of the constructor
  double dt;       // Time step of the contact sequence
  double dt_tsid;  // Time step of TSID
  double T_gait;   // Gait period
  double h_ref;    // Reference height for the trunk
  int n_periods;   // Number of gait periods
  int k_mpc;       // Number of TSID iterations for one iteration of the MPC
  bool on_solo8;   //  Whether we are working on solo8 or not

  // Predefined quantities
  double k_feedback = 0.03;  // Feedback gain for the feedback term of the planner
  double g = 9.81;           // Value of the gravity acceleartion
  double L = 0.155;          // Value of the maximum allowed deviation due to leg length
  bool is_static = false;    // Flag for static gait
  int pt_line = 0;
  int pt_sum = 0;

  // Number of time steps in the prediction horizon
  int n_steps;

  // Feet index vector
  std::vector<int> feet;
  std::vector<double> t0s;
  double t_swing[4] = {0.0, 0.0, 0.0, 0.0};

  // Constant sized matrices
  Eigen::MatrixXd gait = Eigen::MatrixXd::Zero(N0_gait, 5);
  Eigen::MatrixXd fsteps = Eigen::MatrixXd::Zero(N0_gait, 13);
  Eigen::Matrix<double, N0_gait, 5> desired_gait = Eigen::Matrix<double, N0_gait, 5>::Zero();
  Eigen::Matrix<double, N0_gait, 5> new_desired_gait = Eigen::Matrix<double, N0_gait, 5>::Zero();
  Eigen::Matrix<double, 3, 4> shoulders = Eigen::Matrix<double, 3, 4>::Zero();  // Position of shoulders in local frame
  Eigen::Matrix<double, 19, 1> q_static = Eigen::Matrix<double, 19, 1>::Zero();
  Eigen::Matrix<double, 3, 1> RPY_static = Eigen::Matrix<double, 3, 1>::Zero();
  Eigen::Matrix<double, 1, 12> o_feet_contact = Eigen::Matrix<double, 1, 12>::Zero();  // Feet matrix in world frame
  Eigen::Matrix<double, 3, 4> next_footstep =
      Eigen::Matrix<double, 3, 4>::Zero();  // To store the result of the compute_next_footstep function
  Eigen::Matrix<double, 3, 3> R =
      Eigen::Matrix<double, 3, 3>::Zero();  // Predefined matrices for compute_footstep function
  Eigen::Matrix<double, 3, 3> R_1 =
      Eigen::Matrix<double, 3, 3>::Zero();  // Predefined matrices for compute_next_footstep function
  Eigen::Matrix<double, 3, 3> R_2 = Eigen::Matrix<double, 3, 3>::Zero();
  Eigen::Matrix<double, N0_gait, 1> dt_cum = Eigen::Matrix<double, N0_gait, 1>::Zero();
  Eigen::Matrix<double, N0_gait, 1> angle = Eigen::Matrix<double, N0_gait, 1>::Zero();
  Eigen::Matrix<double, N0_gait, 1> dx = Eigen::Matrix<double, N0_gait, 1>::Zero();
  Eigen::Matrix<double, N0_gait, 1> dy = Eigen::Matrix<double, N0_gait, 1>::Zero();
  Eigen::Matrix<double, 3, 1> q_tmp = Eigen::Matrix<double, 3, 1>::Zero();
  Eigen::Matrix<double, 3, 1> q_dxdy = Eigen::Matrix<double, 3, 1>::Zero();
  Eigen::Matrix<double, 3, 1> RPY = Eigen::Matrix<double, 3, 1>::Zero();
  Eigen::Matrix<double, 3, 1> b_v_cur = Eigen::Matrix<double, 3, 1>::Zero();
  Eigen::Matrix<double, 6, 1> b_v_ref = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 3, 1> cross = Eigen::Matrix<double, 3, 1>::Zero();
  Eigen::Matrix<double, 6, 1> vref_in = Eigen::Matrix<double, 6, 1>::Zero();

  Eigen::Matrix<double, N0_gait, 5> gait_p = Eigen::Matrix<double, N0_gait, 5>::Zero();  // Past gait
  Eigen::MatrixXd gait_c = Eigen::MatrixXd::Zero(1, 5);  // Current gait (MatrixXd for bindings)
  Eigen::Matrix<double, N0_gait, 5> gait_f = Eigen::Matrix<double, N0_gait, 5>::Zero();      // Future gait
  Eigen::Matrix<double, N0_gait, 5> gait_f_des = Eigen::Matrix<double, N0_gait, 5>::Zero();  // Future desired gait

  // Time interval vector
  Eigen::Matrix<double, 1, Eigen::Dynamic> dt_vector;

  // Reference trajectory matrix of size 12 by (1 + N)  with the current state of
  // the robot in column 0 and the N steps of the prediction horizon in the others
  Eigen::MatrixXd xref;

  // Foot trajectory generator
  double max_height_feet = 0.05;  // * (1000/312.5);  // height * correction coefficient
  double t_lock_before_touchdown = 0.07;
  std::vector<TrajGen> myTrajGen;
  // TODO

  // Variables for foot trajectory generator
  int i_end_gait = 0;
  Eigen::Matrix<double, 1, 4> t_stance =
      Eigen::Matrix<double, 1, 4>::Zero();  // Total duration of current stance phase for each foot
  // Eigen::Matrix<double, 1, 4> t_swing = Eigen::Matrix<double, 1, 4>::Zero(); // Total duration of current swing
  // phase for each foot
  Eigen::Matrix<double, 2, 4> footsteps_target = Eigen::Matrix<double, 2, 4>::Zero();
  Eigen::MatrixXd goals = Eigen::MatrixXd::Zero(3, 4);   // Store 3D target position for feet
  Eigen::MatrixXd vgoals = Eigen::MatrixXd::Zero(3, 4);  // Store 3D target velocity for feet
  Eigen::MatrixXd agoals = Eigen::MatrixXd::Zero(3, 4);  // Store 3D target acceleration for feet
  Eigen::Matrix<double, 6, 4> mgoals =
      Eigen::Matrix<double, 6, 4>::Zero();  // Storage variable for the trajectory generator

  Eigen::Matrix<double, 11, 4> res_gen = Eigen::Matrix<double, 11, 4>::Zero();
  /*curves::bezier_t::curve_constraints_t constrts;
  constrts = curves::bezier_t::curve_constraints_t(3);*/

  std::vector<std::vector<curves::point3_t>> pr_feet;
  std::vector<curves::bezier_t::num_t> T_min;
  std::vector<curves::bezier_t::num_t> T_max;
  std::vector<curves::bezier_t> c_feet;
  // curves::bezier_t::curve_constraints<point3_t> constraints;
  curves::bezier_t::curve_constraints_t constraints;

 public:
  Planner();
  Planner(double dt_in, double dt_tsid_in, int n_periods_in, double T_gait_in, int k_mpc_in, bool on_solo8_in,
          double h_ref_in, const Eigen::MatrixXd &fsteps_in);

  void Print();

  int create_trot();
  int roll(int k);
  int compute_footsteps(Eigen::MatrixXd q_cur, Eigen::MatrixXd v_cur, Eigen::MatrixXd v_ref);
  int compute_next_footstep(int j);
  int getRefStates(Eigen::MatrixXd q, Eigen::MatrixXd v, Eigen::MatrixXd vref, double z_average);
  int update_target_footsteps();
  int update_trajectory_generator(int k, double h_estim);
  int run_planner(int k, const Eigen::MatrixXd &q, const Eigen::MatrixXd &v, const Eigen::MatrixXd &b_vref,
                  double h_estim, double z_average);

  int roll_exp(int k);

  // Accessor
  Eigen::MatrixXd get_xref();
  Eigen::MatrixXd get_fsteps();
  Eigen::MatrixXd get_gait();
  Eigen::MatrixXd get_goals();
  Eigen::MatrixXd get_vgoals();
  Eigen::MatrixXd get_agoals();
};

#endif  // PLANNER_H_INCLUDED
