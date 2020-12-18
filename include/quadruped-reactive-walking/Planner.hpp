#ifndef PLANNER_H_INCLUDED
#define PLANNER_H_INCLUDED

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "pinocchio/math/rpy.hpp"

#define N0_gait 20
// Number of rows in the gait matrix. Arbitrary value that should be set high enough so that there is always at
// least one empty line at the end of the gait matrix

typedef Eigen::MatrixXd matXd;

class TrajGen {
  /* Class that generates a reference trajectory in position, velocity and acceleration that feet it swing phase
     should follow */

 private:
  double lastCoeffs_x[6];  // Coefficients for the X component
  double lastCoeffs_y[6];  // Coefficients for the Y component
  double h = 0.05;  // Apex height of the swinging trajectory
  double time_adaptative_disabled = 0.2;  // Target lock before the touchdown
  double x1 = 0.0;  // Target for the X component
  double y1 = 0.0;  // Target for the Y component
  Eigen::Matrix<double, 11, 1> result = Eigen::Matrix<double, 11, 1>::Zero();  // Output of the generator

  // Coefficients
  double Ax5 = 0.0, Ax4 = 0.0, Ax3 = 0.0, Ax2 = 0.0, Ax1 = 0.0, Ax0 = 0.0, Ay5 = 0.0, Ay4 = 0.0, Ay3 = 0.0, Ay2 = 0.0,
         Ay1 = 0.0, Ay0 = 0.0, Az6 = 0.0, Az5 = 0.0, Az4 = 0.0, Az3 = 0.0;

 public:
  TrajGen();  // Empty constructor
  TrajGen(double h_in, double t_lock_in, double x_in, double y_in);  // Default constructor
  Eigen::Matrix<double, 11, 1> get_next_foot(double x0, double dx0, double ddx0, double y0, double dy0, double ddy0,
                                             double x1_in, double y1_in, double t0, double t1, double dt);
};

class Planner {
  /* Planner that outputs current and future locations of footsteps, the reference trajectory of the base and
     the position, velocity, acceleration commands for feet in swing phase based on the reference velocity given by
     the user and the current position/velocity of the base */
    
 private:
  // Inputs of the constructor
  double dt;       // Time step of the contact sequence (time step of the MPC)
  double dt_tsid;  // Time step of TSID
  double T_gait;   // Gait period
  double T_mpc;    // MPC period (prediction horizon)
  double h_ref;    // Reference height for the trunk
  int k_mpc;       // Number of TSID iterations for one iteration of the MPC
  bool on_solo8;   //  Whether we are working on solo8 or not

  // Predefined quantities
  double k_feedback = 0.03;  // Feedback gain for the feedback term of the planner
  double g = 9.81;           // Value of the gravity acceleartion
  double L = 0.155;          // Value of the maximum allowed deviation due to leg length
  bool is_static = false;    // Flag for static gait

  // Number of time steps in the prediction horizon
  int n_steps; // T_mpc / time step of the MPC

  // Feet index vector
  std::vector<int> feet;
  std::vector<double> t0s;
  double t_remaining = 0.0;
  double t_swing[4] = {0.0, 0.0, 0.0, 0.0};

  // Constant sized matrices
  Eigen::MatrixXd fsteps = Eigen::MatrixXd::Zero(N0_gait, 13);
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
  Eigen::MatrixXd gait_f = Eigen::MatrixXd::Zero(N0_gait, 5);                                // Current and future gait
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

  // Variables for foot trajectory generator
  int i_end_gait = 0;
  Eigen::Matrix<double, 1, 4> t_stance =
      Eigen::Matrix<double, 1, 4>::Zero();  // Total duration of current stance phase for each foot
  Eigen::Matrix<double, 2, 4> footsteps_target = Eigen::Matrix<double, 2, 4>::Zero();
  Eigen::MatrixXd goals = Eigen::MatrixXd::Zero(3, 4);   // Store 3D target position for feet
  Eigen::MatrixXd vgoals = Eigen::MatrixXd::Zero(3, 4);  // Store 3D target velocity for feet
  Eigen::MatrixXd agoals = Eigen::MatrixXd::Zero(3, 4);  // Store 3D target acceleration for feet
  Eigen::Matrix<double, 6, 4> mgoals =
      Eigen::Matrix<double, 6, 4>::Zero();  // Storage variable for the trajectory generator

  Eigen::Matrix<double, 11, 4> res_gen = Eigen::Matrix<double, 11, 4>::Zero(); // Result of the generator

 public:
  Planner();
  Planner(double dt_in, double dt_tsid_in, double T_gait_in, double T_mpc_in, int k_mpc_in, bool on_solo8_in,
          double h_ref_in, const Eigen::MatrixXd &fsteps_in);

  void Print();

  int create_walk();
  int create_trot();
  int create_pacing();
  int create_bounding();
  int create_static();
  int create_gait_f();
  int roll(int k);
  int handle_joystick(int code, const Eigen::MatrixXd &q);
  int compute_footsteps(Eigen::MatrixXd q_cur, Eigen::MatrixXd v_cur, Eigen::MatrixXd v_ref);
  double get_stance_swing_duration(int i, int j, double value);
  int compute_next_footstep(int i, int j);
  int getRefStates(Eigen::MatrixXd q, Eigen::MatrixXd v, Eigen::MatrixXd vref, double z_average);
  int update_target_footsteps();
  int update_trajectory_generator(int k, double h_estim);
  int run_planner(int k, const Eigen::MatrixXd &q, const Eigen::MatrixXd &v, const Eigen::MatrixXd &b_vref,
                  double h_estim, double z_average, int joystick_code);

  // Accessors (to retrieve C data from Python)
  Eigen::MatrixXd get_xref();
  Eigen::MatrixXd get_fsteps();
  Eigen::MatrixXd get_gait();
  Eigen::MatrixXd get_goals();
  Eigen::MatrixXd get_vgoals();
  Eigen::MatrixXd get_agoals();
};

#endif  // PLANNER_H_INCLUDED
