#ifndef PLANNER_H_INCLUDED
#define PLANNER_H_INCLUDED

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>

typedef Eigen::MatrixXd matXd;

const int N0_gait = 20;

class Planner {
  private:

  // Inputs of the constructor
  double dt;  // Time step of the contact sequence
  double dt_tsid;  // Time step of TSID
  double T_gait;  // Gait period
  double h_ref;  // Reference height for the trunk
  int n_periods;  // Number of gait periods
  int k_mpc;  // Number of TSID iterations for one iteration of the MPC
  bool on_solo8;  //  Whether we are working on solo8 or not

  // Predefined quantities
  double k_feedback = 0.03;  // Feedback gain for the feedback term of the planner
  double g = 9.81;  // Value of the gravity acceleartion
  double L = 0.155;  // Value of the maximum allowed deviation due to leg length
  bool is_static = false; // Flag for static gait
  int pt_line = 0;
  int pt_sum = 0;

  // Number of time steps in the prediction horizon
  int n_steps;

  // Constant sized matrices
  Eigen::Matrix<double, N0_gait, 5> gait = Eigen::Matrix<double, N0_gait, 5>::Zero();
  Eigen::Matrix<double, N0_gait, 13> fsteps = Eigen::Matrix<double, N0_gait, 13>::Zero();
  Eigen::Matrix<double, N0_gait, 5> desired_gait = Eigen::Matrix<double, N0_gait, 5>::Zero();
  Eigen::Matrix<double, N0_gait, 5> new_desired_gait = Eigen::Matrix<double, N0_gait, 5>::Zero();
  Eigen::Matrix<double, 3, 4> shoulders = Eigen::Matrix<double, 3, 4>::Zero();  // Position of shoulders in local frame
  Eigen::Matrix<double, 19, 1> q_static = Eigen::Matrix<double, 19, 1>::Zero();
  Eigen::Matrix<double, 3, 1> RPY_static = Eigen::Matrix<double, 3, 1>::Zero();
  Eigen::Matrix<double, 1, 12> o_feet_contact = Eigen::Matrix<double, 1, 12>::Zero();  // Feet matrix in world frame
  Eigen::Matrix<double, 3, 4> next_footstep = Eigen::Matrix<double, 3, 4>::Zero(); // To store the result of the compute_next_footstep function
  Eigen::Matrix<double, 3, 3> R = Eigen::Matrix<double, 3, 3>::Zero(); // Predefined matrices for compute_footstep function
  Eigen::Matrix<double, N0_gait, 1> dt_cum = Eigen::Matrix<double, N0_gait, 1>::Zero();
  Eigen::Matrix<double, N0_gait, 1> angle = Eigen::Matrix<double, N0_gait, 1>::Zero();
  Eigen::Matrix<double, N0_gait, 1> dx = Eigen::Matrix<double, N0_gait, 1>::Zero();
  Eigen::Matrix<double, N0_gait, 1> dy = Eigen::Matrix<double, N0_gait, 1>::Zero();
  Eigen::Matrix<double, 3, 1> q_tmp = Eigen::Matrix<double, 3, 1>::Zero();
  Eigen::Matrix<double, 3, 1> q_dxdy = Eigen::Matrix<double, 3, 1>::Zero();
  Eigen::Matrix<double, 3, 1> RPY = Eigen::Matrix<double, 3, 1>::Zero();

  // Time interval vector
  Eigen::Matrix<double, 1, Eigen::Dynamic> dt_vector;

  // Reference trajectory matrix of size 12 by (1 + N)  with the current state of
  // the robot in column 0 and the N steps of the prediction horizon in the others
  Eigen::Matrix<double, 12, Eigen::Dynamic> xref;

  // Foot trajectory generator
  double max_height_feet = 0.05;
  double t_lock_before_touchdown = 0.07;
  // TODO

  // Variables for foot trajectory generator
  int i_end_gait = -1;
  Eigen::Matrix<double, 1, 4> t_stance = Eigen::Matrix<double, 1, 4>::Zero(); // Total duration of current stance phase for each foot
  Eigen::Matrix<double, 1, 4> t_swing = Eigen::Matrix<double, 1, 4>::Zero(); // Total duration of current swing phase for each foot
  Eigen::Matrix<double, 2, 4> footsteps_target = Eigen::Matrix<double, 2, 4>::Zero();
  Eigen::Matrix<double, 3, 4> goals = Eigen::Matrix<double, 3, 4>::Zero(); // Store 3D target position for feet
  Eigen::Matrix<double, 3, 4> vgoals = Eigen::Matrix<double, 3, 4>::Zero();  // Store 3D target velocity for feet
  Eigen::Matrix<double, 3, 4> agoals = Eigen::Matrix<double, 3, 4>::Zero();  // Store 3D target acceleration for feet
  Eigen::Matrix<double, 6, 4> mgoals = Eigen::Matrix<double, 6, 4>::Zero();  // Storage variable for the trajectory generator

 public:
  Planner();
  Planner(double dt_in, double dt_tsid_in, int n_periods_in, double T_gait_in,
          int k_mpc_in, bool on_solo8_in, double h_ref_in, const Eigen::MatrixXd & fsteps_in);

  void Print();
  
  int create_trot();
  int roll(int k);
  int compute_footsteps(Eigen::Matrix<double, 7, 1> q_cur, Eigen::Matrix<double, 6, 1> v_cur, Eigen::Matrix<double, 7, 1> v_ref);

};

#endif  // PLANNER_H_INCLUDED
