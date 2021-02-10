#ifndef INVKIN_H_INCLUDED
#define INVKIN_H_INCLUDED

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include "pinocchio/math/rpy.hpp"
#include "pinocchio/spatial/explog.hpp"

class InvKin {
  /* Planner that outputs current and future locations of footsteps, the
     reference trajectory of the base and the position, velocity, acceleration
     commands for feet in swing phase based on the reference velocity given by
     the user and the current position/velocity of the base */

  /*private:
   // Inputs of the constructor
   double dt;       // Time step of the contact sequence (time step of the MPC)
   double dt_tsid;  // Time step of TSID
   double T_gait;   // Gait period
   double T_mpc;    // MPC period (prediction horizon)
   double h_ref;    // Reference height for the trunk
   int k_mpc;       // Number of TSID iterations for one iteration of the MPC
   bool on_solo8;   //  Whether we are working on solo8 or not

   // Predefined quantities
   double k_feedback = 0.03;  // Feedback gain for the feedback term of the
   planner double g = 9.81;           // Value of the gravity acceleartion
   double L = 0.155;          // Value of the maximum allowed deviation due to
   leg length bool is_static = false;    // Flag for static gait

   // Number of time steps in the prediction horizon
   int n_steps; // T_mpc / time step of the MPC

   // Feet index vector
   std::vector<int> feet;
   std::vector<double> t0s;
   double t_remaining = 0.0;
   double t_swing[4] = {0.0, 0.0, 0.0, 0.0};

   // Constant sized matrices
   Eigen::MatrixXd fsteps = Eigen::MatrixXd::Zero(N0_gait, 13);
   Eigen::Matrix<double, 3, 4> shoulders = Eigen::Matrix<double, 3, 4>::Zero();
   // Position of shoulders in local frame Eigen::Matrix<double, 19, 1> q_static
   = Eigen::Matrix<double, 19, 1>::Zero(); Eigen::Matrix<double, 3, 1>
   RPY_static = Eigen::Matrix<double, 3, 1>::Zero(); Eigen::Matrix<double, 1,
   12> o_feet_contact = Eigen::Matrix<double, 1, 12>::Zero();  // Feet matrix in
   world frame Eigen::Matrix<double, 3, 4> next_footstep = Eigen::Matrix<double,
   3, 4>::Zero();  // To store the result of the compute_next_footstep function
   Eigen::Matrix<double, 3, 3> R =
       Eigen::Matrix<double, 3, 3>::Zero();  // Predefined matrices for
   compute_footstep function Eigen::Matrix<double, 3, 3> R_1 =
       Eigen::Matrix<double, 3, 3>::Zero();  // Predefined matrices for
   compute_next_footstep function Eigen::Matrix<double, 3, 3> R_2 =
   Eigen::Matrix<double, 3, 3>::Zero(); Eigen::Matrix<double, N0_gait, 1> dt_cum
   = Eigen::Matrix<double, N0_gait, 1>::Zero(); Eigen::Matrix<double, N0_gait,
   1> angle = Eigen::Matrix<double, N0_gait, 1>::Zero(); Eigen::Matrix<double,
   N0_gait, 1> dx = Eigen::Matrix<double, N0_gait, 1>::Zero();
   Eigen::Matrix<double, N0_gait, 1> dy = Eigen::Matrix<double, N0_gait,
   1>::Zero(); Eigen::Matrix<double, 3, 1> q_tmp = Eigen::Matrix<double, 3,
   1>::Zero(); Eigen::Matrix<double, 3, 1> q_dxdy = Eigen::Matrix<double, 3,
   1>::Zero(); Eigen::Matrix<double, 3, 1> RPY = Eigen::Matrix<double, 3,
   1>::Zero(); Eigen::Matrix<double, 3, 1> b_v_cur = Eigen::Matrix<double, 3,
   1>::Zero(); Eigen::Matrix<double, 6, 1> b_v_ref = Eigen::Matrix<double, 6,
   1>::Zero(); Eigen::Matrix<double, 3, 1> cross = Eigen::Matrix<double, 3,
   1>::Zero(); Eigen::Matrix<double, 6, 1> vref_in = Eigen::Matrix<double, 6,
   1>::Zero();

   Eigen::Matrix<double, N0_gait, 5> gait_p = Eigen::Matrix<double, N0_gait,
   5>::Zero();  // Past gait Eigen::MatrixXd gait_f =
   Eigen::MatrixXd::Zero(N0_gait, 5);                                // Current
   and future gait Eigen::Matrix<double, N0_gait, 5> gait_f_des =
   Eigen::Matrix<double, N0_gait, 5>::Zero();  // Future desired gait

   // Time interval vector
   Eigen::Matrix<double, 1, Eigen::Dynamic> dt_vector;

   // Reference trajectory matrix of size 12 by (1 + N)  with the current state
   of
   // the robot in column 0 and the N steps of the prediction horizon in the
   others Eigen::MatrixXd xref;

   // Foot trajectory generator
   double max_height_feet = 0.05;  // * (1000/312.5);  // height * correction
   coefficient double t_lock_before_touchdown = 0.07; std::vector<TrajGen>
   myTrajGen;

   // Variables for foot trajectory generator
   int i_end_gait = 0;
   Eigen::Matrix<double, 1, 4> t_stance =
       Eigen::Matrix<double, 1, 4>::Zero();  // Total duration of current stance
   phase for each foot Eigen::Matrix<double, 2, 4> footsteps_target =
   Eigen::Matrix<double, 2, 4>::Zero(); Eigen::MatrixXd goals =
   Eigen::MatrixXd::Zero(3, 4);   // Store 3D target position for feet
   Eigen::MatrixXd vgoals = Eigen::MatrixXd::Zero(3, 4);  // Store 3D target
   velocity for feet Eigen::MatrixXd agoals = Eigen::MatrixXd::Zero(3, 4);  //
   Store 3D target acceleration for feet Eigen::Matrix<double, 6, 4> mgoals =
       Eigen::Matrix<double, 6, 4>::Zero();  // Storage variable for the
   trajectory generator

   Eigen::Matrix<double, 11, 4> res_gen = Eigen::Matrix<double, 11, 4>::Zero();
   // Result of the generator
   */

 private:
  // Inputs of the constructor
  double dt;  // Time step of the contact sequence (time step of the MPC)

  // Matrices initialisation
  Eigen::Matrix<double, 4, 3> feet_position_ref = Eigen::Matrix<double, 4, 3>::Zero();
  Eigen::Matrix<double, 4, 3> feet_velocity_ref = Eigen::Matrix<double, 4, 3>::Zero();
  Eigen::Matrix<double, 4, 3> feet_acceleration_ref = Eigen::Matrix<double, 4, 3>::Zero();
  Eigen::Matrix<double, 1, 4> flag_in_contact = Eigen::Matrix<double, 1, 4>::Zero();
  Eigen::Matrix<double, 3, 3> base_orientation_ref = Eigen::Matrix<double, 3, 3>::Zero();
  Eigen::Matrix<double, 1, 3> base_angularvelocity_ref = Eigen::Matrix<double, 1, 3>::Zero();
  Eigen::Matrix<double, 1, 3> base_angularacceleration_ref = Eigen::Matrix<double, 1, 3>::Zero();
  Eigen::Matrix<double, 1, 3> base_position_ref = Eigen::Matrix<double, 1, 3>::Zero();
  Eigen::Matrix<double, 1, 3> base_linearvelocity_ref = Eigen::Matrix<double, 1, 3>::Zero();
  Eigen::Matrix<double, 1, 3> base_linearacceleration_ref = Eigen::Matrix<double, 1, 3>::Zero();
  Eigen::Matrix<double, 6, 1> x_ref = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 1> x = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 1> dx_ref = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 1> dx = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 18, 18> J = Eigen::Matrix<double, 18, 18>::Zero();
  Eigen::Matrix<double, 18, 18> invJ = Eigen::Matrix<double, 18, 18>::Zero();
  Eigen::Matrix<double, 1, 18> acc = Eigen::Matrix<double, 1, 18>::Zero();
  Eigen::Matrix<double, 1, 18> x_err = Eigen::Matrix<double, 1, 18>::Zero();
  Eigen::Matrix<double, 1, 18> dx_r = Eigen::Matrix<double, 1, 18>::Zero();

  Eigen::Matrix<double, 4, 3> pfeet_err = Eigen::Matrix<double, 4, 3>::Zero();
  Eigen::Matrix<double, 4, 3> vfeet_ref = Eigen::Matrix<double, 4, 3>::Zero();
  Eigen::Matrix<double, 4, 3> afeet = Eigen::Matrix<double, 4, 3>::Zero();
  Eigen::Matrix<double, 1, 3> e_basispos = Eigen::Matrix<double, 1, 3>::Zero();
  Eigen::Matrix<double, 1, 3> abasis = Eigen::Matrix<double, 1, 3>::Zero();
  Eigen::Matrix<double, 1, 3> e_basisrot = Eigen::Matrix<double, 1, 3>::Zero();
  Eigen::Matrix<double, 1, 3> awbasis = Eigen::Matrix<double, 1, 3>::Zero();

  Eigen::MatrixXd ddq = Eigen::MatrixXd::Zero(18, 1);
  Eigen::MatrixXd q_step = Eigen::MatrixXd::Zero(18, 1);
  Eigen::MatrixXd dq_cmd = Eigen::MatrixXd::Zero(18, 1);

  // Gains
  double Kp_base_orientation = 100.0;
  double Kd_base_orientation = 2.0 * std::sqrt(Kp_base_orientation);
  
  double Kp_base_position = 100.0;
  double Kd_base_position = 2.0 * std::sqrt(Kp_base_position);

  double Kp_flyingfeet = 1000.0;
  double Kd_flyingfeet = 5.0 * std::sqrt(Kp_flyingfeet);

 public:
  InvKin();
  InvKin(double dt_in);
  
  Eigen::Matrix<double, 1, 3> cross3(Eigen::Matrix<double, 1, 3> left, Eigen::Matrix<double, 1, 3> right);

  Eigen::MatrixXd refreshAndCompute(const Eigen::MatrixXd &x_cmd, const Eigen::MatrixXd &contacts,
                                    const Eigen::MatrixXd &goals, const Eigen::MatrixXd &vgoals, const Eigen::MatrixXd &agoals,
                                    const Eigen::MatrixXd &posf, const Eigen::MatrixXd &vf, const Eigen::MatrixXd &wf, const Eigen::MatrixXd &af,
                                    const Eigen::MatrixXd &Jf, const Eigen::MatrixXd &posb, const Eigen::MatrixXd &rotb, const Eigen::MatrixXd &vb,
                                    const Eigen::MatrixXd &ab, const Eigen::MatrixXd &Jb);
  Eigen::MatrixXd computeInvKin(const Eigen::MatrixXd &posf, const Eigen::MatrixXd &vf, const Eigen::MatrixXd &wf, const Eigen::MatrixXd &af,
                                const Eigen::MatrixXd &Jf, const Eigen::MatrixXd &posb, const Eigen::MatrixXd &rotb, const Eigen::MatrixXd &vb, const Eigen::MatrixXd &ab,
                                const Eigen::MatrixXd &Jb);
  Eigen::MatrixXd get_q_step();
  Eigen::MatrixXd get_dq_cmd();
  /*void Print();

  int create_walk();
  int create_trot();
  int create_gait_f();
  int roll(int k);
  int compute_footsteps(Eigen::MatrixXd q_cur, Eigen::MatrixXd v_cur,
  Eigen::MatrixXd v_ref); double get_stance_swing_duration(int i, int j, double
  value); int compute_next_footstep(int i, int j); int
  getRefStates(Eigen::MatrixXd q, Eigen::MatrixXd v, Eigen::MatrixXd vref,
  double z_average); int update_target_footsteps(); int
  update_trajectory_generator(int k, double h_estim); int run_planner(int k,
  const Eigen::MatrixXd &q, const Eigen::MatrixXd &v, const Eigen::MatrixXd
  &b_vref, double h_estim, double z_average);*/

  // Accessors (to retrieve C data from Python)
  /*Eigen::MatrixXd get_xref();
  Eigen::MatrixXd get_fsteps();
  Eigen::MatrixXd get_gait();
  Eigen::MatrixXd get_goals();
  Eigen::MatrixXd get_vgoals();
  Eigen::MatrixXd get_agoals();*/
};

template<typename _Matrix_Type_>
_Matrix_Type_ pseudoInverse(const _Matrix_Type_ &a, double epsilon = std::numeric_limits<double>::epsilon())
{
	Eigen::JacobiSVD< _Matrix_Type_ > svd(a ,Eigen::ComputeThinU | Eigen::ComputeThinV);
	double tolerance = epsilon * static_cast<double>(std::max(a.cols(), a.rows())) *svd.singularValues().array().abs()(0);
	return svd.matrixV() *  (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().adjoint();
}


#endif  // INVKIN_H_INCLUDED
