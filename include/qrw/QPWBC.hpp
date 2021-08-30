///////////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief This is the header for QPWBC and WbcWrapper classes
///
/// \details WbcWrapper provides an interface for the user to solve the whole body control problem
///          Internally it calls first the InvKin class to solve an inverse kinematics problem then calls the QPWBC
///          class to solve a box QP problem based on result from the inverse kinematic and desired ground forces
///
//////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef QPWBC_H_INCLUDED
#define QPWBC_H_INCLUDED

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <limits>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "osqp.h"
#include "qrw/InvKin.hpp"
#include "qrw/Params.hpp"
#include "other/st_to_cc.hpp"

// For wrapper
#include "pinocchio/multibody/model.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/compute-all-terms.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/crba.hpp"

class QPWBC {
 public:
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Constructor
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  QPWBC();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Initialize with given data
  ///
  /// \param[in] params Object that stores parameters
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void initialize(Params &params);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Destructor
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ~QPWBC() {}  // Empty destructor

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Run one iteration of the whole WBC QP problem by calling all the necessary functions (data retrieval,
  ///        update of constraint matrices, update of the solver, running the solver, retrieving result)
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  int run(const Eigen::MatrixXd &M, const Eigen::MatrixXd &Jc, const Eigen::MatrixXd &f_cmd,
          const Eigen::MatrixXd &RNEA, const Eigen::MatrixXd &k_contact);

  // Getters
  Eigen::MatrixXd get_f_res();    // Return the f_res matrix
  Eigen::MatrixXd get_ddq_res();  // Return the ddq_res matrix
  Eigen::MatrixXd get_H();        // Return the H matrix

 private:
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Print positions and value of coefficients in a csc matrix
  ///
  /// \param[in] M (csc*): pointer to the csc matrix you want to print
  /// \param[in] name (char*): name that should be displayed for the matrix (one char)
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void my_print_csc_matrix(csc *M, const char *name);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Save positions and value of coefficients of a csc matrix in a csc file
  ///
  /// \param[in] M (csc*): pointer to the csc matrix you want to save
  /// \param[in] filename (string): name of the generated csv file
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void save_csc_matrix(csc *M, std::string filename);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Save positions and value of coefficients of a dense matrix in a csc file
  ///
  /// \param[in] M (double*): pointer to the dense matrix you want to save
  /// \param[in] size (int): size of the dense matrix
  /// \param[in] filename (string): name of the generated csv file
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void save_dns_matrix(double *M, int size, std::string filename);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Add a new non-zero coefficient to the sparse ML matrix by filling the triplet r_ML / c_ML / v_ML
  ///
  /// \param[in] i Row index of the new entry
  /// \param[in] j Column index of the new entry
  /// \param[in] v Value of the new entry
  /// \param[in] r_ML Pointer to the table that contains row indexes
  /// \param[in] c_ML Pointer to the table that contains column indexes
  /// \param[in] v_ML Pointer to the table that contains values
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  inline void add_to_ML(int i, int j, double v, int *r_ML, int *c_ML, double *v_ML);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Add a new non-zero coefficient to the P matrix by filling the triplet r_P / c_P / v_P
  ///
  /// \param[in] i Row index of the new entry
  /// \param[in] j Column index of the new entry
  /// \param[in] v Value of the new entry
  /// \param[in] r_P Pointer to the table that contains row indexes
  /// \param[in] c_P Pointer to the table that contains column indexes
  /// \param[in] v_P Pointer to the table that contains values
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  inline void add_to_P(int i, int j, double v, int *r_P, int *c_P, double *v_P);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Create the constraint matrices (M.X = N and L.X <= K)
  ///        Create the weight matrices P and Q (cost 1/2 x^T * P * X + X^T * Q)
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  int create_matrices();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Create the M and L matrices involved in the constraint equations
  ///        the solution has to respect: M.X = N and L.X <= K
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  int create_ML();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Create the weight matrices P and Q in the cost function
  ///        1/2 x^T.P.x + x^T.q of the QP problem
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  int create_weight_matrices();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Compute all matrices of the Box QP problem
  ///
  /// \param[in] M joint space inertia matrix computed with crba
  /// \param[in] Jc Jacobian of contact points
  /// \param[in] f_cmd reference contact forces coming from the MPC
  /// \param[in] RNEA joint torques according to the current state of the system and the desired joint accelerations
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void compute_matrices(const Eigen::MatrixXd &M, const Eigen::MatrixXd &Jc, const Eigen::MatrixXd &f_cmd,
                        const Eigen::MatrixXd &RNEA);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Update P and Q matrices in the cost function xT P x + 2 xT Q
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void update_PQ();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Initialize the solver (first iteration) or update it (next iterations) then call the OSQP solver to solve
  ///        the QP problem
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  int call_solver();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Extract relevant information from the output of the QP solver
  ///
  /// \param[in] f_cmd Reference contact forces received from the MPC
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  int retrieve_result(const Eigen::MatrixXd &f_cmd);

  Params *params_;  // Object that stores parameters

  int cpt_ML = 0;  // Counter of ML coefficients
  int cpt_P = 0;   // Counter of P coefficients

  // Set to True after the creation of the QP problem during the first call of the solver
  bool initialized = false;

  // Weight matrices of initial QP
  Eigen::Matrix<double, 6, 6> Q1 = Eigen::Matrix<double, 6, 6>::Identity();
  Eigen::Matrix<double, 12, 12> Q2 = Eigen::Matrix<double, 12, 12>::Identity();

  // Friction coefficient
  const double mu = 0.9;

  // Generatrix of the linearized friction cone
  Eigen::Matrix<double, 20, 12> G = Eigen::Matrix<double, 20, 12>::Zero();

  // Transformation matrices
  Eigen::Matrix<double, 6, 6> Y = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 12> X = Eigen::Matrix<double, 6, 12>::Zero();
  Eigen::Matrix<double, 6, 6> Yinv = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 12> A = Eigen::Matrix<double, 6, 12>::Zero();
  Eigen::Matrix<double, 6, 1> gamma = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 12, 12> H = Eigen::Matrix<double, 12, 12>::Zero();
  Eigen::Matrix<double, 12, 1> g = Eigen::Matrix<double, 12, 1>::Zero();

  // Results
  // Eigen::Matrix<double, 12, 1> lambdas = Eigen::Matrix<double, 12, 1>::Zero();
  Eigen::MatrixXd f_res = Eigen::MatrixXd::Zero(12, 1);
  Eigen::MatrixXd ddq_res = Eigen::MatrixXd::Zero(12, 1);

  // Matrix ML
  const static int size_nz_ML = 20 * 12;  // 4 * (4 * 2 + 1);
  csc *ML;                                // Compressed Sparse Column matrix

  // Matrix NK
  const static int size_nz_NK = 20;
  double v_NK_up[size_nz_NK] = {};   // matrix NK (upper bound)
  double v_NK_low[size_nz_NK] = {};  // matrix NK (lower bound)
  double v_warmxf[size_nz_NK] = {};  // matrix NK (lower bound)

  // Matrix P
  const static int size_nz_P = 6 * 13;  // 6*13; // 12*13/2;
  csc *P;                               // Compressed Sparse Column matrix

  // Matrix Q
  const static int size_nz_Q = 12;
  double Q[size_nz_Q] = {};  // Q is full of zeros

  // OSQP solver variables
  OSQPWorkspace *workspce = new OSQPWorkspace();
  OSQPData *data;
  OSQPSettings *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));
};

class WbcWrapper {
 public:
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Empty constructor
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  WbcWrapper();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Destructor.
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ~WbcWrapper() {}

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Initializer
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void initialize(Params &params);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Run and solve one iteration of the whole body control (matrix update, invkin, QP)
  ///
  /// \param[in] q Estimated positions of the 12 actuators
  /// \param[in] dq Estimated velocities of the 12 actuators
  /// \param[in] f_cmd Reference contact forces received from the MPC
  /// \param[in] contacts Contact status of the four feet
  /// \param[in] pgoals Desired positions of the four feet in base frame
  /// \param[in] vgoals Desired velocities of the four feet in base frame
  /// \param[in] agoals Desired accelerations of the four feet in base frame
  /// \param[in] q_mpc Estimated configuration vector given to the MPC
  /// \param[in] v_mpc Estimated velocity vector given to the MPC
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void compute(VectorN const &q, VectorN const &dq, VectorN const &f_cmd, MatrixN const &contacts,
               MatrixN const &pgoals, MatrixN const &vgoals, MatrixN const &agoals, VectorN const &q_mpc,
               VectorN const &v_mpc);

  VectorN get_qdes() { return qdes_; }
  VectorN get_vdes() { return vdes_; }
  VectorN get_tau_ff() { return tau_ff_; }
  VectorN get_ddq_cmd() { return ddq_cmd_; }
  VectorN get_f_with_delta() { return f_with_delta_; }
  VectorN get_ddq_with_delta() { return ddq_with_delta_; }
  MatrixN get_feet_pos() { return invkin_->get_posf().transpose(); }
  MatrixN get_feet_err() { return log_feet_pos_target - invkin_->get_posf().transpose(); }
  MatrixN get_feet_vel() { return invkin_->get_vf().transpose(); }
  MatrixN get_feet_pos_target() { return log_feet_pos_target; }
  MatrixN get_feet_vel_target() { return log_feet_vel_target; }
  MatrixN get_feet_acc_target() { return log_feet_acc_target; }

 private:
  Params *params_;  // Object that stores parameters
  QPWBC *box_qp_;   // QP problem solver for the whole body control
  InvKin *invkin_;  // Inverse Kinematics solver for the whole body control

  pinocchio::Model model_;  // Pinocchio model for frame computations
  pinocchio::Data data_;    // Pinocchio datas for frame computations

  Eigen::Matrix<double, 18, 18> M_;  // Mass matrix
  Eigen::Matrix<double, 12, 6> Jc_;  // Jacobian matrix
  Matrix14 k_since_contact_;         // Number of time step during which feet have been in the current stance phase
  Vector12 qdes_;                    // Desired actuator positions
  Vector12 vdes_;                    // Desired actuator velocities
  Vector12 tau_ff_;                  // Desired actuator torques (feedforward)

  Vector18 ddq_cmd_;         // Actuator accelerations computed by Inverse Kinematics
  Vector19 q_default_;       // Default configuration vector to compute the mass matrix
  Vector12 f_with_delta_;    // Contact forces with deltas found by QP solver
  Vector18 ddq_with_delta_;  // Actuator accelerations with deltas found by QP solver

  Matrix43 posf_tmp_;  // Temporary matrix to store posf_ from invkin_

  Matrix34 log_feet_pos_target;  // Store the target feet positions
  Matrix34 log_feet_vel_target;  // Store the target feet velocities
  Matrix34 log_feet_acc_target;  // Store the target feet accelerations

  int k_log_;                          // Counter for logging purpose
  int indexes_[4] = {10, 18, 26, 34};  // Indexes of feet frames in this order: [FL, FR, HL, HR]
};

#endif  // QPWBC_H_INCLUDED
