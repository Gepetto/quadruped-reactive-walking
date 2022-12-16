#ifndef MPC_H_INCLUDED
#define MPC_H_INCLUDED

#include <limits>
#include <vector>

#include "osqp.h"
#include "other/st_to_cc.hpp"
#include "qrw/Params.hpp"

class MPC {
 public:
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Constructor
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  MPC();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Constructor with parameters
  ///
  /// \param[in] params Object that stores parameters
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  MPC(Params &params);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Destructor
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ~MPC() {}  // Empty destructor

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Run one iteration of the whole MPC by calling all the necessary
  /// functions (data retrieval, update
  ///        of constraint matrices, update of the solver, running the solver,
  ///        retrieving result)
  ///
  /// \param[in] num_iter Number of the current iteration of the MPC
  /// \param[in] xref_in Reference state trajectory over the prediction horizon
  /// \param[in] fsteps_in Footsteps location over the prediction horizon stored
  /// in a Nx12 matrix
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  int run(int num_iter, const MatrixN &xref_in, const MatrixN &fsteps_in);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Retrieve the value of the cost function at the end of the
  /// resolution \return the cost value
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  float retrieve_cost();

  // Getters
  MatrixN get_latest_result();  // Return the latest desired contact forces that
                                // have been computed
  MatrixNi get_gait();          // Return the gait matrix
  VectorNi get_Sgait();         // Return the S_gait matrix
  double *get_x_next();         // Return the next predicted state of the base

 private:
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Print positions and value of coefficients in a csc matrix
  ///
  /// \param[in] M (csc*): pointer to the csc matrix you want to print
  /// \param[in] name (char*): name that should be displayed for the matrix (one
  /// char)
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void my_print_csc_matrix(csc *M, const char *name);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Save positions and value of coefficients of a csc matrix in a csc
  /// file
  ///
  /// \param[in] M (csc*): pointer to the csc matrix you want to save
  /// \param[in] filename (string): name of the generated csv file
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void save_csc_matrix(csc *M, std::string filename);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Save positions and value of coefficients of a dense matrix in a csc
  /// file
  ///
  /// \param[in] M (double*): pointer to the dense matrix you want to save
  /// \param[in] size (int): size of the dense matrix
  /// \param[in] filename (string): name of the generated csv file
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void save_dns_matrix(double *M, int size, std::string filename);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Add a new non-zero coefficient to the sparse ML matrix by filling
  /// the triplet r_ML / c_ML / v_ML
  ///
  /// \param[in] i Row index of the new entry
  /// \param[in] j Column index of the new entry
  /// \param[in] v Value of the new entry
  /// \param[in] r_ML Pointer to the table that contains row indexes
  /// \param[in] c_ML Pointer to the table that contains column indexes
  /// \param[in] v_ML Pointer to the table that contains values
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  inline void add_to_ML(int i, int j, double v, int *r_ML, int *c_ML,
                        double *v_ML);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Add a new non-zero coefficient to the P matrix by filling the
  /// triplet r_P / c_P / v_P
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
  /// \brief Create the N and K matrices involved in the MPC constraint
  /// equations
  ///        the solution has to respect: M.X = N and L.X <= K
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  int create_NK();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Create the weight matrices P and Q in the cost function
  ///        1/2 x^T.P.x + x^T.q of the QP problem
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  int create_weight_matrices();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Update the M, N, L and K constraint matrices depending on what
  /// happened
  ///
  /// \param[in] fsteps Footsteps location over the prediction horizon stored in
  /// a Nx12 matrix
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  int update_matrices(MatrixN fsteps);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Update the M and L constaint matrices depending on the current
  /// state of the gait
  ///
  /// \param[in] fsteps Footsteps location over the prediction horizon stored in
  /// a Nx12 matrix
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  int update_ML(MatrixN fsteps);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Update the N and K matrices involved in the MPC constraint
  /// equations M.X = N and L.X <= K
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  int update_NK();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Initialize the solver (first iteration) or update it (next
  /// iterations) then call the
  ///        OSQP solver to solve the QP problem
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  int call_solver(int);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Extract relevant information from the output of the QP solver
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  int retrieve_result();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Returns the skew matrix of a 3 by 1 column vector
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  Matrix3 getSkew(Vector3 v);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Construct an array of size 12*N that contains information about the
  /// contact state of feet.
  ///        This matrix is used to enable/disable contact forces in the QP
  ///        problem. N is the number of time step in the prediction horizon.
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  int construct_S();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Reconstruct the gait matrix based on the fsteps matrix since only
  /// the last one is received by the MPC
  ///
  /// \param[in] fsteps Footsteps location over the prediction horizon stored in
  /// a Nx12 matrix
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  int construct_gait(MatrixN fsteps_in);

  Params *params_;  // Object that stores parameters
  double dt;        // Time step
  double mass;      // Total mass
  double mu;        // Friction coefficient
  int n_steps;      // Number of time steps in the prediction horizon
  int cpt_ML;       // Counter of ML coefficients
  int cpt_P;        // Counter of P coefficients

  Matrix3 gI;                                 // Inertia matrix
  Matrix34 footholds = Matrix34::Zero();      // Initial position of footsteps
  Vector12 footholds_tmp = Vector12::Zero();  // Updated position of footsteps
  Matrix34 lever_arms =
      Matrix34::Zero();  // Lever arms of footsteps with the center of mass
  MatrixN4i gait;        // Contact status over the prediction horizon
  MatrixN4i inv_gait;    // Inversed contact status over the prediction horizon
  Vector12 g = Vector12::Zero();  // Gravity vector
  Vector3 offset_CoM =
      Vector3::Zero();  // Offset of the CoM position compared to center of base

  Matrix12 A = Matrix12::Identity();  // Of evolution X+ = A*X + B*f + C
  Matrix12 B = Matrix12::Zero();      // Of evolution X+ = A*X + B*f + C
  Vector12 x0 = Vector12::Zero();     // Current state of the robot
  double x_next[12] =
      {};  // Next state of the robot (difference with reference state)
  MatrixN x_f_applied;  // Next predicted state of the robot + Desired contact
                        // forces to reach it

  // Matrix ML
  const static int size_nz_ML = 15000;
  csc *ML;  // Compressed Sparse Column matrix

  // Indices that are used to udpate ML
  int i_x_B[12 * 4] = {};
  int i_y_B[12 * 4] = {};
  int i_update_B[12 * 4] = {};
  // TODO FOR S ????

  // Matrix NK
  const static int size_nz_NK = 45000;
  double v_NK_up[size_nz_NK] = {};   // maxtrix NK (upper bound)
  double v_NK_low[size_nz_NK] = {};  // maxtrix NK (lower bound)
  double v_warmxf[size_nz_NK] = {};  // maxtrix NK (lower bound)

  // Matrix P
  const static int size_nz_P = 45000;
  csc *P;  // Compressed Sparse Column matrix

  // Matrix Q
  const static int size_nz_Q = 45000;
  double Q[size_nz_Q] = {};  // Q is full of zeros

  // OSQP solver variables
  OSQPWorkspace *workspce = new OSQPWorkspace();
  OSQPData *data;
  OSQPSettings *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));

  // Matrices whose size depends on the arguments sent to the constructor
  // function
  Matrix12N xref;   // Reference state trajectory over the prediction horizon
  VectorN x;        // State vector
  VectorNi S_gait;  // Matrix used to enable/disable feet in the solver
  VectorN warmxf;   // Vector to store the solver warm start
  VectorN NK_up;    // Upper constraint limit
  VectorN NK_low;   // Lower constraint limit
  // There is no M.X = N and L.X <= K, it's actually NK_low <= ML.X <= NK_up for
  // the solver
  MatrixN D;       // Matrix used to create NK matrix
  VectorNi i_off;  // Coefficient offsets to directly update the data field
  // Sparse matrix coefficents are all stored in M->x so if we know the indexes
  // we can update them directly
};

#endif  // MPC_H_INCLUDED
