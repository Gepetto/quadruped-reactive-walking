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

#include "osqp.h"
#include "other/st_to_cc.hpp"
#include "qrw/Params.hpp"

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
  /// \param[in] M Generalized mass matrix
  /// \param[in] Jc Contact Jacobian
  /// \param[in] ddq_cmd Command base accelerations
  /// \param[in] f_cmd Desired contact forces of the MPC
  /// \param[in] RNEA M(q) ddq + NLE(q, dq) computed by rnea(q_wbc_, dq_wbc_, ddq_cmd_)
  /// \param[in] k_contact Number of time step during which feet have been in the current stance phase
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  int run(const MatrixN &M, const MatrixN &Jc, const MatrixN &ddq_cmd, const MatrixN &f_cmd, const MatrixN &RNEA,
          const MatrixN &k_contact);

  // Getters
  MatrixN get_f_res();    // Return the f_res matrix
  MatrixN get_ddq_res();  // Return the ddq_res matrix
  MatrixN get_H();        // Return the H matrix

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
  int create_matrices(const Eigen::Matrix<double, 12, 6> &Jc, const Eigen::Matrix<double, 12, 1> &f_cmd,
                      const Eigen::Matrix<double, 6, 1> &RNEA);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Create the M and L matrices involved in the constraint equations
  ///        the solution has to respect: M.X = N and L.X <= K
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  int create_ML();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Create the N and K matrices involved in the constraint equations
  ///        the solution has to respect: M.X = N and L.X <= K
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  int create_NK(const Eigen::Matrix<double, 6, 12> &JcT, const Eigen::Matrix<double, 12, 1> &f_cmd,
                const Eigen::Matrix<double, 6, 1> &RNEA);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Create the weight matrices P and Q in the cost function
  ///        1/2 x^T.P.x + x^T.q of the QP problem
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  int create_weight_matrices();

  int update_matrices(const Eigen::Matrix<double, 6, 6> &M, const Eigen::Matrix<double, 12, 6> &Jc,
                      const Eigen::Matrix<double, 12, 1> &f_cmd, const Eigen::Matrix<double, 6, 1> &RNEA);

  int update_ML(const Eigen::Matrix<double, 6, 6> &M, const Eigen::Matrix<double, 6, 12> &JcT);

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
  void compute_matrices(const MatrixN &M, const MatrixN &Jc, const MatrixN &f_cmd, const MatrixN &RNEA);

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
  int retrieve_result(const Eigen::Matrix<double, 6, 1> &ddq_cmd, const Eigen::Matrix<double, 12, 1> &f_cmd);

  Params *params_;  // Object that stores parameters

  int cpt_ML = 0;  // Counter of ML coefficients
  int cpt_P = 0;   // Counter of P coefficients

  // Set to True after the creation of the QP problem during the first call of the solver
  bool initialized = false;

  // Weight matrices of initial QP
  Eigen::Matrix<double, 6, 6> Q1 = Eigen::Matrix<double, 6, 6>::Identity();
  Eigen::Matrix<double, 12, 12> Q2 = Eigen::Matrix<double, 12, 12>::Identity();

  // Friction coefficient
  const double mu = 0.5;

  // Minimum and maximum normal contact force
  double Fz_max = 0.0;
  double Fz_min = 0.0;

  // Cumulative non zero coefficients per column in friction cone constraint block
  // In each column: 2, 2, 5, 2, 2, 5, 2, 2, 5, 2, 2, 5
  int fric_nz[12] = {2, 4, 9, 11, 13, 18, 20, 22, 27, 29, 31, 36};

  // Generalized mass matrix and contact Jacobian (transposed)
  // Eigen::Matrix<double, 6, 6> M = Eigen::Matrix<double, 6, 6>::Zero();
  // Eigen::Matrix<double, 6, 12> JcT = Eigen::Matrix<double, 6, 12>::Zero();

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
  MatrixN f_res = MatrixN::Zero(12, 1);
  MatrixN ddq_res = MatrixN::Zero(6, 1);

  // Matrix ML
  const static int size_nz_ML = (20 + 6) * 18;
  csc *ML;  // Compressed Sparse Column matrix

  // Matrix NK
  const static int size_nz_NK = (20 + 6);
  double v_NK_up[size_nz_NK] = {};   // matrix NK (upper bound)
  double v_NK_low[size_nz_NK] = {};  // matrix NK (lower bound)
  double v_warmxf[size_nz_NK] = {};  // matrix NK (lower bound)

  // Matrix P
  const static int size_nz_P = 18;
  csc *P;  // Compressed Sparse Column matrix

  // Matrix Q
  const static int size_nz_Q = 18;
  double Q[size_nz_Q] = {};  // Q is full of zeros

  // OSQP solver variables
  OSQPWorkspace *workspce = new OSQPWorkspace();
  OSQPData *data;
  OSQPSettings *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));
};

#endif  // QPWBC_H_INCLUDED
