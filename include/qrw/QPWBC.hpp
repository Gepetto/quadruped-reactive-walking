#ifndef QPWBC_H_INCLUDED
#define QPWBC_H_INCLUDED

#include "qrw/InvKin.hpp" // For pseudoinverse
#include "qrw/Params.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <limits>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "osqp.h"
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
 private:
  
  Params* params_;  // Object that stores parameters

  int cpt_ML = 0;
  int cpt_P = 0;

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
  Eigen::Matrix<double, 12, 12>  H = Eigen::Matrix<double, 12, 12>::Zero();
  Eigen::Matrix<double, 12, 1> g = Eigen::Matrix<double, 12, 1>::Zero();

  // Results
  // Eigen::Matrix<double, 12, 1> lambdas = Eigen::Matrix<double, 12, 1>::Zero();
  Eigen::MatrixXd f_res = Eigen::MatrixXd::Zero(12, 1);
  Eigen::MatrixXd ddq_res = Eigen::MatrixXd::Zero(12, 1);
  
  // Matrix ML
  const static int size_nz_ML = 20*12; //4 * (4 * 2 + 1);
  csc *ML;  // Compressed Sparse Column matrix

  // Matrix NK
  const static int size_nz_NK = 20;
  double v_NK_up[size_nz_NK] = {};   // matrix NK (upper bound)
  double v_NK_low[size_nz_NK] = {};  // matrix NK (lower bound)
  double v_warmxf[size_nz_NK] = {};  // matrix NK (lower bound)

  // Matrix P
  const static int size_nz_P = 6*13; // 6*13; // 12*13/2;
  csc *P;  // Compressed Sparse Column matrix

  // Matrix Q
  const static int size_nz_Q = 12;
  double Q[size_nz_Q] = {};  // Q is full of zeros

  // OSQP solver variables
  OSQPWorkspace *workspce = new OSQPWorkspace();
  OSQPData *data;
  OSQPSettings *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));

 public:
  
  QPWBC(); // Constructor
  void initialize(Params& params);

  // Functions
  inline void add_to_ML(int i, int j, double v, int *r_ML, int *c_ML, double *v_ML); // function to fill the triplet r/c/v
  inline void add_to_P(int i, int j, double v, int *r_P, int *c_P, double *v_P); // function to fill the triplet r/c/v
  int create_matrices();
  int create_ML();
  int create_weight_matrices();
  void compute_matrices(const Eigen::MatrixXd &M, const Eigen::MatrixXd &Jc, const Eigen::MatrixXd &f_cmd, const Eigen::MatrixXd &RNEA);
  void update_PQ();
  int call_solver();
  int retrieve_result(const Eigen::MatrixXd &f_cmd);
  int run(const Eigen::MatrixXd &M, const Eigen::MatrixXd &Jc, const Eigen::MatrixXd &f_cmd, const Eigen::MatrixXd &RNEA, const Eigen::MatrixXd &k_contact);

  // Getters
  Eigen::MatrixXd get_f_res();
  Eigen::MatrixXd get_ddq_res();
  Eigen::MatrixXd get_H();

  // Utils
  void my_print_csc_matrix(csc *M, const char *name);
  void save_csc_matrix(csc *M, std::string filename);
  void save_dns_matrix(double *M, int size, std::string filename);

};

class WbcWrapper
{
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
    void initialize(Params& params);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Compute the remaining and total duration of a swing phase or a stance phase based
    ///        on the content of the gait matrix
    ///
    /// \param[in] i considered phase (row of the gait matrix)
    /// \param[in] j considered foot (col of the gait matrix)
    /// \param[in] value 0.0 for swing phase detection, 1.0 for stance phase detection
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    void compute(VectorN const& q, VectorN const& dq, MatrixN const& f_cmd, MatrixN const& contacts,
                 MatrixN const& pgoals, MatrixN const& vgoals, MatrixN const& agoals);

    VectorN get_qdes() { return qdes_; }
    VectorN get_vdes() { return vdes_; }
    VectorN get_tau_ff() { return tau_ff_; }
    VectorN get_f_with_delta() { return f_with_delta_; }
    MatrixN get_feet_pos() { return MatrixN::Zero(3, 4); }
    MatrixN get_feet_err() { return MatrixN::Zero(3, 4); }
    MatrixN get_feet_vel() { return MatrixN::Zero(3, 4); }
    MatrixN get_feet_pos_target() { return MatrixN::Zero(3, 4); }
    MatrixN get_feet_vel_target() { return MatrixN::Zero(3, 4); }
    MatrixN get_feet_acc_target() { return MatrixN::Zero(3, 4); }

private:
    
    Params* params_;  // Object that stores parameters
    QPWBC* box_qp_;  // QP problem solver for the whole body control
    InvKin* invkin_;  // Inverse Kinematics solver for the whole body control

    pinocchio::Model model_;  // Pinocchio model for frame computations
    pinocchio::Data data_;  // Pinocchio datas for frame computations

    Eigen::Matrix<double, 18, 18> M_;  // Mass matrix
    Eigen::Matrix<double, 12, 6> Jc_;  // Jacobian matrix
    Eigen::Matrix<double, 1, 4> k_since_contact_;
    Vector12 qdes_;  // Desired actuator positions
    Vector12 vdes_;  // Desired actuator velocities
    Vector12 tau_ff_;  // Desired actuator torques (feedforward)

    Vector18 ddq_cmd_;  // Actuator accelerations computed by Inverse Kinematics
    Vector19 q_default_;  // Default configuration vector to compute the mass matrix
    Vector12 f_with_delta_;  // Contact forces with deltas found by QP solver
    Vector18 ddq_with_delta_;  // Actuator accelerations with deltas found by QP solver

    Matrix43 posf_tmp_;  // Temporary matrix to store posf_ from invkin_

    int k_log_;  // Counter for logging purpose
    int indexes_[4] = {10, 18, 26, 34};  // Indexes of feet frames in this order: [FL, FR, HL, HR]
};

#endif  // QPWBC_H_INCLUDED
