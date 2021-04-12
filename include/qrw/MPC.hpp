#ifndef MPC_H_INCLUDED
#define MPC_H_INCLUDED

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <limits>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "osqp_folder/include/osqp.h"
#include "osqp_folder/include/cs.h"
#include "osqp_folder/include/auxil.h"
#include "osqp_folder/include/util.h"
#include "osqp_folder/include/osqp_configure.h"
#include "other/st_to_cc.hpp"

typedef Eigen::MatrixXd matXd;

class MPC {
 private:
  double dt, mass, mu, T_gait, h_ref;
  int n_steps, cpt_ML, cpt_P;

  Eigen::Matrix<double, 3, 3> gI;
  Eigen::Matrix<double, 6, 1> q;
  Eigen::Matrix<double, 6, 1> v = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 3, 4> footholds = Eigen::Matrix<double, 3, 4>::Zero();
  Eigen::Matrix<double, 1, 12> footholds_tmp = Eigen::Matrix<double, 12, 1>::Zero();
  Eigen::Matrix<double, 3, 4> lever_arms = Eigen::Matrix<double, 3, 4>::Zero();
  Eigen::Matrix<int, 20, 5> gait = Eigen::Matrix<int, 20, 5>::Zero();
  Eigen::Matrix<double, 12, 1> g = Eigen::Matrix<double, 12, 1>::Zero();

  Eigen::Matrix<double, 12, 12> A = Eigen::Matrix<double, 12, 12>::Identity();
  Eigen::Matrix<double, 12, 12> B = Eigen::Matrix<double, 12, 12>::Zero();
  Eigen::Matrix<double, 12, 1> x0 = Eigen::Matrix<double, 12, 1>::Zero();
  double x_next[12] = {};
  Eigen::MatrixXd x_f_applied;

  // Matrix ML
  const static int size_nz_ML = 5000;
  // int r_ML [size_nz_ML] = {}; // row indexes of non-zero values in matrix ML
  // int c_ML [size_nz_ML] = {}; // col indexes of non-zero values in matrix ML
  // double v_ML [size_nz_ML] = {};  // non-zero values in matrix ML
  // csc* ML_triplet; // Compressed Sparse Column matrix (triplet format)
  csc *ML;  // Compressed Sparse Column matrix
  inline void add_to_ML(int i, int j, double v, int *r_ML, int *c_ML,
                        double *v_ML);                                            // function to fill the triplet r/c/v
  inline void add_to_P(int i, int j, double v, int *r_P, int *c_P, double *v_P);  // function to fill the triplet r/c/v

  // Indices that are used to udpate ML
  int i_x_B[12 * 4] = {};
  int i_y_B[12 * 4] = {};
  int i_update_B[12 * 4] = {};
  // TODO FOR S ????

  // Matrix NK
  const static int size_nz_NK = 5000;
  double v_NK_up[size_nz_NK] = {};   // maxtrix NK (upper bound)
  double v_NK_low[size_nz_NK] = {};  // maxtrix NK (lower bound)
  double v_warmxf[size_nz_NK] = {};  // maxtrix NK (lower bound)

  // Matrix P
  const static int size_nz_P = 5000;
  // c_int r_P [size_nz_P] = {}; // row indexes of non-zero values in matrix ML
  // c_int c_P [size_nz_P] = {}; // col indexes of non-zero values in matrix ML
  // c_float v_P [size_nz_P] = {};  // non-zero values in matrix ML
  csc *P;  // Compressed Sparse Column matrix

  // Matrix Q
  const static int size_nz_Q = 5000;
  double Q[size_nz_Q] = {};  // Q is full of zeros

  // OSQP solver variables
  OSQPWorkspace *workspce = new OSQPWorkspace();
  OSQPData *data;
  OSQPSettings *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));

  // Matrices whose size depends on the arguments sent to the constructor function
  Eigen::Matrix<double, 12, Eigen::Dynamic> xref;
  Eigen::Matrix<double, Eigen::Dynamic, 1> x;
  Eigen::Matrix<int, Eigen::Dynamic, 1> S_gait;
  Eigen::Matrix<double, Eigen::Dynamic, 1> warmxf;
  Eigen::Matrix<double, Eigen::Dynamic, 1> NK_up;
  Eigen::Matrix<double, Eigen::Dynamic, 1> NK_low;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> D;
  Eigen::Matrix<int, Eigen::Dynamic, 1> i_off;

 public:
  MPC();
  MPC(double dt_in, int n_steps_in, double T_gait_in);

  int create_matrices();
  int create_ML();
  int create_NK();
  int create_weight_matrices();
  int update_matrices(Eigen::MatrixXd fsteps);
  int update_ML(Eigen::MatrixXd fsteps);
  int update_NK();
  int call_solver(int);
  int retrieve_result();
  double *get_x_next();
  int run(int num_iter, const Eigen::MatrixXd &xref_in, const Eigen::MatrixXd &fsteps_in);

  Eigen::Matrix<double, 3, 3> getSkew(Eigen::Matrix<double, 3, 1> v);
  int construct_S();
  int construct_gait(Eigen::MatrixXd fsteps_in);

  // Getters
  Eigen::MatrixXd get_latest_result();
  Eigen::MatrixXd get_gait();
  Eigen::MatrixXd get_Sgait();


  // Utils
  double gethref() { return h_ref; }
  void my_print_csc_matrix(csc *M, const char *name);
  void save_csc_matrix(csc *M, std::string filename);
  void save_dns_matrix(double *M, int size, std::string filename);

  // Bindings
  void run_python(const matXd &xref_py, const matXd &fsteps_py);

  // Eigen::Matrix<double, 12, 12> getA() { return A; }
  // Eigen::MatrixXf getML() { return ML; }
  /*void setDate(int year, int month, int day);
  int getYear();
  int getMonth();
  int getDay();*/
};

#endif  // MPC_H_INCLUDED
