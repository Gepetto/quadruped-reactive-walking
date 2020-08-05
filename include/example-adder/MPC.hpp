#ifndef MPC_H_INCLUDED
#define MPC_H_INCLUDED

#include <iostream>
#include <cmath>
#include <limits>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "osqp_folder/include/osqp.h"
#include "osqp_folder/include/cs.h"

class MPC
{
private:
    double dt, mass, mu, T_gait, h_ref;
    int n_steps, cpt_ML, cpt_P;

    Eigen::Matrix<double, 3, 3> gI;
    Eigen::Matrix<double, 6, 1> q;
    Eigen::Matrix<double, 6, 1> v = Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 3, 4> footholds = Eigen::Matrix<double, 3, 4>::Zero();
    Eigen::Matrix<double, 3, 4> lever_arms = Eigen::Matrix<double, 3, 4>::Zero();
    Eigen::Matrix<int, 20, 5> gait = Eigen::Matrix<int, 20, 5>::Zero();
    Eigen::Matrix<double, 12, 1> g = Eigen::Matrix<double, 12, 1>::Zero();

    Eigen::Matrix<double, 12, 12> A = Eigen::Matrix<double, 12, 12>::Zero();
    Eigen::Matrix<double, 12, 12> B = Eigen::Matrix<double, 12, 12>::Zero();
    Eigen::Matrix<double, 12, 1> x0 = Eigen::Matrix<double, 12, 1>::Zero();
    double f_applied [12] = {};

    // Matrix ML
    const static int size_nz_ML = 5000;
    c_int r_ML [size_nz_ML] = {}; // row indexes of non-zero values in matrix ML
    c_int c_ML [size_nz_ML] = {}; // col indexes of non-zero values in matrix ML
    c_float v_ML [size_nz_ML] = {};  // non-zero values in matrix ML
    csc* ML; // Compressed Sparse Column matrix
    inline void add_to_ML(int i, int j, double v); // function to fill the triplet r/c/v
    inline void add_to_P(int i, int j, double v); // function to fill the triplet r/c/v

    // Indices that are used to udpate ML
    int i_x_B [12*4] = {};
    int i_y_B [12*4] = {};
    int i_update_B [12*4] = {};
    // TODO FOR S ????

    // Matrix NK
    const static int size_nz_NK = 5000;
    double v_NK_up [size_nz_NK] = {};  // maxtrix NK (upper bound)
    double v_NK_low [size_nz_NK] = {};  // maxtrix NK (lower bound)

    // Matrix P
    const static int size_nz_P = 5000;
    c_int r_P [size_nz_P] = {}; // row indexes of non-zero values in matrix ML
    c_int c_P [size_nz_P] = {}; // col indexes of non-zero values in matrix ML
    c_float v_P [size_nz_P] = {};  // non-zero values in matrix ML
    csc* P; // Compressed Sparse Column matrix

    // Matrix Q
    const static int size_nz_Q = 5000;
    double Q [size_nz_Q] = {};  // Q is full of zeros

    // OSQP solver variables
    OSQPWorkspace * workspce = new OSQPWorkspace();
    OSQPData * data = new OSQPData();
    OSQPSettings * settings = new OSQPSettings();

    // Matrices whose size depends on the arguments sent to the constructor function
    Eigen::Matrix<double, 12, Eigen::Dynamic> xref;
    Eigen::Matrix<double, 12, Eigen::Dynamic> x;
    Eigen::Matrix<int, Eigen::Dynamic, 1> S_gait;
    Eigen::Matrix<double, Eigen::Dynamic, 1> warmxf;
    Eigen::Matrix<double, Eigen::Dynamic, 1> NK_up;
    Eigen::Matrix<double, Eigen::Dynamic, 1> NK_low;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> D;
    Eigen::Matrix<int, Eigen::Dynamic, 1> i_off;

public:

    MPC(double dt_in, int n_steps_in, double T_gait_in);

    int create_matrices();
    int create_ML();
    int create_NK();
    int create_weight_matrices();
    int update_matrices(Eigen::Matrix<double, 20, 13> fsteps);
    int update_ML(Eigen::Matrix<double, 20, 13> fsteps);
    int update_NK();
    int call_solver(int);
    int retrieve_result();
    double * get_latest_result();
    int run(Eigen::Matrix<double, 12, Eigen::Dynamic> xref_in, Eigen::Matrix<double, 20, 13> fsteps_in);
    
    Eigen::Matrix<double, 3, 3> getSkew(Eigen::Matrix<double, 3, 1> v);
    int construct_S();
    int construct_gait(Eigen::Matrix<double, 20, 13> fsteps_in);
    
    // Accessor
    double gethref() { return h_ref; }

    //Eigen::Matrix<double, 12, 12> getA() { return A; }
    //Eigen::MatrixXf getML() { return ML; }
    /*void setDate(int year, int month, int day);
    int getYear();
    int getMonth();
    int getDay();*/
    
};
// Eigen::Matrix<double, 1, 2> get_projection_on_border(Eigen::Matrix<double,1,2>  robot, Eigen::Matrix<double, 1, 6> data_closest, double const& angle);


#endif // MPC_H_INCLUDED