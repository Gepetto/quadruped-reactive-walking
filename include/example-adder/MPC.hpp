#ifndef MPC_H_INCLUDED
#define MPC_H_INCLUDED

#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "osqp_folder/include/osqp.h"
#include "osqp_folder/include/cs.h"

class MPC
{
private:
    float dt, mass, mu, T_gait, h_ref;
    int n_steps, cpt;

    Eigen::Matrix<float, 3, 3> gI;
    Eigen::Matrix<float, 6, 1> q;
    Eigen::Matrix<float, 6, 1> v = Eigen::Matrix<float, 6, 1>::Zero();
    Eigen::Matrix<float, 3, 4> footholds = Eigen::Matrix<float, 3, 4>::Zero();
    Eigen::Matrix<float, 3, 4> lever_arms = Eigen::Matrix<float, 3, 4>::Zero();
    Eigen::Matrix<int, 20, 5> gait = Eigen::Matrix<int, 20, 5>::Zero();

    Eigen::Matrix<float, 12, 12> A = Eigen::Matrix<float, 12, 12>::Zero();
    Eigen::Matrix<float, 12, 12> B = Eigen::Matrix<float, 12, 12>::Zero();

    // Matrix ML
    const static int size_nz_ML = 5000;
    c_int r_ML [size_nz_ML] = {}; // row indexes of non-zero values in matrix ML
    c_int c_ML [size_nz_ML] = {}; // col indexes of non-zero values in matrix ML
    c_float v_ML [size_nz_ML] = {};  // non-zero values in matrix ML
    csc* ML; // Compressed Sparse Column matrix
    inline void add_to_ML(int i, int j, float v); // function to fill the triplet r/c/v

    // Indices that are used to udpate ML
    int i_x_B [12*4] = {};
    int i_y_B [12*4] = {};
    int i_update_B [12*4] = {};
    // TODO FOR S ????

    // Matrix NK
    const static int size_nz_NK = 1000;
    c_float v_NK [size_nz_NK] = {};  // maxtrix NK

    // Matrices whose size depends on the arguments sent to the constructor function
    Eigen::Matrix<float, 12, Eigen::Dynamic> xref;
    Eigen::Matrix<float, 12, Eigen::Dynamic> x;
    Eigen::Matrix<int, Eigen::Dynamic, 1> S_gait;
    Eigen::Matrix<float, Eigen::Dynamic, 1> warmxf;

public:

    MPC(float dt_in, int n_steps_in, float T_gait_in);

    int create_matrices();
    int create_ML();
    int create_NK();
    int create_weight_matrices();

    int construct_S();
    Eigen::Matrix<float, 3, 3> getSkew(Eigen::Matrix<float, 3, 1> v);
    float gethref() { return h_ref; }
    
    
    //Eigen::Matrix<float, 12, 12> getA() { return A; }
    //Eigen::MatrixXf getML() { return ML; }
    /*void setDate(int year, int month, int day);
    int getYear();
    int getMonth();
    int getDay();*/
    
};
// Eigen::Matrix<float, 1, 2> get_projection_on_border(Eigen::Matrix<float,1,2>  robot, Eigen::Matrix<float, 1, 6> data_closest, float const& angle);


#endif // MPC_H_INCLUDED