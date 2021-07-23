#ifndef INVKIN_H_INCLUDED
#define INVKIN_H_INCLUDED

#include "pinocchio/math/rpy.hpp"
#include "pinocchio/spatial/explog.hpp"
#include "pinocchio/multibody/model.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/compute-all-terms.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include "qrw/Params.hpp"
#include "qrw/Types.h"

class InvKin
{
public:
    InvKin();
    void initialize(Params& params);

    void refreshAndCompute(Matrix14 const& contacts, Matrix43 const& pgoals, Matrix43 const& vgoals, Matrix43 const& agoals);
    
    void run_InvKin(VectorN const& q, VectorN const& dq, MatrixN const& contacts, MatrixN const& pgoals, MatrixN const& vgoals, MatrixN const& agoals);

    Eigen::MatrixXd get_q_step() { return q_step_; }
    Eigen::MatrixXd get_dq_cmd() { return dq_cmd_; }
    VectorN get_q_cmd() { return q_cmd_; }
    VectorN get_ddq_cmd() { return ddq_cmd_; }
    int get_foot_id(int i) { return foot_ids_[i];}

private:
    // Inputs of the constructor
    Params* params_;

    // Matrices initialisation
    
    Matrix12 invJ;
    Matrix112 acc;
    Matrix112 x_err;
    Matrix112 dx_r;

    Matrix43 pfeet_err;
    Matrix43 vfeet_ref;
    Matrix43 afeet;
    Matrix13 e_basispos;
    Matrix13 abasis;
    Matrix13 e_basisrot;
    Matrix13 awbasis;

    int foot_ids_[4] = {0, 0, 0, 0};

    Matrix43 posf_;
    Matrix43 vf_;
    Matrix43 wf_;
    Matrix43 af_;
    Matrix12 Jf_;
    Eigen::Matrix<double, 6, 12> Jf_tmp_;

    Vector12 ddq_cmd_;
    Vector12 dq_cmd_;
    Vector12 q_cmd_;
    Vector12 q_step_;

    pinocchio::Model model_;  // Pinocchio model for frame computations and inverse kinematics
    pinocchio::Data data_;  // Pinocchio datas for frame computations and inverse kinematics

    Eigen::Matrix<double, 4, 3> feet_position_ref = Eigen::Matrix<double, 4, 3>::Zero();
    Eigen::Matrix<double, 4, 3> feet_velocity_ref = Eigen::Matrix<double, 4, 3>::Zero();
    Eigen::Matrix<double, 4, 3> feet_acceleration_ref = Eigen::Matrix<double, 4, 3>::Zero();
    Eigen::Matrix<double, 1, 4> flag_in_contact = Eigen::Matrix<double, 1, 4>::Zero();
};

template <typename _Matrix_Type_>
_Matrix_Type_ pseudoInverse(const _Matrix_Type_& a, double epsilon = std::numeric_limits<double>::epsilon())
{
    Eigen::JacobiSVD<_Matrix_Type_> svd(a, Eigen::ComputeThinU | Eigen::ComputeThinV);
    double tolerance = epsilon * static_cast<double>(std::max(a.cols(), a.rows())) * svd.singularValues().array().abs()(0);
    return svd.matrixV() * (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().adjoint();
}
#endif  // INVKIN_H_INCLUDED
