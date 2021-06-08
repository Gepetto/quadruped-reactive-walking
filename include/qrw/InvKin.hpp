#ifndef INVKIN_H_INCLUDED
#define INVKIN_H_INCLUDED

#include "pinocchio/math/rpy.hpp"
#include "pinocchio/spatial/explog.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

class InvKin
{
public:
    InvKin();
    InvKin(double dt_in);

    Eigen::Matrix<double, 1, 3> cross3(Eigen::Matrix<double, 1, 3> left, Eigen::Matrix<double, 1, 3> right);

    Eigen::MatrixXd refreshAndCompute(const Eigen::MatrixXd& contacts,
                                      const Eigen::MatrixXd& goals, const Eigen::MatrixXd& vgoals, const Eigen::MatrixXd& agoals,
                                      const Eigen::MatrixXd& posf, const Eigen::MatrixXd& vf, const Eigen::MatrixXd& wf,
                                      const Eigen::MatrixXd& af, const Eigen::MatrixXd& Jf);
    Eigen::MatrixXd get_q_step();
    Eigen::MatrixXd get_dq_cmd();

private:
    // Inputs of the constructor
    double dt;  // Time step of the contact sequence (time step of the MPC)

    // Matrices initialisation
    Eigen::Matrix<double, 4, 3> feet_position_ref = Eigen::Matrix<double, 4, 3>::Zero();
    Eigen::Matrix<double, 4, 3> feet_velocity_ref = Eigen::Matrix<double, 4, 3>::Zero();
    Eigen::Matrix<double, 4, 3> feet_acceleration_ref = Eigen::Matrix<double, 4, 3>::Zero();
    Eigen::Matrix<double, 1, 4> flag_in_contact = Eigen::Matrix<double, 1, 4>::Zero();
    
    Eigen::Matrix<double, 12, 12> invJ = Eigen::Matrix<double, 12, 12>::Zero();
    Eigen::Matrix<double, 1, 12> acc = Eigen::Matrix<double, 1, 12>::Zero();
    Eigen::Matrix<double, 1, 12> x_err = Eigen::Matrix<double, 1, 12>::Zero();
    Eigen::Matrix<double, 1, 12> dx_r = Eigen::Matrix<double, 1, 12>::Zero();

    Eigen::Matrix<double, 4, 3> pfeet_err = Eigen::Matrix<double, 4, 3>::Zero();
    Eigen::Matrix<double, 4, 3> vfeet_ref = Eigen::Matrix<double, 4, 3>::Zero();
    Eigen::Matrix<double, 4, 3> afeet = Eigen::Matrix<double, 4, 3>::Zero();
    Eigen::Matrix<double, 1, 3> e_basispos = Eigen::Matrix<double, 1, 3>::Zero();
    Eigen::Matrix<double, 1, 3> abasis = Eigen::Matrix<double, 1, 3>::Zero();
    Eigen::Matrix<double, 1, 3> e_basisrot = Eigen::Matrix<double, 1, 3>::Zero();
    Eigen::Matrix<double, 1, 3> awbasis = Eigen::Matrix<double, 1, 3>::Zero();

    Eigen::MatrixXd ddq = Eigen::MatrixXd::Zero(12, 1);
    Eigen::MatrixXd q_step = Eigen::MatrixXd::Zero(12, 1);
    Eigen::MatrixXd dq_cmd = Eigen::MatrixXd::Zero(12, 1);

    // Gains
    double Kp_flyingfeet = 100.0; // 1000
    double Kd_flyingfeet = 2.0 * std::sqrt(Kp_flyingfeet); // 5.0 *
};

template <typename _Matrix_Type_>
_Matrix_Type_ pseudoInverse(const _Matrix_Type_& a, double epsilon = std::numeric_limits<double>::epsilon())
{
    Eigen::JacobiSVD<_Matrix_Type_> svd(a, Eigen::ComputeThinU | Eigen::ComputeThinV);
    double tolerance = epsilon * static_cast<double>(std::max(a.cols(), a.rows())) * svd.singularValues().array().abs()(0);
    return svd.matrixV() * (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().adjoint();
}
#endif  // INVKIN_H_INCLUDED
