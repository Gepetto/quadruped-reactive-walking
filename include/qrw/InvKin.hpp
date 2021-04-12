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

    Eigen::MatrixXd refreshAndCompute(const Eigen::MatrixXd& x_cmd, const Eigen::MatrixXd& contacts,
                                      const Eigen::MatrixXd& goals, const Eigen::MatrixXd& vgoals, const Eigen::MatrixXd& agoals,
                                      const Eigen::MatrixXd& posf, const Eigen::MatrixXd& vf, const Eigen::MatrixXd& wf, const Eigen::MatrixXd& af,
                                      const Eigen::MatrixXd& Jf, const Eigen::MatrixXd& posb, const Eigen::MatrixXd& rotb, const Eigen::MatrixXd& vb,
                                      const Eigen::MatrixXd& ab, const Eigen::MatrixXd& Jb);
    Eigen::MatrixXd computeInvKin(const Eigen::MatrixXd& posf, const Eigen::MatrixXd& vf, const Eigen::MatrixXd& wf, const Eigen::MatrixXd& af,
                                  const Eigen::MatrixXd& Jf, const Eigen::MatrixXd& posb, const Eigen::MatrixXd& rotb, const Eigen::MatrixXd& vb, const Eigen::MatrixXd& ab,
                                  const Eigen::MatrixXd& Jb);
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
};

template <typename _Matrix_Type_>
_Matrix_Type_ pseudoInverse(const _Matrix_Type_& a, double epsilon = std::numeric_limits<double>::epsilon())
{
    Eigen::JacobiSVD<_Matrix_Type_> svd(a, Eigen::ComputeThinU | Eigen::ComputeThinV);
    double tolerance = epsilon * static_cast<double>(std::max(a.cols(), a.rows())) * svd.singularValues().array().abs()(0);
    return svd.matrixV() * (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().adjoint();
}
#endif  // INVKIN_H_INCLUDED
