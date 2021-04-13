#include "qrw/StatePlanner.hpp"

StatePlanner::StatePlanner()
    : dt_(0.0)
    , T_mpc_(0.0)
    , h_ref_(0.0)
    , n_steps_(0)
    , RPY_(Vector3::Zero())
{
    // Empty
}

void StatePlanner::initialize(double dt_in, double T_mpc_in, double h_ref_in)
{
    dt_ = dt_in;
    T_mpc_ = T_mpc_in;
    h_ref_ = h_ref_in;
    n_steps_ = (int)std::lround(T_mpc_in / dt_in);
    xref_ = MatrixN::Zero(12, 1 + n_steps_);
    dt_vector_ = VectorN::LinSpaced(n_steps_, dt_, T_mpc_);
}

void StatePlanner::computeRefStates(VectorN const& q, Vector6 const& v, Vector6 const& vref, double z_average)
{
    Eigen::Quaterniond quat(q(6), q(3), q(4), q(5));  // w, x, y, z
    RPY_ << pinocchio::rpy::matrixToRpy(quat.toRotationMatrix());

    // Update yaw and yaw velocity
    xref_.block(5, 1, 1, n_steps_) = vref(5) * dt_vector_.transpose();
    for (int i = 0; i < n_steps_; i++)
    {
        xref_(11, 1 + i) = vref(5);
    }

    // Update x and y velocities taking into account the rotation of the base over the prediction horizon
    for (int i = 0; i < n_steps_; i++)
    {
        xref_(6, 1 + i) = vref(0) * std::cos(xref_(5, 1 + i)) - vref(1) * std::sin(xref_(5, 1 + i));
        xref_(7, 1 + i) = vref(0) * std::sin(xref_(5, 1 + i)) + vref(1) * std::cos(xref_(5, 1 + i));
    }

    // Update x and y depending on x and y velocities (cumulative sum)
    if (vref(5) != 0)
    {
        for (int i = 0; i < n_steps_; i++)
        {
            xref_(0, 1 + i) = (vref(0) * std::sin(vref(5) * dt_vector_(i)) + vref(1) * (std::cos(vref(5) * dt_vector_(i)) - 1.0)) / vref(5);
            xref_(1, 1 + i) = (vref(1) * std::sin(vref(5) * dt_vector_(i)) - vref(0) * (std::cos(vref(5) * dt_vector_(i)) - 1.0)) / vref(5);
        }
    }
    else
    {
        for (int i = 0; i < n_steps_; i++)
        {
            xref_(0, 1 + i) = vref(0) * dt_vector_(i);
            xref_(1, 1 + i) = vref(1) * dt_vector_(i);
        }
    }

    for (int i = 0; i < n_steps_; i++)
    {
        xref_(5, 1 + i) += RPY_(2);
        xref_(2, 1 + i) = h_ref_ + z_average;
        xref_(8, 1 + i) = 0.0;
    }

    // No need to update Z velocity as the reference is always 0
    // No need to update roll and roll velocity as the reference is always 0 for those
    // No need to update pitch and pitch velocity as the reference is always 0 for those

    // Update the current state
    xref_.block(0, 0, 3, 1) = q.head(3);
    xref_.block(3, 0, 3, 1) = RPY_;
    xref_.block(6, 0, 3, 1) = v.head(3);
    xref_.block(9, 0, 3, 1) = v.tail(3);

    for (int i = 0; i < n_steps_; i++)
    {
        xref_(0, 1 + i) += xref_(0, 0);
        xref_(1, 1 + i) += xref_(1, 0);
    }

    /*if (gait_->getIsStatic())
    {
        Vector19 q_static = gait_->getQStatic();
        Eigen::Quaterniond quatStatic(q_static(6, 0), q_static(3, 0), q_static(4, 0), q_static(5, 0));  // w, x, y, z
        RPY_ << pinocchio::rpy::matrixToRpy(quatStatic.toRotationMatrix());

        for (int i = 0; i < n_steps_; i++)
        {
            xref_.block(0, 1 + i, 3, 1) = q_static.block(0, 0, 3, 1);
            xref_.block(3, 1 + i, 3, 1) = RPY_;
        }
    }*/

}
