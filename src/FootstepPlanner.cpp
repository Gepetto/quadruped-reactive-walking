#include "qrw/FootstepPlanner.hpp"

FootstepPlanner::FootstepPlanner()
    : gait_(NULL)
    , k_feedback(0.03)
    , g(9.81)
    , L(0.155)
    , nextFootstep_(Matrix34::Zero())
    , footsteps_()
    , Rz(Matrix3::Zero())
    , dt_cum(VectorN::Zero(N0_gait))
    , yaws(VectorN::Zero(N0_gait))
    , dx(VectorN::Zero(N0_gait))
    , dy(VectorN::Zero(N0_gait))
    , q_tmp(Vector3::Zero())
    , q_dxdy(Vector3::Zero())
    , RPY(Vector3::Zero())
    , b_v(Vector3::Zero())
    , b_vref(Vector6::Zero())
{
    // Empty
}

void FootstepPlanner::initialize(double dt_in,
                                 double T_mpc_in,
                                 double h_ref_in,
                                 MatrixN const& shouldersIn,
                                 Gait & gaitIn)
{
    dt = dt_in;
    T_mpc = T_mpc_in;
    h_ref = h_ref_in;
    n_steps = (int)std::lround(T_mpc_in / dt_in);
    shoulders_ = shouldersIn;
    currentFootstep_ = shouldersIn.block(0, 0, 3, 4);
    gait_ = &gaitIn;
    targetFootstep_ = shouldersIn;
    footsteps_.fill(Matrix34::Zero());
    Rz(2, 2) = 1.0;
}

void FootstepPlanner::compute_footsteps(VectorN const& q, Vector6 const& v, Vector6 const& vref)
{
    footsteps_.fill(Matrix34::Zero());
    MatrixN gait = gait_->getCurrentGait();

    // Set current position of feet for feet in stance phase
    for (int j = 0; j < 4; j++)
    {
        if (gait(0, j) == 1.0)
        {
            footsteps_[0].col(j) = currentFootstep_.col(j);
        }
    }

    // Cumulative time by adding the terms in the first column (remaining number of timesteps)
    // Get future yaw yaws compared to current position
    dt_cum(0) = dt;
    yaws(0) = vref(5) * dt_cum(0) + RPY(2);
    for (int j = 1; j < N0_gait; j++)
    {
        dt_cum(j) = gait.row(j).isZero() ? dt_cum(j - 1) : dt_cum(j - 1) + gait(j) * dt;
        yaws(j) = vref(5) * dt_cum(j) + RPY(2);
    }

    // Displacement following the reference velocity compared to current position
    if (vref(5, 0) != 0)
    {
        for (int j = 0; j < N0_gait; j++)
        {
            dx(j) = (v(0) * std::sin(vref(5) * dt_cum(j)) + v(1) * (std::cos(vref(5) * dt_cum(j)) - 1.0)) / vref(5);
            dy(j) = (v(1) * std::sin(vref(5) * dt_cum(j)) - v(0) * (std::cos(vref(5) * dt_cum(j)) - 1.0)) / vref(5);
        }
    }
    else
    {
        for (int j = 0; j < N0_gait; j++)
        {
            dx(j) = v(0) * dt_cum(j);
            dy(j) = v(1) * dt_cum(j);
        }
    }

    // Get current and reference velocities in base frame (rotated yaw)
    b_v = Rz.transpose() * v.head(3);
    b_vref.head(3) = Rz.transpose() * vref.head(3);
    b_vref.tail(3) = Rz.transpose() * vref.tail(3);

    // Update the footstep matrix depending on the different phases of the gait (swing & stance)
    int i = 1;
    while (!gait.row(i).isZero())
    {
        // Feet that were in stance phase and are still in stance phase do not move
        for (int j = 0; j < 4; j++)
        {
            if (gait(i - 1, j) * gait(i, j) > 0)
            {
                footsteps_[i].col(j) = footsteps_[i - 1].col(j);
            }
        }

        // Current position without height
        Vector3 q_tmp = q.head(3);
        q_tmp(2) = 0.0;

        // Feet that were in swing phase and are now in stance phase need to be updated
        for (int j = 0; j < 4; j++)
        {
            if ((1 - gait(i - 1, j)) * gait(i, j) > 0)
            {
                // Offset to the future position
                q_dxdy << dx(i - 1, 0), dy(i - 1, 0), 0.0;

                // Get future desired position of footsteps
                compute_next_footstep(i, j);

                // Get desired position of footstep compared to current position
                double c = std::cos(yaws(i - 1));
                double s = std::sin(yaws(i - 1));
                Rz.topLeftCorner<2, 2>() << c, -s, s, c;

                footsteps_[i].col(j) = (Rz * nextFootstep_.col(j) + q_tmp + q_dxdy).transpose();
            }
        }
        i++;
    }
}

void FootstepPlanner::compute_next_footstep(int i, int j)
{
    nextFootstep_ = Matrix34::Zero();

    double t_stance = gait_->getPhaseDuration(i, j, 1.0);  // 1.0 for stance phase

    // Add symmetry term
    nextFootstep_.col(j) = t_stance * 0.5 * b_v;

    // Add feedback term
    nextFootstep_.col(j) += k_feedback * (b_v - b_vref.head(3));

    // Add centrifugal term
    Vector3 cross;
    cross << b_v(1) * b_vref(5) - b_v(2) * b_vref(4), b_v(2) * b_vref(3) - b_v(0) * b_vref(5), 0.0;
    nextFootstep_.col(j) += 0.5 * std::sqrt(h_ref / g) * cross;

    // Legs have a limited length so the deviation has to be limited
    nextFootstep_(0, j) = std::min(nextFootstep_(0, j), L);
    nextFootstep_(0, j) = std::max(nextFootstep_(0, j), -L);
    nextFootstep_(1, j) = std::min(nextFootstep_(1, j), L);
    nextFootstep_(1, j) = std::max(nextFootstep_(1, j), -L);

    // Add shoulders
    nextFootstep_.col(j) += shoulders_.col(j);

    // Remove Z component (working on flat ground)
    nextFootstep_.row(2) = Vector4::Zero().transpose();
}

void FootstepPlanner::update_target_footsteps()
{
    for (int i = 0; i < 4; i++)
    {
        int index = 0;
        while (footsteps_[index](0, i) == 0.0)
        {
            index++;
        }
        targetFootstep_.col(i) << footsteps_[index](0, i), footsteps_[index](1, i), 0.0;
    }
}

MatrixN FootstepPlanner::computeTargetFootstep(VectorN const& q,
                                               Vector6 const& v,
                                               Vector6 const& b_vref)
{
    // Get the reference velocity in world frame (given in base frame)
    Eigen::Quaterniond quat(q(6), q(3), q(4), q(5));  // w, x, y, z
    RPY << pinocchio::rpy::matrixToRpy(quat.toRotationMatrix());

    double c = std::cos(RPY(2));
    double s = std::sin(RPY(2));
    Rz.topLeftCorner<2, 2>() << c, -s, s, c;

    Vector6 vref = b_vref;
    vref.head(3) = Rz * b_vref.head(3);
    

    // Compute the desired location of footsteps over the prediction horizon
    compute_footsteps(q, v, vref);

    // Update desired location of footsteps on the ground
    update_target_footsteps();
    return targetFootstep_;
}

void FootstepPlanner::updateNewContact() // Gait const& gait) // MaxtrixN const& currentGait)
{
    // Entering new contact phase, store positions of feet that are now in contact
    for (int i = 0; i < 4; i++)
    {
        if (gait_->getCurrentGaitCoeff(0, i) == 1.0)   //if (currentGait(0, 1 + i) == 1.0)
        {
            currentFootstep_.col(i) = (footsteps_[1]).col(i);
        }
    }
}

MatrixN FootstepPlanner::getFootsteps() { return vectorToMatrix(footsteps_); }
MatrixN FootstepPlanner::getTargetFootsteps() { return targetFootstep_; }

MatrixN FootstepPlanner::vectorToMatrix(std::array<Matrix34, N0_gait> const& array)
{
    MatrixN M = MatrixN::Zero(N0_gait, 12);
    for (int i = 0; i < N0_gait; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            M.row(i).segment<3>(3 * j) = array[i].col(j);
        }
    }
    return M;
}