#include "qrw/Planner.hpp"

Planner::Planner()
    : gait_(nullptr)
    , footstepPlanner_()
    , fooTrajectoryGenerator_()
    , targetFootstep_(Matrix34::Zero())
{
}

Planner::Planner(double dt_in,
                 double dt_tsid_in,
                 double T_gait_in,
                 double T_mpc_in,
                 int k_mpc_in,
                 double h_ref_in,
                 Matrix34 const& intialFootsteps,
                 Matrix34 const& shouldersIn)
    : gait_(std::make_shared<Gait>())
    , targetFootstep_(shouldersIn)
{
    gait_->initialize(dt_in, T_gait_in, T_mpc_in);
    //footstepPlanner_.initialize(dt_in, k_mpc_in, T_mpc_in, h_ref_in, shouldersIn, gait_);
    //fooTrajectoryGenerator_.initialize(0.05, 0.07, shouldersIn, intialFootsteps, dt_tsid_in, k_mpc_in, gait_);
}

void Planner::run_planner(int const k,
                          VectorN const& q,
                          Vector6 const& v,
                          Vector6 const& b_vref)
{
    targetFootstep_ = footstepPlanner_.computeTargetFootstep(q, v, b_vref);
    fooTrajectoryGenerator_.update(k, targetFootstep_);

}

void Planner::updateGait(int const k,
                         int const k_mpc,
                         VectorN const& q,
                         int const joystickCode)
{
    gait_->changeGait(joystickCode, q);
    //footstepPlanner_.rollGait(k, k_mpc);
}

void Planner::setGait(MatrixN const& gaitMatrix)
{
    gait_->setGait(gaitMatrix);
}

MatrixN Planner::get_fsteps() { return footstepPlanner_.getFootsteps(); }
MatrixN Planner::get_gait() { return gait_->getCurrentGait(); }
Matrix3N Planner::get_goals() { return fooTrajectoryGenerator_.getFootPosition(); }
Matrix3N Planner::get_vgoals() { return fooTrajectoryGenerator_.getFootVelocity(); }
Matrix3N Planner::get_agoals() { return fooTrajectoryGenerator_.getFootAcceleration(); }