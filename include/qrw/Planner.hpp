///////////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief This is the header for Planner class
///
/// \details Planner that outputs current and future locations of footsteps, the reference
///          trajectory of the base and the position, velocity, acceleration commands for feet in
///          swing phase based on the reference velocity given by the user and the current
///          position/velocity of the base
///
//////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef PLANNER_H_INCLUDED
#define PLANNER_H_INCLUDED

#include "qrw/FootTrajectoryGenerator.hpp"
#include "qrw/FootstepPlanner.hpp"
#include "qrw/Gait.hpp"
#include "qrw/Types.h"

// Number of rows in the gait matrix. Arbitrary value that should be set high enough so that there is always at
// least one empty line at the end of the gait matrix

// Order of feet/legs: FL, FR, HL, HR

class Planner
{
public:
    ////////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief Empty constructor
    ////////////////////////////////////////////////////////////////////////////////////////////////
    Planner();

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Default constructor
    ///
    /// \param[in] dt_in
    /// \param[in] dt_tsid_in
    /// \param[in] T_gait_in
    /// \param[in] T_mpc_in
    /// \param[in] k_mpc_in
    /// \param[in] h_ref_in
    /// \param[in] fsteps_in
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    Planner(double dt_in,
            double dt_tsid_in,
            double T_gait_in,
            double T_mpc_in,
            int k_mpc_in,
            double h_ref_in,
            Matrix34 const& intialFootsteps,
            Matrix34 const& shouldersIn);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief Destructor.
    ////////////////////////////////////////////////////////////////////////////////////////////////
    ~Planner() {}


    void Print();

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Run the planner for one iteration of the main control loop
    ///
    ///  \param[in] k  number of time steps since the start of the simulation
    ///  \param[in] q  current position vector of the flying base in world frame (linear and angular stacked)
    ///  \param[in] v  current velocity vector of the flying base in world frame (linear and angular stacked)
    ///  \param[in] b_vref  desired velocity vector of the flying base in base frame (linear and angular stacked)
    ///  \param[in] z_average  average height of feet currently in stance phase
    ///  \param[in] joystick_code  integer to trigger events with the joystick
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    void run_planner(int const k,
                     VectorN const& q,
                     Vector6 const& v,
                     Vector6 const& b_vref,
                     double const z_average,
                     int const joystickCode);

    // Accessors (to retrieve C data from Python)
    MatrixN get_xref();
    MatrixN get_fsteps();
    MatrixN get_gait();
    Matrix3N get_goals();
    Matrix3N get_vgoals();
    Matrix3N get_agoals();

private:
    std::shared_ptr<Gait> gait_;  // Gait object to hold the gait information
    FootstepPlanner footstepPlanner_;
    FootTrajectoryGenerator fooTrajectoryGenerator_;
    Matrix34 targetFootstep_;
};

#endif  // PLANNER_H_INCLUDED
