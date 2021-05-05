///////////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief This is the header for StatePlanner class
///
/// \details Planner that outputs the reference trajectory of the base based on the reference 
///          velocity given by the user and the current position/velocity of the base
///
//////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef STATEPLANNER_H_INCLUDED
#define STATEPLANNER_H_INCLUDED

#include "pinocchio/math/rpy.hpp"
#include "qrw/Types.h"


class StatePlanner
{
public:
    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Empty constructor
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    StatePlanner();

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Destructor.
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    ~StatePlanner() {}

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Initializer
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    void initialize(double dt_in, double T_mpc_in, double h_ref_in);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Compute the reference trajectory of the CoM for each time step of the
    ///        predition horizon. The ouput is a matrix of size 12 by (N+1) with N the number
    ///        of time steps in the gait cycle (T_gait/dt) and 12 the position, orientation,
    ///        linear velocity and angular velocity vertically stacked. The first column contains
    ///        the current state while the remaining N columns contains the desired future states.
    ///
    /// \param[in] q current position vector of the flying base in horizontal frame (linear and angular stacked)
    /// \param[in] v current velocity vector of the flying base in horizontal frame (linear and angular stacked)
    /// \param[in] vref desired velocity vector of the flying base in horizontal frame (linear and angular stacked)
    /// \param[in] z_average average height of feet currently in stance phase
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    void computeReferenceStates(VectorN const& q, Vector6 const& v, Vector6 const& vref, double z_average);

    MatrixN getReferenceStates() { return referenceStates_; }
    int getNSteps() { return n_steps_; }

private:
    double dt_;         // Time step of the contact sequence (time step of the MPC)
    double h_ref_;       // Reference height for the trunk
    int n_steps_;        // Number of time steps in the prediction horizon

    Vector3 RPY_;        // To store roll, pitch and yaw angles 

    // Reference trajectory matrix of size 12 by (1 + N)  with the current state of
    // the robot in column 0 and the N steps of the prediction horizon in the others
    MatrixN referenceStates_;

    VectorN dt_vector_;  // Vector containing all time steps in the prediction horizon

};

#endif  // STATEPLANNER_H_INCLUDED
