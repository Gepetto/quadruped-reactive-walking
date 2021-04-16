///////////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief This is the header for FootstepPlanner class
///
/// \details Planner that outputs current and future locations of footsteps, the reference
///          trajectory of the base based on the reference velocity given by the user and the current
///          position/velocity of the base
///
//////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef FOOTSTEPPLANNER_H_INCLUDED
#define FOOTSTEPPLANNER_H_INCLUDED

#include "pinocchio/math/rpy.hpp"
#include "qrw/Gait.hpp"
#include "qrw/Types.h"


// Order of feet/legs: FL, FR, HL, HR

class FootstepPlanner
{
public:
    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Empty constructor
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    FootstepPlanner();

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Default constructor
    ///
    /// \param[in] dt_in Time step of the contact sequence (time step of the MPC)
    /// \param[in] T_mpc_in MPC period (prediction horizon)
    /// \param[in] h_ref_in Reference height for the trunk
    /// \param[in] shoulderIn Position of shoulders in local frame
    /// \param[in] gaitIn Gait object to hold the gait informations
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    void initialize(double dt_in,
                    double T_mpc_in,
                    double h_ref_in,
                    MatrixN const& shouldersIn,
                    Gait & gaitIn);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Destructor.
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    ~FootstepPlanner() {}


    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Compute the desired location of footsteps and update relevant matrices
    ///
    ///  \param[in] q  current position vector of the flying base in world frame (linear and angular stacked)
    ///  \param[in] v  current velocity vector of the flying base in world frame (linear and angular stacked)
    ///  \param[in] b_vref  desired velocity vector of the flying base in base frame (linear and angular stacked)
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    MatrixN computeTargetFootstep(VectorN const& q,
                                  Vector6 const& v,
                                  Vector6 const& b_vref);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Refresh feet position when entering a new contact phase
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    void updateNewContact();

    MatrixN getFootsteps();
    MatrixN getTargetFootsteps();

private:
    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Compute a X by 13 matrix containing the remaining number of steps of each phase of the gait (first column)
    ///        and the [x, y, z]^T desired position of each foot for each phase of the gait (12 other columns).
    ///        For feet currently touching the ground the desired position is where they currently are.
    ///
    /// \param[in] q current position vector of the flying base in world frame(linear and angular stacked)
    /// \param[in] v current velocity vector of sthe flying base in world frame(linear and angular stacked)
    /// \param[in] vref desired velocity vector of the flying base in world frame(linear and angular stacked)
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    void compute_footsteps(VectorN const& q, Vector6 const& v, Vector6 const& vref);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Compute the target location on the ground of a given foot for an upcoming stance phase
    ///
    /// \param[in] i considered phase (row of the gait matrix)
    /// \param[in] j considered foot (col of the gait matrix)
    ///
    /// \retval Matrix with the next footstep positions
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    void compute_next_footstep(int i, int j);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Update desired location of footsteps using information coming from the footsteps planner
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    void update_target_footsteps();

    MatrixN vectorToMatrix(std::array<Matrix34, N0_gait> const& array);

    Gait* gait_; // Gait object to hold the gait informations

    double dt;      // Time step of the contact sequence (time step of the MPC)
    double T_gait;  // Gait period
    double T_mpc;   // MPC period (prediction horizon)
    double h_ref;   // Reference height for the trunk

    // Predefined quantities
    double k_feedback;  // Feedback gain for the feedback term of the planner
    double g;           // Value of the gravity acceleartion
    double L;           // Value of the maximum allowed deviation due to leg length

    // Number of time steps in the prediction horizon
    int n_steps;  // T_mpc / time step of the MPC

    // Constant sized matrices
    Matrix34 shoulders_;        // Position of shoulders in local frame
    Matrix34 currentFootstep_;  // Feet matrix in world frame
    Matrix34 nextFootstep_;     // Feet matrix in world frame
    Matrix34 targetFootstep_;
    std::array<Matrix34, N0_gait> footsteps_;

    Matrix3 Rz;  // Predefined matrices for compute_footstep function
    VectorN dt_cum;
    VectorN yaws;
    VectorN dx;
    VectorN dy;

    Vector3 q_tmp;
    Vector3 q_dxdy;
    Vector3 RPY;
    Vector3 b_v;
    Vector6 b_vref;

};

#endif  // FOOTSTEPPLANNER_H_INCLUDED
