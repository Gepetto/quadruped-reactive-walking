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
#include <vector>

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
    /// \param[in] dt_wbc_in Time step of whole body control
    /// \param[in] T_mpc_in MPC period (prediction horizon)
    /// \param[in] h_ref_in Reference height for the trunk
    /// \param[in] shoulderIn Position of shoulders in local frame
    /// \param[in] gaitIn Gait object to hold the gait informations
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    void initialize(double dt_in,
                    double dt_wbc_in,    
                    double T_mpc_in,
                    double h_ref_in,
                    MatrixN const& shouldersIn,
                    Gait& gaitIn,
                    int N_gait);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Destructor.
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    ~FootstepPlanner() {}

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Refresh footsteps locations (computation and update of relevant matrices)
    ///
    ///  \param[in] refresh  true if we move one step further in the gait
    ///  \param[in] k  number of remaining wbc time step for the current mpc time step (wbc frequency is higher so there are inter-steps)
    ///  \param[in] q  current position vector of the flying base in horizontal frame (linear and angular stacked)
    ///  \param[in] b_v  current velocity vector of the flying base in horizontal frame (linear and angular stacked)
    ///  \param[in] b_vref  desired velocity vector of the flying base in horizontal frame (linear and angular stacked)
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    MatrixN updateFootsteps(bool refresh, int k, VectorN const& q, Vector6 const& b_v, Vector6 const& b_vref);

    MatrixN getFootsteps();
    MatrixN getTargetFootsteps();
    MatrixN getRz();

private:
    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Compute the desired location of footsteps and update relevant matrices
    ///
    ///  \param[in] k  number of remaining wbc time step for the current mpc time step (wbc frequency is higher so there are inter-steps)
    ///  \param[in] q  current position vector of the flying base in horizontal frame (linear and angular stacked)
    ///  \param[in] b_v  current velocity vector of the flying base in horizontal frame (linear and angular stacked)
    ///  \param[in] b_vref  desired velocity vector of the flying base in horizontal frame (linear and angular stacked)
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    MatrixN computeTargetFootstep(int k, VectorN const& q, Vector6 const& b_v, Vector6 const& b_vref);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Refresh feet position when entering a new contact phase
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    void updateNewContact();

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Compute a X by 13 matrix containing the remaining number of steps of each phase of the gait (first column)
    ///        and the [x, y, z]^T desired position of each foot for each phase of the gait (12 other columns).
    ///        For feet currently touching the ground the desired position is where they currently are.
    ///
    /// \param[in] b_v current velocity vector of sthe flying base in horizontal frame (linear and angular stacked)
    /// \param[in] b_vref desired velocity vector of the flying base in horizontal frame (linear and angular stacked)
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    void computeFootsteps(int k, Vector6 const& b_v, Vector6 const& b_vref);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Compute the target location on the ground of a given foot for an upcoming stance phase
    ///
    /// \param[in] i considered phase (row of the gait matrix)
    /// \param[in] j considered foot (col of the gait matrix)
    /// \param[in] b_v current velocity vector of sthe flying base in horizontal frame (linear and angular stacked)
    /// \param[in] b_vref desired velocity vector of the flying base in horizontal frame (linear and angular stacked)
    ///
    /// \retval Matrix with the next footstep positions
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    void computeNextFootstep(int i, int j, Vector6 const& b_v, Vector6 const& b_vref);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Update desired location of footsteps using information coming from the footsteps planner
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    void updateTargetFootsteps();

    MatrixN vectorToMatrix(std::vector<Matrix34> const& array);

    Gait* gait_;  // Gait object to hold the gait informations

    double dt;      // Time step of the contact sequence (time step of the MPC)
    double dt_wbc;  // Time step of the whole body control
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
    Matrix34 currentFootstep_;  // Feet matrix
    Matrix34 nextFootstep_;     // Temporary matrix to perform computations
    Matrix34 targetFootstep_;   // In horizontal frame
    Matrix34 o_targetFootstep_;  // targetFootstep_ in world frame
    std::vector<Matrix34> footsteps_;

    MatrixN Rz;  // Rotation matrix along z axis
    VectorN dt_cum;
    VectorN yaws;
    VectorN dx;
    VectorN dy;

    Vector3 q_tmp;
    Vector3 q_dxdy;
    Vector3 RPY_;
    Eigen::Quaterniond quat_;
};

#endif  // FOOTSTEPPLANNER_H_INCLUDED
