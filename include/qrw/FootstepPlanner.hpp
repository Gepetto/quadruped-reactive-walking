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

#include "pinocchio/multibody/model.hpp"
#include "pinocchio/multibody/data.hpp"

#include "qrw/Gait.hpp"
#include "qrw/Params.hpp"
#include <vector>

// Order of feet/legs: FL, FR, HL, HR

class FootstepPlanner {
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
  /// \param[in] params Object that stores parameters
  /// \param[in] gaitIn Gait object to hold the gait informations
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void initialize(Params& params, Gait& gaitIn);

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
  ///  \param[in] k Numero of the current loop
  ///  \param[in] q Current position vector of the flying base in horizontal frame (linear and angular stacked) +
  ///  actuators
  ///  \param[in] b_v Current velocity vector of the flying base in horizontal frame (linear and angular
  ///  stacked)
  ///  \param[in] b_vref Desired velocity vector of the flying base in horizontal frame (linear and angular
  ///  stacked)
  ///  \param[in] ftgPositions Target foot positions outputted by the foot trajectory generator (in world frame)
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  MatrixN updateFootsteps(int k, VectorN const& q, Vector6 const& b_v, Vector6 const& b_vref,
                          MatrixN const& ftgPositions);

  MatrixN getFootsteps();
  MatrixN getTargetFootsteps();

 private:
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Compute the desired location of footsteps and update relevant matrices
  ///
  ///  \param[in] k Numero of the current loop
  ///  \param[in] q Current position vector of the flying base in horizontal frame (linear and
  ///  angular stacked)
  ///  \param[in] b_v Current velocity vector of the flying base in horizontal frame (linear and
  ///  angular stacked)
  ///  \param[in] b_vref Desired velocity vector of the flying base in horizontal frame (linear and
  ///  angular stacked)
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  MatrixN computeTargetFootstep(int k, Vector6 const& q, Vector6 const& b_v, Vector6 const& b_vref);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Refresh feet position when entering a new contact phase
  ///
  ///  \param[in] q Current configuration vector
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void updateNewContact(Vector18 const& q);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Compute a X by 13 matrix containing the remaining number of steps of each phase of the gait (first column)
  ///        and the [x, y, z]^T desired position of each foot for each phase of the gait (12 other columns).
  ///        For feet currently touching the ground the desired position is where they currently are.
  ///
  /// \param[in] k Number of remaining wbc time step for the current mpc time step (wbc frequency is higher so there
  ///  are inter-steps)
  /// \param[in] b_v Current velocity vector of sthe flying base in horizontal frame (linear and angular stacked)
  /// \param[in] b_vref Desired velocity vector of the flying base in horizontal frame (linear and angular stacked)
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void computeFootsteps(int k, Vector6 const& b_v, Vector6 const& b_vref);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Compute the target location on the ground of a given foot for an upcoming stance phase
  ///
  /// \param[in] i Considered phase (row of the gait matrix)
  /// \param[in] j Considered foot (col of the gait matrix)
  /// \param[in] b_v Current velocity vector of sthe flying base in horizontal frame (linear and angular stacked)
  /// \param[in] b_vref Desired velocity vector of the flying base in horizontal frame (linear and angular stacked)
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

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Transform a std::vector of N 3x4 matrices into a single Nx12 matrix
  ///
  /// \param[in] array The std::vector of N 3x4 matrices to transform
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  MatrixN vectorToMatrix(std::vector<Matrix34> const& array);

  Params* params_;  // Params object to store parameters
  Gait* gait_;      // Gait object to hold the gait informations

  double dt;      // Time step of the contact sequence (time step of the MPC)
  double dt_wbc;  // Time step of the whole body control
  double h_ref;   // Reference height for the trunk
  int k_mpc_;     // Number of wbc time steps for each MPC time step

  // Predefined quantities
  double g;  // Value of the gravity acceleartion
  double L;  // Value of the maximum allowed deviation due to leg length

  // Number of time steps in the prediction horizon
  int n_steps;  // T_mpc / time step of the MPC

  // Constant sized matrices
  Matrix34 footsteps_under_shoulders_;  // Positions of footsteps to be "under the shoulder"
  Matrix34 footsteps_offset_;           // Hardcoded offset to add to footsteps positions
  Matrix34 currentFootstep_;            // Feet matrix
  Matrix34 nextFootstep_;               // Temporary matrix to perform computations
  Matrix34 targetFootstep_;             // In horizontal frame
  Matrix34 o_targetFootstep_;           // targetFootstep_ in world frame
  std::vector<Matrix34> footsteps_;     // Desired footsteps locations for each step of the horizon
  RowVector4 previousGait_;             // Last gait status used to update footsteps
  RowVector4 previousHeight_;           // Height of the last contact phase for each foot

  MatrixN Rz;      // Rotation matrix along z axis
  VectorN dt_cum;  // Cumulated time vector
  VectorN yaws;    // Predicted yaw variation for each cumulated time
  VectorN dx;      // Predicted x displacement for each cumulated time
  VectorN dy;      // Predicted y displacement for each cumulated time

  Vector3 q_dxdy;  // Temporary storage variable for offset to the future position
  Vector3 RPY_;    // Temporary storage variable for roll pitch yaw orientation

  pinocchio::Model model_;          // Pinocchio model for forward kinematics
  pinocchio::Data data_;            // Pinocchio datas for forward kinematics
  int foot_ids_[4] = {0, 0, 0, 0};  // Indexes of feet frames
  Matrix34 pos_feet_;               // Estimated feet positions based on measurements
  Vector19 q_FK_;                   // Estimated state of the base (height, roll, pitch, joints)
};

#endif  // FOOTSTEPPLANNER_H_INCLUDED
