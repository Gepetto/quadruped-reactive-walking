///////////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief This is the header for FootTrajectoryGenerator class
///
/// \details This class generates a reference trajectory for the swing foot, in position, velocity
///          and acceleration
///
//////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef TRAJGEN_H_INCLUDED
#define TRAJGEN_H_INCLUDED

#include "qrw/Gait.hpp"
#include "qrw/Params.hpp"

class FootTrajectoryGenerator {
 public:
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Constructor
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  FootTrajectoryGenerator();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Initialize with given data
  ///
  /// \param[in] params Object that stores parameters
  /// \param[in] gait Gait object to hold the gait informations
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void initialize(Params &params, Gait &gait);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Destructor.
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ~FootTrajectoryGenerator() {}  // Empty constructor

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Update the next foot position, velocity and acceleration, and the foot goal position
  ///
  /// \param[in] j Foot id
  /// \param[in] targetFootstep Desired target location at the end of the swing phase
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void updateFootPosition(int const j, Vector3 const &targetFootstep);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Update the 3D desired position for feet in swing phase by using a 5-th order polynomial that lead them
  ///        to the desired position on the ground (computed by the footstep planner)
  ///
  /// \param[in] k Numero of the current loop
  /// \param[in] targetFootstep Desired target locations at the end of the swing phases
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void update(int k, MatrixN const &targetFootstep);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Compute the whole swing trajectory from the current position to the target on the ground
  ///        Mainly for debug purpose to plot the desired feet trajectory in the air
  ///
  /// \param[in] j Foot id
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  MatrixN getTrajectoryToTarget(int const j);

  MatrixN getTargetPosition() { return targetFootstep_; }  // Get the foot goal position
  MatrixN getFootPosition() { return position_; }          // Get the next foot position
  MatrixN getFootVelocity() { return velocity_; }          // Get the next foot velocity
  MatrixN getFootAcceleration() { return acceleration_; }  // Get the next foot acceleration
  MatrixN getFootJerk() { return jerk_; }                  // Get the next foot jerk
  Vector4 getT0s() { return t0s; }                         // Get the t0s for each foot
  Vector4 getTswing() { return t_swing; }                  // Get the flying period for each foot

 private:
  Gait *gait_;        // Target lock before the touchdown
  double dt_wbc;      // Time step of the whole body control
  int k_mpc;          // Number of wbc time steps for each MPC time step
  double maxHeight_;  // Apex height of the swinging trajectory
  double lockTime_;   // Target lock before the touchdown
  double vertTime_;   // Duration during which feet move only along Z when taking off and landing

  Eigen::Matrix<double, 1, 4> feet;  // Column indexes of feet currently in swing phase
  Vector4 t0s;                    // Elapsed time since the start of the swing phase movement
  Vector4 t_swing;                // Swing phase duration for each foot
  Vector4 stepHeight_;            // Height of incoming landing point

  Matrix34 targetFootstep_;  // Target for the X Y Z components

  Matrix64 Ax;  // Coefficients for the X component
  Matrix64 Ay;  // Coefficients for the Y component

  Matrix34 position_;      // Position computed in updateFootPosition
  Matrix34 velocity_;      // Velocity computed in updateFootPosition
  Matrix34 acceleration_;  // Acceleration computed in updateFootPosition
  Matrix34 jerk_;          // Jerk computed in updateFootPosition

  Matrix34 position_base_;      // Position computed in updateFootPosition in base frame
  Matrix34 velocity_base_;      // Velocity computed in updateFootPosition in base frame
  Matrix34 acceleration_base_;  // Acceleration computed in updateFootPosition in base frame
};
#endif  // TRAJGEN_H_INCLUDED
