///////////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief This is the header for Gait class
///
/// \details Planner that outputs current and future locations of footsteps, the
/// reference
///          trajectory of the base and the position, velocity, acceleration
///          commands for feet in swing phase based on the reference velocity
///          given by the user and the current position/velocity of the base
///
//////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef GAIT_H_INCLUDED
#define GAIT_H_INCLUDED

#include "qrw/Params.hpp"

// Order of feet/legs: FL, FR, HL, HR

class Gait {
 public:
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Empty constructor
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  Gait();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Destructor.
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ~Gait() {}

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Initializer
  ///
  /// \param[in] params Object that stores parameters
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void initialize(Params& params);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Compute the total duration of a swing phase or a stance phase based
  ///        on the content of the gait matrix
  ///
  /// \param[in] i Considered phase (row of the gait matrix)
  /// \param[in] j Considered foot (col of the gait matrix)
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  double getPhaseDuration(int i, int j);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Compute the duration of a swing phase or a stance phase since its
  /// start based
  ///        on the content of the gait matrix
  /// \details We suppose that the phase starts after the start of past gait
  /// matrix
  ///
  /// \param[in] i Considered phase (row of the gait matrix)
  /// \param[in] j Considered foot (col of the gait matrix)
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  double getElapsedTime(int i, int j);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Compute the remaining duration of a swing phase or a stance phase
  /// based
  ///        on the content of the gait matrix
  /// \details We suppose that the end of the phase is reached before the end of
  /// desiredGait matrix
  ///
  /// \param[in] i Considered phase (row of the gait matrix)
  /// \param[in] j Considered foot (col of the gait matrix)
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  double getRemainingTime(int i, int j);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Compute the total duration of a gait phase based on the content
  ///        of the gait matrix
  ///
  /// \param[in] i Considered phase (row of the gait matrix)
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  double getPhaseDuration(int i);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Compute the duration of a gait phase based on the content of the
  /// gait matrix \details We suppose that the phase starts after the start of
  /// past gait matrix
  ///
  /// \param[in] i Considered phase (row of the gait matrix)
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  double getElapsedTime(int i);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Compute the remaining duration of a gait phase phase based on the
  /// content
  ///        of the gait matrix
  /// \details We suppose that the end of the phase is reached before the end of
  /// desiredGait matrix
  ///
  /// \param[in] i Considered phase (row of the gait matrix)
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  double getRemainingTime(int i);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Handle the joystick code to trigger events (change of gait for
  /// instance) \details We suppose that the end of the phase is reached before
  /// the end of the desiredGait
  ///          matrix and the phase starts after the start of past gait matrix
  ///
  /// \param[in] k Numero of the current loop
  /// \param[in] code Integer to trigger events with the joystick
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  bool changeGait(int const k, int const code);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief  Move one step further in the gait cycle
  ///
  /// \details Decrease by 1 the number of remaining step for the current phase
  /// of the gait
  ///          Transfer current gait phase into past gait matrix
  ///          Insert future desired gait phase at the end of the gait matrix
  ///
  /// \param[in] k Numero of the current loop
  /// \param[in] joystickCode Integer to trigger events with the joystick
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void update(int const k, int const joystickCode);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Move one step further into the gait
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void rollGait();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Directly set the value of newPhase_ variable
  ///
  /// \param[in] value Value the newPhase_ variable should take
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void setNewPhase(bool const& value) { newPhase_ = value; };

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Raise a flag to indicate that the contact of the i-th foot is late
  ///
  /// \param[in] i Index of the late foot (0, 1, 2 or 3)
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void setLate(int i) { isLate_[i] = true; }

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Update the current gait matrix externally (directly set the gait
  /// matrix)
  ///
  /// \param[in] gaitMatrix Gait matrix that should be used for the current gait
  /// matrix
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void setCurrentGait(MatrixN4 const& gaitMatrix);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Update the past gait matrix externally (directly set the gait
  /// matrix)
  ///
  /// \param[in] gaitMatrix Gait matrix that should be used for the past gait
  /// matrix
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void setPastGait(MatrixN4 const& gaitMatrix);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Update the desired gait matrix externally (directly set the gait
  /// matrix)
  ///
  /// \param[in] gaitMatrix Gait matrix that should be used for the desired gait
  /// matrix
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void setDesiredGait(MatrixN4 const& gaitMatrix);

  MatrixN4 getPastGait() { return pastGait_; }
  MatrixN4 getCurrentGait() { return currentGait_; }
  double getCurrentGaitCoeff(int i, int j) { return currentGait_(i, j); }
  MatrixN4 getDesiredGait() { return desiredGait_; }
  bool getIsStatic() { return isStatic_; }
  bool isNewPhase() { return newPhase_; }
  bool isLate(int i) { return isLate_[i]; }

 private:
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Create a static gait with all legs in stance phase
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void createStatic();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief  Create a slow walking gait, raising and moving only one foot at a
  /// time
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void createWalk();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Create a trot gait with diagonaly opposed legs moving at the same
  /// time
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void createTrot();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Create a trot gait with diagonaly opposed legs moving at the same
  /// time and some
  ///        4-stance phases
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void createWalkingTrot();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Create a pacing gait with legs on the same side (left or right)
  /// moving at the same time
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void createPacing();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Create a bounding gait with legs on the same side (front or hind)
  /// moving at the same time
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void createBounding();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Create a transverse gallop gait
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void createTransverseGallop();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Create a custom gallop gait
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void createCustomGallop();

  MatrixN4 pastGait_;     // Past gait
  MatrixN4 currentGait_;  // Current and future gait
  MatrixN4 desiredGait_;  // Future desired gait

  double dt_;  // Time step of the contact sequence (time step of the MPC)
  int k_mpc_;  // Number of wbc time steps for each MPC time step
  int nRows_;  // number of rows in the gait matrix

  bool newPhase_;  // Flag to indicate that the contact status has changed
  bool isStatic_;  // Flag to indicate that all feet are in an infinite stance
                   // phase
  int switchToGait_;  // Memory to store joystick code if the user wants to
                      // change the gait pattern
  bool isLate_[4] = {false, false, false,
                     false};  // Flags to indicate late contacts
};

#endif  // GAIT_H_INCLUDED
