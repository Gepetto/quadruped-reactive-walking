///////////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief This is the header for Joystick class
///
/// \details This class handles computations related to the reference velocity of the robot
///
//////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef JOYSTICK_H_INCLUDED
#define JOYSTICK_H_INCLUDED

#include <fcntl.h>
#include <linux/joystick.h>
#include <stdio.h>
#include <unistd.h>

#include <chrono>

#include "qrw/Params.hpp"
#include "qrw/Types.h"

struct gamepad_struct {
  double v_x = 0.0;    // Up/down status of left pad
  double v_y = 0.0;    // Left/right status of left pad
  double v_z = 0.0;    // Up/down status of right pad
  double w_yaw = 0.0;  // Left/right status of right pad
  int start = 0;       // Status of Start button
  int select = 0;      // Status of Select button
  int cross = 0;       // Status of cross button
  int circle = 0;      // Status of circle button
  int triangle = 0;    // Status of triangle button
  int square = 0;      // Status of square button
  int L1 = 0;          // Status of L1 button
  int R1 = 0;          // Status of R1 button
};

class Joystick {
 public:
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Empty constructor
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  Joystick();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Initialize with given data
  ///
  /// \param[in] params Object that stores parameters
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void initialize(Params& params);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Destructor.
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ~Joystick() {
    if (js != -1) {
      close(js);
    }
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief update the
  ///
  /// \param[in] k Numero of the current loop
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void handle_v_switch(int k);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Update the status of the joystick, either using polynomial interpolation based on predefined profile or
  /// reading the status of the gamepad
  ///
  /// \param[in] k Numero of the current loop
  /// \param[in] velID Identifier of the current velocity profile to be able to handle different scenarios
  /// \param[in] gait_is_static If the Gait is in or is switching to a static gait
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void update_v_ref(int k, int velID, bool gait_is_static);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Check if a gamepad event occured and read its data
  ///
  /// \param[in] fd Identifier of the gamepad object
  /// \param[in] event Gamepad event object
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  int read_event(int fd, struct js_event* event);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Update the status of the joystick by reading the status of the gamepad
  ///
  /// \param[in] k Numero of the current loop
  /// \param[in] gait_is_static If the Gait is in or is switching to a static gait
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void update_v_ref_gamepad(int k, bool gait_is_static);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Update the status of the joystick using polynomial interpolation
  ///
  /// \param[in] k Numero of the current loop
  /// \param[in] velID Identifier of the current velocity profile to be able to handle different scenarios
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void update_v_ref_predefined(int k, int velID);

  Vector6 getPRef() { return p_ref_; }
  Vector6 getVRef() { return v_ref_; }
  int getJoystickCode() { return joystick_code_; }
  bool getStop() { return stop_; }
  bool getStart() { return start_; }
  bool getCross() { return gamepad.cross == 1; }
  bool getCircle() { return gamepad.circle == 1; }
  bool getTriangle() { return gamepad.triangle == 1; }
  bool getSquare() { return gamepad.square == 1; }
  bool getL1() { return gamepad.L1 == 1; }
  bool getR1() { return gamepad.R1 == 1; }

 private:
  Params* params_;

  Vector6 A3_;     // Third order coefficient of the polynomial that generates the velocity profile
  Vector6 A2_;     // Second order coefficient of the polynomial that generates the velocity profile
  Vector6 p_ref_;  // Reference position of the gamepad after low pass filter
  Vector6 p_gp_;   // Raw position reference of the gamepad
  Vector6 v_ref_;  // Reference velocity resulting of the polynomial interpolation or after low pass filter
  Vector6 v_gp_;   // Raw velocity reference of the gamepad
  Vector6 v_ref_heavy_filter_;  // Reference velocity after heavy low pass filter

  int joystick_code_ = 0;   // Code to trigger gait changes
  bool stop_ = false;       // Flag to stop the controller
  bool start_ = false;      // Flag to start the controller
  bool predefined = false;  // Flag to perform polynomial interpolation or read the gamepad

  double dt_mpc = 0.0;  // Time step of the MPC
  double dt_wbc = 0.0;  // Time step of the WBC
  int k_mpc = 0;        // Number of WBC time step for one MPC time step

  VectorNint k_switch;   // Key frames for the polynomial velocity interpolation
  Matrix6N v_switch;  // Target velocity for the key frames

  // How much the gamepad velocity and position is filtered to avoid sharp changes
  double gp_alpha_vel = 0.0;                 // Low pass filter coefficient for v_ref_ (if gamepad-controlled)
  double gp_alpha_pos = 0.0;                 // Low pass filter coefficient for p_ref_
  double gp_alpha_vel_heavy_filter = 0.002;  // Low pass filter coefficient for v_ref_heavy_filter_

  // Maximum velocity values
  double vXScale = 0.3;   // Lateral
  double vYScale = 0.5;   // Forward
  double vYawScale = 1.0;  // Rotation

  // Maximum position values
  double pRollScale = -0.32;  // Lateral
  double pPitchScale = 0.32;  // Forward
  double pHeightScale = 0.0;  // Raise/Lower the base
  double pYawScale = -0.35;   // Rotation Yaw

  // Variable to handle the automatic static/trot switching
  bool switch_static = false;   // Flag to switch to a static gait
  bool lock_gp = true;          // Flag to lock the output velocity when we are switching back to trot gait
  double lock_duration_ = 1.0;  // Duration of the lock in seconds
  std::chrono::time_point<std::chrono::system_clock> lock_time_static_;  // Timestamp of the start of the lock
  std::chrono::time_point<std::chrono::system_clock> lock_time_L1_;      // Timestamp of the latest L1 pressing

  // Gamepad client variables
  struct gamepad_struct gamepad;  // Structure that stores gamepad status
  const char* device;             // Gamepad device object
  int js;                         // Identifier of the gamepad object
  struct js_event event;          // Gamepad event object
};

#endif  // JOYSTICK_H_INCLUDED
