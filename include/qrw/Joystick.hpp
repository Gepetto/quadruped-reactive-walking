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
#include <stdio.h>
#include <unistd.h>
#include <linux/joystick.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "qrw/Types.h"
#include "qrw/Params.hpp"

struct gamepad_struct
{
	double v_x = 0.0;
	double v_y = 0.0;
  double v_z = 0.0;
	double w_yaw = 0.0;
	int start = 0;
	int select = 0;
  int cross = 0;
  int circle = 0;
  int triangle = 0;
  int square = 0;
  int L1 = 0;
  int R1 = 0;
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
  ~Joystick() { if (js != -1) {close(js);} }

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Compute the remaining and total duration of a swing phase or a stance phase based
  ///        on the content of the gait matrix
  ///
  /// \param[in] k Numero of the current loop
  /// \param[in] k_switch Information about the position of key frames
  /// \param[in] v_switch Information about the desired velocity for each key frame
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  VectorN handle_v_switch(double k, VectorN const& k_switch, MatrixN const& v_switch);

  void update_v_ref(int k, int velID, bool gait_is_static, Vector6 h_v);
  int read_event(int fd, struct js_event *event);
  void update_v_ref_gamepad(int k, bool gait_is_static, Vector6 h_v);

  Vector6 getPRef() { return p_ref_; }
  Vector6 getVRef() { return v_ref_; }
  int getJoystickCode() { return joystick_code_; }
  bool getStop() { return stop_; }
  bool getStart() { return start_; }
  bool getCross() { return gamepad.cross == 1;}
  bool getCircle() { return gamepad.circle == 1;}
  bool getTriangle() { return gamepad.triangle == 1;}
  bool getSquare() { return gamepad.square == 1;}
  bool getL1() { return gamepad.L1 == 1;}
  bool getR1() { return gamepad.R1 == 1;}

 private:
  Params* params_;

  Vector6 A3_;     // Third order coefficient of the polynomial that generates the velocity profile
  Vector6 A2_;     // Second order coefficient of the polynomial that generates the velocity profile
  Vector6 p_ref_;  // Reference position
  Vector6 p_gp_;
  Vector6 v_ref_;  // Reference velocity resulting of the polynomial interpolation
  Vector6 v_gp_;

  int joystick_code_ = 0;
  bool stop_ = false;
  bool start_ = false;
  bool predefined = false;

  double dt_mpc = 0.0;
  double dt_wbc = 0.0;
  int k_mpc = 0;

  // How much the gamepad velocity and position is filtered to avoid sharp changes
  double gp_alpha_vel = 0.0;
  double gp_alpha_pos = 0.0;

  // Maximum velocity values
  double vXScale = 0.3;  // Lateral
  double vYScale = 0.5;  // Forward
  double vYawScale = 1.0;  // Rotation

  // Maximum position values
  double pRollScale = -0.32;  // Lateral
  double pPitchScale = -0.28;  // Forward
  double pHeightScale = 0.025;  // Forward
  double pYawScale = -0.35;  // Rotation

  // Variable to handle the automatic static/trot switching
  bool switch_static = false;
  bool lock_gp = true;
  double lock_duration_ = 1.0;
  std::chrono::time_point<std::chrono::system_clock> lock_time_static_;
  std::chrono::time_point<std::chrono::system_clock> lock_time_L1_;

  // Gamepad client variables
  struct gamepad_struct gamepad;
  const char *device;
  int js;
  struct js_event event;

};

#endif  // JOYSTICK_H_INCLUDED
