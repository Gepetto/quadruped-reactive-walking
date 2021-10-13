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
#include <Eigen/Core>
#include <Eigen/Dense>
#include "qrw/Types.h"
#include "qrw/Params.hpp"

struct gamepad_struct
{
	double v_x = 0.0;
	double v_y = 0.0;
	double w_yaw = 0.0;
	int start = 0;
	int select = 0;
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

  void update_v_ref(int k, int velID);
  int read_event(int fd, struct js_event *event);
  void update_v_ref_gamepad();

  Vector6 getVRef() { return v_ref_; }
  int getJoystickCode() { return joystick_code_; }
  bool getStop() { return stop_; }

 private:
  Vector6 A3_;     // Third order coefficient of the polynomial that generates the velocity profile
  Vector6 A2_;     // Second order coefficient of the polynomial that generates the velocity profile
  Vector6 v_ref_;  // Reference velocity resulting of the polynomial interpolation
  Vector6 v_gp_;
  int joystick_code_ = 0;
  bool stop_ = false;
  bool predefined = false;

  // How much the gamepad velocity is filtered to avoid sharp changes
  double alpha = 0.001;

  // Maximum velocity values
  double vXScale = 0.6;  // Lateral
  double vYScale = 1.0;  // Forward
  double vYawScale = 1.2;  // Rotation

  // Gamepad client variables
  struct gamepad_struct gamepad;
  const char *device;
  int js;
  struct js_event event;

};

#endif  // JOYSTICK_H_INCLUDED
