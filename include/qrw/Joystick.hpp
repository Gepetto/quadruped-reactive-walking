///////////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief This is the header for Joystick class
///
/// \details This class handles computations related to the reference velocity of the robot
///
//////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef JOYSTICK_H_INCLUDED
#define JOYSTICK_H_INCLUDED

#include <iostream>
#include <fstream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "qrw/Types.h"

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
  /// \brief Destructor.
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ~Joystick() {}

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

  Vector6 getVRef() { return v_ref_; }
  int getJoystickCode() { return joystick_code_; }
  bool getStop() { return stop_; }

 private:
  Vector6 A3_;     // Third order coefficient of the polynomial that generates the velocity profile
  Vector6 A2_;     // Second order coefficient of the polynomial that generates the velocity profile
  Vector6 v_ref_;  // Reference velocity resulting of the polynomial interpolation
  int joystick_code_ = 0;
  bool stop_ = false;
};

#endif  // JOYSTICK_H_INCLUDED
