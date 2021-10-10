#include "qrw/Joystick.hpp"

Joystick::Joystick() : A3_(Vector6::Zero()), A2_(Vector6::Zero()),
                       v_ref_(Vector6::Zero()), v_gp_(Vector6::Zero()) {}

void Joystick::initialize(Params &params)
{
  predefined = params.predefined_vel;
}

VectorN Joystick::handle_v_switch(double k, VectorN const& k_switch, MatrixN const& v_switch) {
  int i = 1;
  while ((i < k_switch.rows()) && k_switch[i] <= k) {
    i++;
  }
  if (i != k_switch.rows()) {
    double ev = k - k_switch[i - 1];
    double t1 = k_switch[i] - k_switch[i - 1];
    A3_ = 2 * (v_switch.col(i - 1) - v_switch.col(i)) / pow(t1, 3);
    A2_ = (-3.0 / 2.0) * t1 * A3_;
    v_ref_ = v_switch.col(i - 1) + A2_ * pow(ev, 2) + A3_ * pow(ev, 3);
  }
  return v_ref_;
}

void Joystick::update_v_ref(int k, int velID)
{
  /* ONLY GAMEPAD CONTROL FOR NOW
  if (predefined):
    update_v_ref_predefined(k, velID);
  else:
  */
  
  update_v_ref_gamepad();
}

void Joystick::update_v_ref_gamepad()
{
  // Create the gamepad client
  // TODO

  // Retrieve data from gamepad
  double vX = 0.0 * vXScale;
  double vY = 0.0 * vYScale;
  double vYaw = 0.0 * vYawScale;
  v_gp_ << vY, vX, 0.0, 0.0, 0.0, vYaw;

  // Low pass filter to slow down the changes of velocity when moving the joysticks
  double dead_zone = 0.004;
  for (int i = 0; i < 6; i++) {
    if (v_gp_(i, 0) > -dead_zone && v_gp_(i, 0) < dead_zone) { v_gp_(i, 0) = 0.0; }
  }
  v_ref_ = alpha * v_gp_ + (1 - alpha) * v_ref_;

  // Switch to safety controller if the Back key is pressed
  // if (gp.backButton.value) { stop = true; }  // TODO
}