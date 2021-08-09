#include "qrw/Joystick.hpp"

Joystick::Joystick() : A3_(Vector6::Zero()), A2_(Vector6::Zero()), v_ref_(Vector6::Zero()) {}

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
