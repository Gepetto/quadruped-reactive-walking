#include "quadruped-reactive-walking/InvKin.hpp"

InvKin::InvKin(double dt_in) {

  // Parameters from the main controller
  dt = dt_in;

  // Reference position of feet
  feet_position_ref << 0.1946, 0.1946, -0.1946, -0.1946, 0.14695, -0.14695, 0.14695, -0.14695, 0.0191028, 0.0191028, 0.0191028, 0.0191028;
}

InvKin::InvKin() {}

