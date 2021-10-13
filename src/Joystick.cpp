#include "qrw/Joystick.hpp"

Joystick::Joystick() : A3_(Vector6::Zero()), A2_(Vector6::Zero()),
                       v_ref_(Vector6::Zero()), v_gp_(Vector6::Zero()) {}

void Joystick::initialize(Params &params)
{
  predefined = params.predefined_vel;

  // Gamepad initialisation
  device = "/dev/input/js0";
  js = open(device, O_RDONLY | O_NONBLOCK);
  if (js == -1) {
    perror("Could not open joystick");
  }
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

int Joystick::read_event(int fd, struct js_event *event)
{
    ssize_t bytes;
    bytes = read(fd, event, sizeof(*event));
    if (bytes == sizeof(*event))
        return 0;
    /* Error, could not read full event. */
    return -1;
}

void Joystick::update_v_ref_gamepad()
{
  // Read information from gamepad client
  if (read_event(js, &event) == 0)
  {
    if (event.type == JS_EVENT_BUTTON)
    {
      if    (event.number == 9) gamepad.start = event.value;
      else if(event.number == 8) gamepad.select = event.value;
    }
    else if (event.type == JS_EVENT_AXIS)
    {
      if     (event.number == 0) gamepad.v_y   = + event.value / 32767.0;
      else if(event.number == 1) gamepad.v_x   = - event.value / 32767.0;
      else if(event.number == 3) gamepad.w_yaw = + event.value / 32767.0;
    }
  }
  // printf("Start:%d  Stop:%d  Vx:%f \tVy:%f \tWyaw:%f\n",gamepad.start,gamepad.select,gamepad.v_x,gamepad.v_y,gamepad.w_yaw);

  // Retrieve data from gamepad
  double vX = gamepad.v_x * vXScale;
  double vY = gamepad.v_y * vYScale;
  double vYaw = gamepad.w_yaw * vYawScale;
  v_gp_ << vY, vX, 0.0, 0.0, 0.0, vYaw;

  // Low pass filter to slow down the changes of velocity when moving the joysticks
  double dead_zone = 0.004;
  for (int i = 0; i < 6; i++) {
    if (v_gp_(i, 0) > -dead_zone && v_gp_(i, 0) < dead_zone) { v_gp_(i, 0) = 0.0; }
  }
  v_ref_ = alpha * v_gp_ + (1 - alpha) * v_ref_;

  // Switch to safety controller if the select key is pressed
  if (gamepad.select == 1) { stop_ = true; }
  if (gamepad.start == 1) { start_ = true; }
}