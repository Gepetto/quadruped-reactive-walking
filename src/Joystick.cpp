#include "qrw/Joystick.hpp"

Joystick::Joystick() : A3_(Vector6::Zero()), A2_(Vector6::Zero()),
                       v_ref_(Vector6::Zero()), v_gp_(Vector6::Zero()),
                       p_ref_(Vector6::Zero()), p_gp_(Vector6::Zero()),
                       v_ref_heavy_filter_(Vector6::Zero()) {}

void Joystick::initialize(Params &params)
{
  params_ = &params;
  dt_wbc = params.dt_wbc;
  dt_mpc = params.dt_mpc;
  k_mpc = static_cast<int>(std::round(params.dt_mpc / params.dt_wbc));
  predefined = params.predefined_vel;
  gp_alpha_vel = params.gp_alpha_vel;
  gp_alpha_pos = 0.0;
  p_ref_.setZero();
  p_ref_(2, 0) = params.h_ref;

  lock_time_L1_ = std::chrono::system_clock::now();

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

void Joystick::update_v_ref(int k, int velID, bool gait_is_static, Vector6 h_v)
{
  /* ONLY GAMEPAD CONTROL FOR NOW
  if (predefined):
    update_v_ref_predefined(k, velID);
  else:
  */
  
  update_v_ref_gamepad(k, gait_is_static, h_v);
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

void Joystick::update_v_ref_gamepad(int k, bool gait_is_static, Vector6 h_v)
{
  // Read information from gamepad client
  if (read_event(js, &event) == 0)
  {
    if (event.type == JS_EVENT_BUTTON)
    {
      switch (event.number) {
        case 9: gamepad.start = event.value; break;
        case 8: gamepad.select = event.value; break;
        case 0: gamepad.cross = event.value; break;
        case 1: gamepad.circle = event.value; break;
        case 2: gamepad.triangle = event.value; break;
        case 3: gamepad.square = event.value; break;
        case 4: gamepad.L1 = event.value; break;
        case 5: gamepad.R1 = event.value; break;
      }
    }
    else if (event.type == JS_EVENT_AXIS)
    {
      if     (event.number == 0) gamepad.v_x   = - event.value / 32767.0;
      else if(event.number == 1) gamepad.v_y   = - event.value / 32767.0;
      else if(event.number == 4) gamepad.v_z   = - event.value / 32767.0;
      else if(event.number == 3) gamepad.w_yaw = - event.value / 32767.0;
    }
  }
  // printf("Start:%d  Stop:%d  Vx:%f \tVy:%f \tWyaw:%f\n",gamepad.start,gamepad.select,gamepad.v_x,gamepad.v_y,gamepad.w_yaw);

  /*if (k < 2000)
  {
    int a = 1;
  }
  else if (k < 4000)
  {
    gamepad.v_x = 0.4;
    gamepad.v_y = 0.4;
  }
  else if (k < 7000)
  {
    gamepad.v_x = 0.0;
    gamepad.v_y = 0.0;
  }
  else if (k < 10000)
  {
    gamepad.L1 = 1;
    gamepad.v_x = std::sin(2 * M_PI * (k - 7000) / 3000);
    gamepad.v_y = std::sin(2 * M_PI * (k - 7000) / 3000);
    gamepad.v_z = 0.7 * std::sin(2 * M_PI * (k - 7000) / 3000);
    gamepad.w_yaw = 0.7 * std::sin(2 * M_PI * (k - 7000) / 3000);
  }
  else if (k < 20000)
  {
    gamepad.L1 = 0;
    gamepad.v_x = 0.0;
    gamepad.v_y = 0.5;
    gamepad.v_z = 0.0;
    gamepad.w_yaw = -0.2;
  }
  else if (k < 22000)
  {
    gamepad.v_x = 0.0;
    gamepad.v_y = 0.0;
    gamepad.w_yaw = 0.0;
  }*/

  if (gamepad.L1 == 1) { lock_time_L1_ = std::chrono::system_clock::now(); }

  // Retrieve data from gamepad for velocity
  double vX = gamepad.v_x * vXScale;
  double vY = gamepad.v_y * vYScale;
  double vYaw = gamepad.w_yaw * vYawScale;
  v_gp_ << vY, vX, 0.0, 0.0, 0.0, vYaw;

  // Dead zone to avoid gamepad noise
  double dead_zone = 0.004;
  for (int i = 0; i < 6; i++) {
    if (v_gp_(i, 0) > -dead_zone && v_gp_(i, 0) < dead_zone) { v_gp_(i, 0) = 0.0; }
  }

  // Retrieve data from gamepad for velocity
  double pRoll = gamepad.v_x * pRollScale;
  double pPitch = gamepad.v_y * pPitchScale;
  double pHeight = gamepad.v_z * pHeightScale + params_->h_ref;
  double pYaw = gamepad.w_yaw * pYawScale;
  p_gp_ << 0.0, 0.0, pHeight, pRoll, pPitch, pYaw;

  // Switch to safety controller if the select key is pressed
  if (gamepad.select == 1) { stop_ = true; }
  if (gamepad.start == 1) { start_ = true; }

  // Joystick code
  joystick_code_ = 0;
  /*
  if (gamepad.cross == 1) {joystick_code_ = 1;}
  else if (gamepad.circle == 1) {joystick_code_ = 2;}
  else if (gamepad.triangle == 1) {joystick_code_ = 3;}
  else if (gamepad.square == 1) {joystick_code_ = 4;}
  */

  if (!getL1() && (k % k_mpc == 0) && (k > static_cast<int>(std::round(1.0 / params_->dt_wbc))))
  {
    // Check joysticks value to trigger the switch between static and trot
    double v_low = 0.04;
    double v_up = 0.08;
    if (!switch_static && std::abs(v_gp_(0, 0)) < v_low && std::abs(v_gp_(1, 0)) < v_low
                       && std::abs(v_gp_(5, 0)) < v_low && std::abs(v_ref_heavy_filter_(0, 0)) < v_low 
                       && std::abs(v_ref_heavy_filter_(1, 0)) < v_low && std::abs(v_ref_heavy_filter_(5, 0)) < v_low)
    {
      switch_static = true;
      lock_gp = true;
      lock_time_static_ = std::chrono::system_clock::now();
    }
    else if (switch_static 
            && (std::abs(v_gp_(0, 0)) > v_up || std::abs(v_gp_(1, 0)) > v_up || std::abs(v_gp_(5, 0)) > v_up))
    {
      switch_static = false;
      lock_gp = true;
      lock_time_static_ = std::chrono::system_clock::now();
    }

    // Set joystick code for gait switch
    if (gait_is_static && !switch_static) {joystick_code_ = 3;}
    else if (!gait_is_static && switch_static) {joystick_code_ = 1;}

  } 

  // Lock gamepad value during switching or after L1 is pressed
  if ((lock_gp && ((std::chrono::duration<double>)(std::chrono::system_clock::now() - lock_time_static_)).count() < lock_duration_)
      || (((std::chrono::duration<double>)(std::chrono::system_clock::now() - lock_time_L1_)).count() < lock_duration_))
  {
    gp_alpha_vel = 0.0;
    gp_alpha_pos = params_->gp_alpha_pos;
  }
  else if (lock_gp) 
  {
    lock_gp = false;
    gp_alpha_vel = params_->gp_alpha_vel;
    p_ref_.setZero();
    p_ref_(2, 0) = params_->h_ref;
    gp_alpha_pos = 0.0;
  }

  // Low pass filter to slow down the changes of velocity when moving the joysticks
  v_ref_ = gp_alpha_vel * v_gp_ + (1 - gp_alpha_vel) * v_ref_;
  if (getL1() && gait_is_static) { v_ref_.setZero(); }

  // Heavily filtered joystick velocity to be used as a trigger for the switch trot/static
  v_ref_heavy_filter_ = gp_alpha_vel_heavy_filter * v_gp_ + (1 - gp_alpha_vel_heavy_filter) * v_ref_heavy_filter_;
  std::cout << v_ref_heavy_filter_.transpose() << std::endl;

  // Low pass filter to slow down the changes of position when moving the joysticks
  p_ref_ = gp_alpha_pos * p_gp_ + (1 - gp_alpha_pos) * p_ref_;
  // std::cout << p_ref_.transpose() << std::endl;
}