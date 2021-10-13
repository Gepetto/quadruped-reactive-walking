///////////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief This is the header for Controller class
///
/// \details Handle the communication between the blocks of the control architecture
///
//////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CONTROLLER_H_INCLUDED
#define CONTROLLER_H_INCLUDED

#include <odri_control_interface/robot.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include "qrw/FakeRobot.hpp"
#include "pinocchio/math/rpy.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include "qrw/Types.h"
#include "qrw/Params.hpp"
#include "qrw/Joystick.hpp"
#include "qrw/Estimator.hpp"
#include "qrw/Gait.hpp"
#include "qrw/FootstepPlanner.hpp"
#include "qrw/StatePlanner.hpp"
#include "qrw/MpcWrapper.hpp"
#include "qrw/FootTrajectoryGenerator.hpp"
#include "qrw/QPWBC.hpp"
#include "qrw/Filter.hpp"

class Controller {
 public:
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Constructor
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  Controller();

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
  /// \brief Destructor
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ~Controller() {}  // Empty destructor

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Compute one iteration of control loop
  ///
  /// \param[in] robot Pointer to the robot interface
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void compute(std::shared_ptr<odri_control_interface::Robot> robot);
  // void compute(FakeRobot *robot);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Initialization of some robot parameters (mass, inertia, ...) based on urdf
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void init_robot();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Perform a security check before sending commands to the robot
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void security_check();

  bool getStart() { return joystick.getStart(); }
  void update_gamepad() { joystick.update_v_ref_gamepad();}

  // Commands to be sent to the robot
  Vector12 P;
  Vector12 D;
  Vector12 q_des;
  Vector12 v_des;
  Vector12 tau_ff;
  Vector12 FF;

  // Control info
  Params* params_;  // Params object to store parameters
  bool error;
  int error_flag;
  Vector12 error_value;
  
 private:
  
  int k;
  int k_mpc;

  // Classes of the different control blocks
  Joystick joystick;
  Estimator estimator;
  Gait gait;
  FootstepPlanner footstepPlanner;
  StatePlanner statePlanner;
  MpcWrapper mpcWrapper;
  FootTrajectoryGenerator footTrajectoryGenerator;
  WbcWrapper wbcWrapper;

  // Filters
  Filter filter_mpc_q = Filter();
  Filter filter_mpc_v = Filter();
  Filter filter_mpc_vref = Filter();
  Vector18 q_filt_mpc;
  Vector6 h_v_filt_mpc;
  Vector6 vref_filt_mpc;

  // Various
  Matrix34 o_targetFootstep;
  Vector18 q_wbc;
  Vector18 dq_wbc;
  Vector12 xgoals;

};

#endif  // CONTROLLER_H_INCLUDED
