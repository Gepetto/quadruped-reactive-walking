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
#include "qrw/WbcWrapper.hpp"
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
  /// \param[in] robot Interface to communicate with the robot
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  // void compute(std::shared_ptr<odri_control_interface::Robot> robot);
  void compute(FakeRobot* robot);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Initialization of some robot parameters (mass, inertia, ...) based on urdf
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void init_robot(Params& params);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Perform a security check before sending commands to the robot
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void security_check();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Check if the Start key of the joystick has been pressed
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  bool getStart() { return joystick.getStart(); }

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Update the status of the joystick (buttons, pads)
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void update_gamepad() { joystick.update_v_ref_gamepad(0, false); }

  // Commands to be sent to the robot
  Vector12 P;       // Proportional gains
  Vector12 D;       // Derivative gains
  Vector12 q_des;   // Desired joint positions
  Vector12 v_des;   // Desired joint velocities
  Vector12 tau_ff;  // Desired joint torques
  Vector12 FF;      // Torque gains (0 < FF < 1)

  // Control info
  Params* params_;       // Object that stores parameters
  bool error;            // Error flag to stop the controller
  int error_flag;        // Value depends on what set the error flag to true
  Vector12 error_value;  // Store data about the error

 private:
  int k;          // Number of wbc time steps since the start of the controller
  int k_mpc;      // Number of wbc time steps for each MPC time step
  double h_ref_;  // Reference height of the base

  // Classes of the different control blocks
  Joystick joystick;                                // Joystick control block
  Estimator estimator;                              // Estimator control block
  Gait gait;                                        // Gait control block
  FootstepPlanner footstepPlanner;                  // Footstep planner control block
  StatePlanner statePlanner;                        // State planner control block
  MpcWrapper mpcWrapper;                            // MPC Wrapper control block
  FootTrajectoryGenerator footTrajectoryGenerator;  // Foot Trajectory Generator control block
  WbcWrapper wbcWrapper;                            // Whole body control Wrapper control block

  // Filters
  Filter filter_mpc_q = Filter();     // Filter object for base position
  Filter filter_mpc_v = Filter();     // Filter object for base velocity
  Filter filter_mpc_vref = Filter();  // Filter object for base reference velocity
  Vector18 q_filt_mpc;                // Filtered base position vector (with joints)
  Vector6 h_v_filt_mpc;               // Filtered base velocity vector (with joints)
  Vector6 vref_filt_mpc;              // Filtered reference base velocity vector

  // Various
  Matrix34 o_targetFootstep;  // Target location of footsteps in ideal world
  Vector18 q_wbc;             // Position vector for whole body control
  Vector18 dq_wbc;            // Velocity vector for whole body control
  Vector12 xgoals;            // Base position, orientation and velocity references for whole body control
  Matrix3 hRb;                // Rotation matrix between horizontal and base frame
  Vector6 p_ref_;             // Position, orientation reference vector from the joystick
  Vector12 f_mpc;             // Contact forces desired by the MPC
};

#endif  // CONTROLLER_H_INCLUDED
