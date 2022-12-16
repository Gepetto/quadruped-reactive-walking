#include <odri_control_interface/calibration.hpp>
#include <odri_control_interface/robot.hpp>
#include <odri_control_interface/utils.hpp>

#include "qrw/Controller.hpp"
#include "qrw/FakeRobot.hpp"
#include "qrw/Params.hpp"
#include "qrw/Types.h"
#ifdef QRW_WITH_MQTT
#include "qrw/mqtt-interface.hpp"
#endif

using namespace odri_control_interface;

#include <chrono>
#include <iostream>
#include <stdexcept>

////////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief Make the robot go to the default initial position and wait for the
/// user to press the Start key to start the main control loop
///
/// \param[in] robot Interface to communicate with the robot
/// \param[in] q_init The default position of the robot
/// \param[in] params Object that stores parameters
/// \param[in] controller Main controller object
///
////////////////////////////////////////////////////////////////////////////////////////////////
// int put_on_the_floor(std::shared_ptr<Robot> robot, Vector12 const& q_init,
// Params & params, Controller & controller)
int put_on_the_floor(FakeRobot* robot, Vector12 const& q_init, Params& params,
                     Controller& controller) {
  printf("PUT ON THE FLOOR\n");

  double Kp_pos = 6.;
  double Kd_pos = 0.3;

  robot->joints->SetPositionGains(Kp_pos * Vector12::Ones());
  robot->joints->SetVelocityGains(Kd_pos * Vector12::Ones());
  robot->joints->SetDesiredPositions(q_init);
  robot->joints->SetDesiredVelocities(Vector12::Zero());
  robot->joints->SetTorques(Vector12::Zero());

  while (!controller.getStart()) {
    controller.update_gamepad();
    robot->ParseSensorData();
    robot->SendCommandAndWaitEndOfCycle(params.dt_wbc);
  }

  // Slow increase till 1/4th of mass is supported by each foot
  /*
  double duration_increase = 2.0;  // in seconds
  double steps = std::round(duration_increase / params.dt_wbc);
  Vector12 tau_ff;
  tau_ff << -0.25, 0.022, 0.5, 0.25, 0.022, 0.5, -0.28, 0.025, 0.575, 0.28,
  0.025, 0.575; for (double i = 0; i < steps; i++) {
    robot->joints->SetTorques(tau_ff * i / steps);
    robot->ParseSensorData();
    robot->SendCommandAndWaitEndOfCycle(params.dt_wbc);
  }
  */

  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief Main function that encompasses the whole control architecture
///
////////////////////////////////////////////////////////////////////////////////////////////////
int main() {
  nice(-20);  // Give the process a high priority.

  // Object that holds all controller parameters
  Params params = Params();

  // Define the robot from a yaml file.
  // std::shared_ptr<Robot> robot = RobotFromYamlFile(CONFIG_SOLO12_YAML);
  FakeRobot* robot = new FakeRobot();

  // Store initial position data.
  Vector12 q_init = Vector12(params.q_init.data());

  // Initialization of variables
  Controller controller;  // Main controller
  controller.initialize(
      params);  // Update urdf dependent parameters (mass, inertia, ...)
  std::thread parallel_thread(
      parallel_loop);  // spawn new thread that runs MPC in parallel
  int k_loop = 0;

  // Initialize the communication, session, joints, wait for motors to be ready
  // and run the joint calibration.
  robot->Initialize(q_init);
  robot->joints->SetZeroCommands();
  robot->ParseSensorData();

#ifdef QRW_WITH_MQTT
  // Initialize MQTT
  MqttInterface mqtt_interface;
  std::thread mqtt_thread(&MqttInterface::start, &mqtt_interface);
  std::string mqtt_ready = "ready";
  std::string mqtt_running = "running";
  std::string mqtt_stopping = "stopping";
  std::string mqtt_done = "done";
  std::string mqtt_placeholder = "TODO";
  mqtt_interface.setStatus(mqtt_ready);
  mqtt_interface.set(robot->powerboard->GetCurrent(),
                     robot->powerboard->GetVoltage(),
                     robot->powerboard->GetEnergy(), mqtt_placeholder);
#endif
  // Wait for Enter input before starting the control loop
  put_on_the_floor(robot, q_init, params, controller);

#ifdef QRW_WITH_MQTT
  mqtt_interface.setStatus(mqtt_running);
#endif
  // std::chrono::time_point<std::chrono::steady_clock>
  // t_log[params.N_SIMULATION - 2]; Main loop
  while ((!robot->IsTimeout()) && (k_loop < params.N_SIMULATION - 2) &&
         (!controller.error)) {
#ifdef QRW_WITH_MQTT
    if (mqtt_interface.getStop()) break;
#endif
    // t_log[k_loop] = std::chrono::steady_clock::now();

    // Parse sensor data from the robot
    robot->ParseSensorData();

    // Run the main controller
    controller.compute(robot);

    // Check that the initial position of actuators is not too far from the
    // desired position of actuators to avoid breaking the robot
    if (k_loop <= 10) {
      Vector12 pos = robot->joints->GetPositions();
      if ((controller.q_des - pos).cwiseAbs().maxCoeff() > 0.15) {
        std::cout << "DIFFERENCE: " << (controller.q_des - pos).transpose()
                  << std::endl;
        std::cout << "q_des: " << controller.q_des.transpose() << std::endl;
        std::cout << "q_mes: " << pos.transpose() << std::endl;
        break;
      }
    }

    // Send commands to the robot
    robot->joints->SetPositionGains(controller.P);
    robot->joints->SetVelocityGains(controller.D);
    robot->joints->SetDesiredPositions(controller.q_des);
    robot->joints->SetDesiredVelocities(controller.v_des);
    robot->joints->SetTorques((controller.FF).cwiseProduct(controller.tau_ff));

    // Checks if the robot is in error state (that is, if any component
    // returns an error). If there is an error, the commands to send
    // are changed to send the safety control.
    robot->SendCommandAndWaitEndOfCycle(params.dt_wbc);

    k_loop++;
    if (k_loop % 1000 == 0) {
#ifdef QRW_WITH_MQTT
      mqtt_interface.set(robot->powerboard->GetCurrent(),
                         robot->powerboard->GetVoltage(),
                         robot->powerboard->GetEnergy(), mqtt_placeholder);
#endif
      std::cout << "Joints: ";
      robot->joints->PrintVector(robot->joints->GetPositions());
      std::cout << std::endl;
    }
  }

  // DAMPING TO GET ON THE GROUND PROGRESSIVELY *********************
#ifdef QRW_WITH_MQTT
  mqtt_interface.setStatus(mqtt_stopping);
#endif
  double t = 0.0;
  double t_max = 2.5;
  while ((!robot->IsTimeout()) && (t < t_max)) {
    // Parse sensor data from the robot
    robot->ParseSensorData();

    // Send commands to the robot
    robot->joints->SetPositionGains(Vector12::Zero());
    robot->joints->SetVelocityGains(0.1 * Vector12::Ones());
    robot->joints->SetDesiredPositions(Vector12::Zero());
    robot->joints->SetDesiredVelocities(Vector12::Zero());
    robot->joints->SetTorques(Vector12::Zero());

    // Checks if the robot is in error state (that is, if any component
    // returns an error). If there is an error, the commands to send
    // are changed to send the safety control.
    robot->SendCommandAndWaitEndOfCycle(params.dt_wbc);

    t += params.dt_wbc;
  }
  // FINAL SHUTDOWN *************************************************

#ifdef QRW_WITH_MQTT
  mqtt_interface.setStatus(mqtt_done);
#endif

  // Whatever happened we send 0 torques to the motors.
  robot->joints->SetZeroCommands();
  robot->SendCommandAndWaitEndOfCycle(params.dt_wbc);

  if (robot->IsTimeout()) {
    printf("Masterboard timeout detected.");
    printf(
        "Either the masterboard has been shut down or there has been a "
        "connection issue with the cable/wifi.");
  }

  // Close parallel thread
  stop_thread();
  parallel_thread.join();
  std::cout << "Parallel thread closed" << std::endl;

#ifdef QRW_WITH_MQTT
  mqtt_interface.stop();
  mqtt_thread.join();
#endif

  /*int duration_log [params.N_SIMULATION-2];
  for (int i = 0; i < params.N_SIMULATION-3; i++)
  {
      duration_log[i] =
  static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(t_log[i+1]
  - t_log[i]).count());
  }
  for (int i = 0; i < params.N_SIMULATION-3; i++)
  {
      std::cout <<
  std::chrono::duration_cast<std::chrono::microseconds>(t_log[i+1] -
  t_log[i]).count() << ", ";
  }
  std::cout << std::endl;*/

  return 0;
}
