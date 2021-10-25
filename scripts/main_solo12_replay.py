# coding: utf8

import os
import threading
from Controller import Controller
import numpy as np
import argparse
from LoggerSensors import LoggerSensors
from LoggerControl import LoggerControl
import libquadruped_reactive_walking as lqrw
import time

params = lqrw.Params()  # Object that holds all controller parameters

if params.SIMULATION:
    from PyBulletSimulator import PyBulletSimulator
else:
    import libodri_control_interface_pywrap as oci
    from solopython.utils.qualisysClient import QualisysClient

def get_input():
    keystrk = input()
    # thread doesn't continue until key is pressed
    # and so it remains alive

def put_on_the_floor(device, q_init):
    """Make the robot go to the default initial position and wait for the user
    to press the Enter key to start the main control loop

    Args:
        device (robot wrapper): a wrapper to communicate with the robot
        q_init (array): the default position of the robot
    """

    print("PUT ON THE FLOOR.")

    Kp_pos = 6.
    Kd_pos = 0.3

    device.joints.set_position_gains(Kp_pos * np.ones(12))
    device.joints.set_velocity_gains(Kd_pos * np.ones(12))
    device.joints.set_desired_positions(q_init)
    device.joints.set_desired_velocities(np.zeros(12))
    device.joints.set_torques(np.zeros(12))

    i = threading.Thread(target=get_input)
    i.start()
    print("Put the robot on the floor and press Enter")

    while i.is_alive():
        device.parse_sensor_data()
        device.send_command_and_wait_end_of_cycle(params.dt_wbc)
    
    print("Start the motion.")

def control_loop(name_interface_clone=None, des_vel_analysis=None):
    """Main function that calibrates the robot, get it into a default waiting position then launch
    the main control loop once the user has pressed the Enter key

    Args:
        name_interface_clone (string): name of the interface that will mimic the movements of the first
    """

    # Check .yaml file for parameters of the controller

    # Read replay data
    replay = np.load("path_of_replay.npz")
    replay_P = replay["P"]
    replay_D = replay["D"]
    replay_q = replay["q"]
    replay_v = replay["dq"]
    replay_tau = replay["tau_ff"]
    params.N_SIMULATION = replay_q.shape[0]

    # Enable or disable PyBullet GUI
    if not params.SIMULATION:
        params.enable_pyb_GUI = False

    # Time variable to keep track of time
    t = 0.0

    # Default position after calibration
    q_init = np.array(params.q_init.tolist())

    # Run a scenario and retrieve data thanks to the logger
    controller = Controller(params, q_init, t)

    ####

    if params.SIMULATION:
        device = PyBulletSimulator()
        qc = None
    else:
        device = oci.robot_from_yaml_file(params.config_file)
        qc = QualisysClient(ip="140.93.16.160", body_id=0)

    if params.LOGGING or params.PLOTTING:
        loggerSensors = LoggerSensors(device, qualisys=qc, logSize=params.N_SIMULATION-3)
        loggerControl = LoggerControl(params.dt_wbc, params.gait.shape[0], params.type_MPC, joystick=controller.joystick,
                                      estimator=controller.estimator, loop=controller,
                                      gait=controller.gait, statePlanner=controller.statePlanner,
                                      footstepPlanner=controller.footstepPlanner,
                                      footTrajectoryGenerator=controller.footTrajectoryGenerator,
                                      logSize=params.N_SIMULATION-3)

    # Number of motors
    nb_motors = 12

    # Initiate communication with the device and calibrate encoders
    if params.SIMULATION:
        device.Init(calibrateEncoders=True, q_init=q_init, envID=params.envID,
                    use_flat_plane=params.use_flat_plane, enable_pyb_GUI=params.enable_pyb_GUI, dt=params.dt_wbc)
    else:
        # Initialize the communication and the session.
        device.initialize(q_init[:])
        device.joints.set_zero_commands()

        device.parse_sensor_data()

        # Wait for Enter input before starting the control loop
        put_on_the_floor(device, q_init)

    # CONTROL LOOP ***************************************************
    t = 0.0
    t_max = (params.N_SIMULATION-2) * params.dt_wbc

    k_log_whole = 0
    T_whole = time.time()
    dT_whole = 0.0
    while ((not device.is_timeout) and (t < t_max) and (not controller.error)):

        # Update sensor data (IMU, encoders, Motion capture)
        device.parse_sensor_data()

        # Check that the initial position of actuators is not too far from the
        # desired position of actuators to avoid breaking the robot
        if (t <= 10 * params.dt_wbc):
            if np.max(np.abs(controller.result.q_des - device.joints.positions)) > 0.15:
                print("DIFFERENCE: ", controller.result.q_des - device.joints.positions)
                print("q_des: ", controller.result.q_des)
                print("q_mes: ", device.joints.positions)
                break

        # Set desired quantities for the actuators
        device.joints.set_position_gains(replay_P[k_log_whole, :])
        device.joints.set_velocity_gains(replay_D[k_log_whole, :])
        device.joints.set_desired_positions(replay_q[k_log_whole, :])
        device.joints.set_desired_velocities(replay_v[k_log_whole, :])
        device.joints.set_torques(params.Kff_main * replay_tau[k_log_whole, :])

        # Call logger
        if params.LOGGING or params.PLOTTING:
            loggerSensors.sample(device, qc)
            loggerControl.sample(controller.joystick, controller.estimator,
                                 controller, controller.gait, controller.statePlanner,
                                 controller.footstepPlanner, controller.footTrajectoryGenerator,
                                 controller.wbcWrapper, dT_whole)

        # Send command to the robot
        for i in range(1):
            device.send_command_and_wait_end_of_cycle(params.dt_wbc)

        t += params.dt_wbc  # Increment loop time

        dT_whole = T_whole
        T_whole = time.time()
        dT_whole = T_whole - dT_whole
        k_log_whole += 1

    # ****************************************************************

    if (t >= t_max):
        finished = True
    else:
        finished = False

    # DAMPING TO GET ON THE GROUND PROGRESSIVELY *********************
    t = 0.0
    t_max = 2.5
    while ((not device.is_timeout) and (t < t_max)):

        device.parse_sensor_data()  # Retrieve data from IMU and Motion capture

        # Set desired quantities for the actuators
        device.joints.set_position_gains(np.zeros(nb_motors))
        device.joints.set_velocity_gains(0.1 * np.ones(nb_motors))
        device.joints.set_desired_positions(np.zeros(nb_motors))
        device.joints.set_desired_velocities(np.zeros(nb_motors))
        device.joints.set_torques(np.zeros(nb_motors))

        # Send command to the robot
        device.send_command_and_wait_end_of_cycle(params.dt_wbc)
        if (t % 1) < 5e-5:
            print('IMU attitude:', device.imu.attitude_euler)
            print('joint pos   :', device.joints.positions)
            print('joint vel   :', device.joints.velocities)
            device.robot_interface.PrintStats()

        t += params.dt_wbc

    # FINAL SHUTDOWN *************************************************

    # Whatever happened we send 0 torques to the motors.
    device.joints.set_torques(np.zeros(nb_motors))
    device.send_command_and_wait_end_of_cycle(params.dt_wbc)

    if device.is_timeout:
        print("Masterboard timeout detected.")
        print("Either the masterboard has been shut down or there has been a connection issue with the cable/wifi.")

    # Plot recorded data
    if params.PLOTTING:
        loggerControl.plotAll(loggerSensors)

    # Save the logs of the Logger object
    if params.LOGGING:
        loggerControl.saveAll(loggerSensors)
        print("Log saved")

    if params.SIMULATION and params.enable_pyb_GUI:
        # Disconnect the PyBullet server (also close the GUI)
        device.Stop()

    if controller.error:
        if (controller.error_flag == 1):
            print("-- POSITION LIMIT ERROR --")
        elif (controller.error_flag == 2):
            print("-- VELOCITY TOO HIGH ERROR --")
        elif (controller.error_flag == 3):
            print("-- FEEDFORWARD TORQUES TOO HIGH ERROR --")
        print(controller.error_value)

    print("End of script")

    return finished, des_vel_analysis


def main():
    """Main function
    """

    parser = argparse.ArgumentParser(description='Playback trajectory to show the extent of solo12 workspace.')
    parser.add_argument('-c',
                        '--clone',
                        required=False,
                        help='Name of the clone interface that will reproduce the movement of the first one \
                              (use ifconfig in a terminal), for instance "enp1s0"')

    os.nice(-20)  #  Set the process to highest priority (from -20 highest to +20 lowest)
    f, v = control_loop(parser.parse_args().clone)  # , np.array([1.5, 0.0, 0.0, 0.0, 0.0, 0.0]))
    print(f, v)
    quit()


if __name__ == "__main__":
    main()
