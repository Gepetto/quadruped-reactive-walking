import os
import threading
import time

import numpy as np

from . import quadruped_reactive_walking as qrw
from .tools.LoggerSensors import LoggerSensors

params = qrw.Params()  # Object that holds all controller parameters

if params.SIMULATION:
    from tools.PyBulletSimulator import PyBulletSimulator
else:
    import libodri_control_interface_pywrap as oci
    from tools.qualisysClient import QualisysClient


def get_input():
    input()


def put_on_the_floor(device, q_init):
    """Make the robot go to the default initial position and wait for the user
    to press the Enter key to start the main control loop

    Args:
        device (robot wrapper): a wrapper to communicate with the robot
        q_init (array): the default position of the robot
    """
    print("PUT ON THE FLOOR.")

    Kp_pos = 3.0
    Kd_pos = 0.3

    device.joints.set_position_gains(Kp_pos * np.ones(12))
    device.joints.set_velocity_gains(Kd_pos * np.ones(12))
    device.joints.set_desired_positions(q_init)
    device.joints.set_desired_velocities(np.zeros(12))
    device.joints.set_torques(np.zeros(12))

    print("Init")
    print(q_init)

    i = threading.Thread(target=get_input)
    i.start()
    print("Put the robot on the floor and press Enter")

    while i.is_alive():
        device.parse_sensor_data()
        device.send_command_and_wait_end_of_cycle(params.dt_wbc)

    print("Start the motion.")


def damp_control(device, nb_motors):
    """
    Damp the control during 2.5 seconds

    Args:
        device  (robot wrapper): a wrapper to communicate with the robot
        nb_motors (int): number of motors
    """
    t = 0.0
    t_max = 2.5
    while (not device.is_timeout) and (t < t_max):
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
            print("IMU attitude:", device.imu.attitude_euler)
            print("joint pos   :", device.joints.positions)
            print("joint vel   :", device.joints.velocities)
            device.robot_interface.PrintStats()

        t += params.dt_wbc


def control_loop(des_vel_analysis=None):
    """
    Main function that calibrates the robot, get it into a default waiting position then launch
    the main control loop once the user has pressed the Enter key
    """
    # Read replay data
    # replay = np.load("/home/odri/git/abonnefoy/Motion/Logs/push_up.npz")
    # replay = np.load("/home/odri/git/abonnefoy/Motion/Logs/one_leg_position.npz")
    # replay = np.load("/home/odri/git/abonnefoy/Motion/Logs/push_up_position.npz")
    replay = np.load("/home/odri/git/abonnefoy/Motion/Logs/full_push_up.npz")

    replay_q = replay["q"][7:, 1:].transpose().copy()
    replay_v = replay["v"][6:, 1:].transpose().copy()
    replay_tau = replay["tau"].transpose().copy()
    params.N_SIMULATION = replay_q.shape[0]
    N = replay_q.shape[0]
    replay_P = 6.0 * np.ones((N, 12))  # replay["P"]
    replay_D = 0.3 * np.ones((N, 12))  # replay["D"]

    # 0.09547498,  1.25215899, -2.01927128, -0.09552912,  1.25175677,
    # -2.01855657,  0.52255345,  1.34166507,  0.11203987, -0.52311626,
    #  1.34141582,  0.11178296]

    # Enable or disable PyBullet GUI
    if not params.SIMULATION:
        params.enable_pyb_GUI = False

    # Time variable to keep track of time
    t = 0.0

    # Default position after calibration
    q_init = (replay_q[0, :]).copy()
    params.q_init = q_init.tolist()

    if params.SIMULATION:
        device = PyBulletSimulator()
        qc = None
    else:
        device = oci.robot_from_yaml_file(params.config_file)
        qc = QualisysClient(ip="140.93.16.160", body_id=0)

    if params.LOGGING or params.PLOTTING:
        loggerSensors = LoggerSensors(
            device, qualisys=qc, logSize=params.N_SIMULATION - 3
        )

    # Initiate communication with the device and calibrate encoders
    if params.SIMULATION:
        device.Init(
            calibrateEncoders=True,
            q_init=q_init,
            envID=params.envID,
            use_flat_plane=params.use_flat_plane,
            enable_pyb_GUI=params.enable_pyb_GUI,
            dt=params.dt_wbc,
        )
    else:
        # Initialize the communication and the session.
        device.initialize(q_init[:])
        device.joints.set_zero_commands()

        device.parse_sensor_data()

        # Wait for Enter input before starting the control loop
        put_on_the_floor(device, q_init)

    # CONTROL LOOP ***************************************************
    t = 0.0
    t_max = (params.N_SIMULATION - 2) * params.dt_wbc

    k_log_whole = 0
    T_whole = time.time()
    dT_whole = 0.0
    np.set_printoptions(precision=2, linewidth=300)
    while (not device.is_timeout) and (t < t_max):

        # Update sensor data (IMU, encoders, Motion capture)
        device.parse_sensor_data()

        # Check that the initial position of actuators is not too far from the
        # desired position of actuators to avoid breaking the robot
        if t <= 10 * params.dt_wbc:
            if (
                np.max(np.abs(replay_q[k_log_whole, :] - device.joints.positions))
                > 0.15
            ):
                print(
                    "DIFFERENCE: ", replay_q[k_log_whole, :] - device.joints.positions
                )
                print("q_des: ", replay_q[k_log_whole, :])
                print("q_mes: ", device.joints.positions)
                break

        # Set desired quantities for the actuators
        device.joints.set_position_gains(replay_P[k_log_whole, :])
        device.joints.set_velocity_gains(replay_D[k_log_whole, :])
        device.joints.set_desired_positions(replay_q[k_log_whole, :])
        device.joints.set_desired_velocities(replay_v[k_log_whole, :])
        device.joints.set_torques(1.0 * replay_tau[k_log_whole, :])

        # Call logger
        if params.LOGGING or params.PLOTTING:
            loggerSensors.sample(device, qc)

        # Send command to the robot
        device.send_command_and_wait_end_of_cycle(params.dt_wbc)

        t += params.dt_wbc

        dT_whole = T_whole
        T_whole = time.time()
        dT_whole = T_whole - dT_whole
        k_log_whole += 1

    # ****************************************************************

    finished = t >= t_max
    damp_control(device, 12)

    device.joints.set_torques(np.zeros(12))
    device.send_command_and_wait_end_of_cycle(params.dt_wbc)

    if device.is_timeout:
        print("Masterboard timeout detected.")
        print(
            "Either the masterboard has been shut down or there has been a connection issue with the cable/wifi."
        )

    if params.LOGGING:
        loggerSensors.saveAll()
        print("Log saved")

    if params.SIMULATION and params.enable_pyb_GUI:
        device.Stop()

    return finished, des_vel_analysis


if __name__ == "__main__":
    os.nice(
        -20
    )  #  Set the process to highest priority (from -20 highest to +20 lowest)
    f, v = control_loop()  # , np.array([1.5, 0.0, 0.0, 0.0, 0.0, 0.0]))
    print("End of script")
    print(f, v)
    quit()
