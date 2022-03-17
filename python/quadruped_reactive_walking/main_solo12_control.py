import threading
import time

import numpy as np

from . import quadruped_reactive_walking as qrw
from .Controller import Controller
from .tools.LoggerControl import LoggerControl
from .tools.LoggerSensors import LoggerSensors

params = qrw.Params()  # Object that holds all controller parameters

if params.SIMULATION:
    from .tools.PyBulletSimulator import PyBulletSimulator
else:
    import libodri_control_interface_pywrap as oci
    from .tools.qualisysClient import QualisysClient


def get_input():
    """
    Thread to get the input
    """
    input()


def put_on_the_floor(device, q_init):
    """
    Make the robot go to the default initial position and wait for the user
    to press the Enter key to start the main control loop

    Args:
        device (robot wrapper): a wrapper to communicate with the robot
        q_init (array): the default position of the robot
    """
    print("PUT ON THE FLOOR.")

    Kp_pos = 6.0
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

    # Slow increase till 1/4th of mass is supported by each foot
    duration_increase = 2.0  # in seconds
    steps = int(duration_increase / params.dt_wbc)
    tau_ff = np.array(
        [0.0, 0.022, 0.5, 0.0, 0.022, 0.5, 0.0, -0.022, -0.5, 0.0, -0.022, -0.5]
    )
    # tau_ff = np.array([0.0, 0.022, 0.5, 0.0, 0.022, 0.5, 0.0, 0.025, 0.575, 0.0, 0.025, 0.575])

    for i in range(steps):
        device.joints.set_torques(tau_ff * i / steps)
        device.parse_sensor_data()
        device.send_command_and_wait_end_of_cycle(params.dt_wbc)

    print("Start the motion.")


def check_position_error(device, controller):
    """
    Check the distance between current and desired position of the joints

    Args:
        device (robot wrapper): a wrapper to communicate with the robot
        controller (array): the controller storing the desired position
    """
    if np.max(np.abs(controller.result.q_des - device.joints.positions)) > 0.15:
        print("DIFFERENCE: ", controller.result.q_des - device.joints.positions)
        print("q_des: ", controller.result.q_des)
        print("q_mes: ", device.joints.positions)
        return True
    return False


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

    Args:
        des_vel_analysis (string)
    """
    if not params.SIMULATION:
        params.enable_pyb_GUI = False

    q_init = np.array(params.q_init.tolist())  # Default position after calibration

    if params.SIMULATION and (des_vel_analysis is not None):
        print(
            "Analysis: %1.1f %1.1f %1.1f"
            % (des_vel_analysis[0], des_vel_analysis[1], des_vel_analysis[5])
        )
        acceleration_rate = 0.1  # m/s^2
        steady_state_duration = 3  # s
        N_analysis = (
            int(np.max(np.abs(des_vel_analysis)) / acceleration_rate / params.dt_wbc)
            + 1
        )
        N_steady = int(steady_state_duration / params.dt_wbc)
        params.N_SIMULATION = N_analysis + N_steady

    controller = Controller(params, q_init, 0.0)

    if params.SIMULATION and (des_vel_analysis is not None):
        controller.joystick.update_for_analysis(des_vel_analysis, N_analysis, N_steady)

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
        loggerControl = LoggerControl(params, logSize=params.N_SIMULATION - 3)

    if params.SIMULATION:
        device.Init(
            calibrateEncoders=True,
            q_init=q_init,
            envID=params.envID,
            use_flat_plane=params.use_flat_plane,
            enable_pyb_GUI=params.enable_pyb_GUI,
            dt=params.dt_wbc,
        )
        # import ForceMonitor
        # myForceMonitor = ForceMonitor.ForceMonitor(device.pyb_sim.robotId, device.pyb_sim.planeId)
    else:  # Initialize the communication and the session.
        device.initialize(q_init[:])
        device.joints.set_zero_commands()
        device.parse_sensor_data()
        put_on_the_floor(device, q_init)

    # CONTROL LOOP ***************************************************

    t = 0.0
    t_max = (params.N_SIMULATION - 2) * params.dt_wbc

    t_log_whole = np.zeros((params.N_SIMULATION))
    k_log_whole = 0
    T_whole = time.time()
    dT_whole = 0.0

    log_Mddq = np.zeros((params.N_SIMULATION, 6))
    log_NLE = np.zeros((params.N_SIMULATION, 6))
    log_JcTf = np.zeros((params.N_SIMULATION, 6))
    log_Mddq_out = np.zeros((params.N_SIMULATION, 6))
    log_JcTf_out = np.zeros((params.N_SIMULATION, 6))
    while (not device.is_timeout) and (t < t_max) and (not controller.error):
        t_start_whole = time.time()

        device.parse_sensor_data()
        if controller.compute(device, qc):
            break

        if t <= 10 * params.dt_wbc and check_position_error(device, controller):
            break

        # Set desired quantities for the actuators
        device.joints.set_position_gains(controller.result.P)
        device.joints.set_velocity_gains(controller.result.D)
        device.joints.set_desired_positions(controller.result.q_des)
        device.joints.set_desired_velocities(controller.result.v_des)
        device.joints.set_torques(
            controller.result.FF * controller.result.tau_ff.ravel()
        )

        log_Mddq[k_log_whole] = controller.wbcWrapper.Mddq
        log_NLE[k_log_whole] = controller.wbcWrapper.NLE
        log_JcTf[k_log_whole] = controller.wbcWrapper.JcTf
        log_Mddq_out[k_log_whole] = controller.wbcWrapper.Mddq_out
        log_JcTf_out[k_log_whole] = controller.wbcWrapper.JcTf_out

        # Call logger
        if params.LOGGING or params.PLOTTING:
            loggerSensors.sample(device, qc)
            loggerControl.sample(
                controller.joystick,
                controller.estimator,
                controller,
                controller.gait,
                controller.statePlanner,
                controller.footstepPlanner,
                controller.footTrajectoryGenerator,
                controller.wbcWrapper,
                dT_whole,
            )

        t_end_whole = time.time()

        # myForceMonitor.display_contact_forces()

        device.send_command_and_wait_end_of_cycle(
            params.dt_wbc
        )  # Send command to the robot
        t += params.dt_wbc  # Increment loop time

        dT_whole = T_whole
        T_whole = time.time()
        dT_whole = T_whole - dT_whole

        t_log_whole[k_log_whole] = t_end_whole - t_start_whole
        k_log_whole += 1

    # ****************************************************************
    finished = t >= t_max
    damp_control(device, 12)

    if params.enable_multiprocessing:
        print("Stopping parallel process MPC")
        controller.mpc_wrapper.stop_parallel_loop()

    if params.solo3D and params.enable_multiprocessing_mip:
        print("Stopping parallel process MIP")
        controller.surfacePlanner.stop_parallel_loop()

    # ****************************************************************

    # Send 0 torques to the motors.
    device.joints.set_torques(np.zeros(12))
    device.send_command_and_wait_end_of_cycle(params.dt_wbc)

    if device.is_timeout:
        print("Masterboard timeout detected.")
        print(
            "Either the masterboard has been shut down or there has been a connection issue with the cable/wifi."
        )

    if params.LOGGING:
        loggerControl.saveAll(loggerSensors, "/home/odri/git/fanny/logs/data")

    if params.PLOTTING:
        loggerControl.plotAllGraphs(loggerSensors)

    if params.SIMULATION and params.enable_pyb_GUI:
        device.Stop()

    print("End of script")
    return finished, des_vel_analysis


if __name__ == "__main__":
    #  os.nice(-20)  #  Set the process to highest priority (from -20 highest to +20 lowest)
    f, v = control_loop()  # , np.array([1.5, 0.0, 0.0, 0.0, 0.0, 0.0]))
    print(f, v)
    quit()
