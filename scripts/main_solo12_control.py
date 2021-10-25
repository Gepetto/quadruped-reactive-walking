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


def clone_movements(name_interface_clone, q_init, cloneP, cloneD, cloneQdes, cloneDQdes, cloneRunning, cloneResult):

    print("-- Launching clone interface --")

    print(name_interface_clone, params.dt_wbc)
    clone = Solo12(name_interface_clone, dt=params.dt_wbc)
    clone.Init(calibrateEncoders=True, q_init=q_init)

    while cloneRunning.value and not clone.hardware.IsTimeout():

        # print(cloneP[:], cloneD[:], cloneQdes[:], cloneDQdes[:], cloneRunning.value, cloneResult.value)
        if cloneResult.value:

            clone.SetDesiredJointPDgains(cloneP[:], cloneD[:])
            clone.SetDesiredJointPosition(cloneQdes[:])
            clone.SetDesiredJointVelocity(cloneDQdes[:])
            clone.SetDesiredJointTorque([0.0] * 12)

            clone.SendCommand(WaitEndOfCycle=True)

            cloneResult.value = False

    return 0


def control_loop(name_interface_clone=None, des_vel_analysis=None):
    """Main function that calibrates the robot, get it into a default waiting position then launch
    the main control loop once the user has pressed the Enter key

    Args:
        name_interface_clone (string): name of the interface that will mimic the movements of the first
    """

    # Check .yaml file for parameters of the controller

    # Name of the interface that is used to communicate with the robot
    name_interface = params.interface

    # Enable or disable PyBullet GUI
    if not params.SIMULATION:
        params.enable_pyb_GUI = False

    # Time variable to keep track of time
    t = 0.0

    # Default position after calibration
    q_init = np.array(params.q_init.tolist())

    if params.SIMULATION and (des_vel_analysis is not None):
        print("Analysis: %1.1f %1.1f %1.1f" % (des_vel_analysis[0], des_vel_analysis[1], des_vel_analysis[5]))
        acceleration_rate = 0.1  # m/s^2
        steady_state_duration = 3  # s
        N_analysis = int(np.max(np.abs(des_vel_analysis)) / acceleration_rate / params.dt_wbc) + 1
        N_steady = int(steady_state_duration / params.dt_wbc)
        params.N_SIMULATION = N_analysis + N_steady

    # Run a scenario and retrieve data thanks to the logger
    controller = Controller(params, q_init, t)

    if params.SIMULATION and (des_vel_analysis is not None):
        controller.joystick.update_for_analysis(des_vel_analysis, N_analysis, N_steady)

    ####

    if params.SIMULATION:
        device = PyBulletSimulator()
        qc = None
    else:
        device = oci.robot_from_yaml_file(params.config_file)
        qc = QualisysClient(ip="140.93.16.160", body_id=0)

    if name_interface_clone is not None:
        print("PASS")
        from multiprocessing import Process, Array, Value
        cloneP = Array('d', [0] * 12)
        cloneD = Array('d', [0] * 12)
        cloneQdes = Array('d', [0] * 12)
        cloneDQdes = Array('d', [0] * 12)
        cloneRunning = Value('b', True)
        cloneResult = Value('b', True)
        clone = Process(target=clone_movements, args=(name_interface_clone, q_init, cloneP,
                        cloneD, cloneQdes, cloneDQdes, cloneRunning, cloneResult))
        clone.start()
        print(cloneResult.value)

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
        # ForceMonitor to display contact forces in PyBullet with red lines
        import ForceMonitor
        myForceMonitor = ForceMonitor.ForceMonitor(device.pyb_sim.robotId, device.pyb_sim.planeId)
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

    t_log_whole = np.zeros((params.N_SIMULATION))
    k_log_whole = 0
    t_start_whole = 0.0
    T_whole = time.time()
    dT_whole = 0.0

    log_Mddq = np.zeros((params.N_SIMULATION, 6))
    log_NLE = np.zeros((params.N_SIMULATION, 6))
    log_JcTf = np.zeros((params.N_SIMULATION, 6))
    log_Mddq_out = np.zeros((params.N_SIMULATION, 6))
    log_JcTf_out = np.zeros((params.N_SIMULATION, 6))
    while ((not device.is_timeout) and (t < t_max) and (not controller.error)):

        t_start_whole = time.time()

        # Update sensor data (IMU, encoders, Motion capture)
        device.parse_sensor_data()

        # Desired torques
        controller.compute(device)

        # Check that the initial position of actuators is not too far from the
        # desired position of actuators to avoid breaking the robot
        if (t <= 10 * params.dt_wbc):
            if np.max(np.abs(controller.result.q_des - device.joints.positions)) > 0.15:
                print("DIFFERENCE: ", controller.result.q_des - device.joints.positions)
                print("q_des: ", controller.result.q_des)
                print("q_mes: ", device.joints.positions)
                break

        # Set desired quantities for the actuators
        device.joints.set_position_gains(controller.result.P)
        device.joints.set_velocity_gains(controller.result.D)
        device.joints.set_desired_positions(controller.result.q_des)
        device.joints.set_desired_velocities(controller.result.v_des)
        device.joints.set_torques(controller.result.FF * controller.result.tau_ff.ravel())

        log_Mddq[k_log_whole] = controller.wbcWrapper.Mddq
        log_NLE[k_log_whole] = controller.wbcWrapper.NLE
        log_JcTf[k_log_whole] = controller.wbcWrapper.JcTf
        log_Mddq_out[k_log_whole] = controller.wbcWrapper.Mddq_out
        log_JcTf_out[k_log_whole] = controller.wbcWrapper.JcTf_out

        # Call logger
        if params.LOGGING or params.PLOTTING:
            loggerSensors.sample(device, qc)
            loggerControl.sample(controller.joystick, controller.estimator,
                                 controller, controller.gait, controller.statePlanner,
                                 controller.footstepPlanner, controller.footTrajectoryGenerator,
                                 controller.wbcWrapper, dT_whole)


        t_end_whole = time.time()

        # myForceMonitor.display_contact_forces()

        # Send command to the robot
        for i in range(1):
            device.send_command_and_wait_end_of_cycle(params.dt_wbc)
        """if (t % 1) < 5e-5:
            print('IMU attitude:', device.imu.attitude_euler)
            print('joint pos   :', device.joints.positions)
            print('joint vel   :', device.joints.velocities)
            device.robot_interface.PrintStats()"""

        """import os
        from matplotlib import pyplot as plt
        import pybullet as pyb
        if (t == 0.0):
            cpt_frames = 0
            step = 10
        if (cpt_frames % step) == 0:
            if (cpt_frames % 1000):
                print(cpt_frames)
            img = pyb.getCameraImage(width=1920, height=1080, renderer=pyb.ER_BULLET_HARDWARE_OPENGL)
            if cpt_frames == 0:
                newpath = r'/tmp/recording'
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
            if (int(cpt_frames/step) < 10):
                plt.imsave('/tmp/recording/frame_000'+str(int(cpt_frames/step))+'.png', img[2])
            elif int(cpt_frames/step) < 100:
                plt.imsave('/tmp/recording/frame_00'+str(int(cpt_frames/step))+'.png', img[2])
            elif int(cpt_frames/step) < 1000:
                plt.imsave('/tmp/recording/frame_0'+str(int(cpt_frames/step))+'.png', img[2])
            else:
                plt.imsave('/tmp/recording/frame_'+str(int(cpt_frames/step))+'.png', img[2])

        cpt_frames += 1"""

        t += params.dt_wbc  # Increment loop time

        dT_whole = T_whole
        T_whole = time.time()
        dT_whole = T_whole - dT_whole

        t_log_whole[k_log_whole] = t_end_whole - t_start_whole
        k_log_whole += 1

    # ****************************************************************

    if (t >= t_max):
        finished = True
    else:
        finished = False

    # Stop clone interface running in parallel process
    if not params.SIMULATION and name_interface_clone is not None:
        cloneResult.value = False

    # Stop MPC running in a parallel process
    if params.enable_multiprocessing:
        print("Stopping parallel process")
        controller.mpc_wrapper.stop_parallel_loop()
    # controller.view.stop()  # Stop viewer

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

    """from matplotlib import pyplot as plt
    N = loggerControl.tstamps.shape[0]
    t_range = np.array([k*loggerControl.dt for k in range(N)])
    plt.figure()
    plt.plot(t_range, log_Mddq[:-3, 0], "r")
    plt.plot(t_range, log_NLE[:-3, 0], "b")
    plt.plot(t_range, log_Mddq[:-3, 0] + log_NLE[:-3, 0], "violet", linestyle="--")
    plt.plot(t_range, log_JcTf[:-3, 0], "g")
    plt.plot(t_range, log_Mddq_out[:-3, 0], "darkred")
    plt.plot(t_range, log_Mddq_out[:-3, 0] + log_NLE[:-3, 0], "darkorchid", linestyle="--")
    plt.plot(t_range, log_JcTf_out[:-3, 0], "mediumblue")
    plt.plot(t_range, loggerControl.planner_gait[:, 0, 0], "k", linewidth=3)
    plt.legend(["Mddq", "NLE", "Mddq+NLE", "JcT f", "Mddq out", "Mddq out + NLE", "JcT f out", "Contact"])
    plt.show(block=True)"""

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

    # os.nice(-20)  #  Set the process to highest priority (from -20 highest to +20 lowest)
    f, v = control_loop(parser.parse_args().clone)  # , np.array([1.5, 0.0, 0.0, 0.0, 0.0, 0.0]))
    print(f, v)
    quit()


if __name__ == "__main__":
    main()
