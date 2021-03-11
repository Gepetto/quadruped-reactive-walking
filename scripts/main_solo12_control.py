# coding: utf8

import os
import sys

sys.path.insert(0, './solopython')


import threading
from Controller import Controller
import numpy as np
import argparse
from LoggerSensors import LoggerSensors
from LoggerControl import LoggerControl


SIMULATION = False
LOGGING = False
PLOTTING = False

if SIMULATION:
    from PyBulletSimulator import PyBulletSimulator
else:
    # from pynput import keyboard
    from solopython.solo12 import Solo12
    from solopython.utils.qualisysClient import QualisysClient

DT = 0.0020

key_pressed = False


def get_input():
    global key_pressed
    keystrk = input('Put the robot on the floor and press Enter \n')
    # thread doesn't continue until key is pressed
    key_pressed = True


def put_on_the_floor(device, q_init):
    """Make the robot go to the default initial position and wait for the user
    to press the Enter key to start the main control loop

    Args:
        device (robot wrapper): a wrapper to communicate with the robot
        q_init (array): the default position of the robot
    """
    global key_pressed
    key_pressed = False
    Kp_pos = 3.
    Kd_pos = 0.01
    imax = 3.0
    pos = np.zeros(device.nb_motors)
    for motor in range(device.nb_motors):
        pos[motor] = q_init[device.motorToUrdf[motor]] * device.gearRatioSigned[motor]
    i = threading.Thread(target=get_input)
    i.start()
    while not key_pressed:
        device.UpdateMeasurment()
        for motor in range(device.nb_motors):
            ref = Kp_pos*(pos[motor] - device.hardware.GetMotor(motor).GetPosition() -
                          Kd_pos*device.hardware.GetMotor(motor).GetVelocity())
            ref = min(imax, max(-imax, ref))
            device.hardware.GetMotor(motor).SetCurrentReference(ref)
        device.SendCommand(WaitEndOfCycle=True)

    print("Start the motion.")


def clone_movements(name_interface_clone, q_init, cloneP, cloneD, cloneQdes, cloneDQdes, cloneRunning, cloneResult):

    print("-- Launching clone interface --")

    print(name_interface_clone, DT)
    clone = Solo12(name_interface_clone, dt=DT)
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


def control_loop(name_interface, name_interface_clone=None):
    """Main function that calibrates the robot, get it into a default waiting position then launch
    the main control loop once the user has pressed the Enter key

    Args:
        name_interface (string): name of the interface that is used to communicate with the robot
        name_interface_clone (string): name of the interface that will mimic the movements of the first
    """

    ################################
    # PARAMETERS OF THE CONTROLLER #
    ################################

    envID = 0  # Identifier of the environment to choose in which one the simulation will happen
    velID = 0  # Identifier of the reference velocity profile to choose which one will be sent to the robot

    dt_wbc = DT  # Time step of the whole body control
    dt_mpc = 0.02  # Time step of the model predictive control
    k_mpc = int(dt_mpc / dt_wbc)
    t = 0.0  # Time
    T_gait = 0.32  # Duration of one gait period
    T_mpc = 0.32   # Duration of the prediction horizon
    N_SIMULATION = 12000  # number of simulated wbc time steps

    # Which MPC solver you want to use
    # True to have PA's MPC, to False to have Thomas's MPC
    type_MPC = True

    # Whether PyBullet feedback is enabled or not
    pyb_feedback = True

    # Whether we are working with solo8 or not
    on_solo8 = False

    # If True the ground is flat, otherwise it has bumps
    use_flat_plane = True

    # If we are using a predefined reference velocity (True) or a joystick (False)
    predefined_vel = False

    # Use complementary filter (False) or kalman filter (True) for the estimator
    kf_enabled = False

    # Enable or disable PyBullet GUI
    enable_pyb_GUI = True
    if not SIMULATION:
        enable_pyb_GUI = False

    # Default position after calibration
    q_init = np.array([0.0, 0.7, -1.4, -0.0, 0.7, -1.4, 0.0, -0.7, +1.4, -0.0, -0.7, +1.4])

    # Run a scenario and retrieve data thanks to the logger
    controller = Controller(q_init, envID, velID, dt_wbc, dt_mpc, k_mpc, t, T_gait, T_mpc, N_SIMULATION, type_MPC,
                            pyb_feedback, on_solo8, use_flat_plane, predefined_vel, enable_pyb_GUI, kf_enabled)

    ####

    if SIMULATION:
        device = PyBulletSimulator()
        qc = None
    else:
        device = Solo12(name_interface, dt=DT)
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

    if LOGGING or PLOTTING:
        loggerSensors = LoggerSensors(device, qualisys=qc, logSize=N_SIMULATION-3)
        loggerControl = LoggerControl(dt_wbc, joystick=controller.joystick, estimator=controller.estimator,
                                      loop=controller, planner=controller.planner, logSize=N_SIMULATION-3)

    # Number of motors
    nb_motors = device.nb_motors

    # Initiate communication with the device and calibrate encoders
    if SIMULATION:
        device.Init(calibrateEncoders=True, q_init=q_init, envID=envID,
                    use_flat_plane=use_flat_plane, enable_pyb_GUI=enable_pyb_GUI, dt=dt_wbc)
    else:
        device.Init(calibrateEncoders=True, q_init=q_init)

        # Wait for Enter input before starting the control loop
        put_on_the_floor(device, q_init)

    # CONTROL LOOP ***************************************************
    t = 0.0
    t_max = (N_SIMULATION-2) * dt_wbc

    while ((not device.hardware.IsTimeout()) and (t < t_max) and (not controller.myController.error)):

        # Update sensor data (IMU, encoders, Motion capture)
        device.UpdateMeasurment()

        # Desired torques
        controller.compute(device)

        # Check that the initial position of actuators is not too far from the
        # desired position of actuators to avoid breaking the robot
        if (t <= 10 * DT):
            if np.max(np.abs(controller.result.q_des - device.q_mes)) > 0.15:
                print("DIFFERENCE: ", controller.result.q_des - device.q_mes)
                print("q_des: ", controller.result.q_des)
                print("q_mes: ", device.q_mes)
                break

        # Set desired quantities for the actuators
        device.SetDesiredJointPDgains(controller.result.P, controller.result.D)
        device.SetDesiredJointPosition(controller.result.q_des)
        device.SetDesiredJointVelocity(controller.result.v_des)
        device.SetDesiredJointTorque(controller.result.tau_ff.ravel())

        # Call logger
        if LOGGING or PLOTTING:
            loggerSensors.sample(device, qc)
            loggerControl.sample(controller.joystick, controller.estimator,
                                 controller, controller.planner, controller.myController)

        # Send command to the robot
        for i in range(1):
            device.SendCommand(WaitEndOfCycle=True)
        """if ((device.cpt % 100) == 0):
            device.Print()"""

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

        t += DT  # Increment loop time

    # ****************************************************************

    # Stop clone interface running in parallel process
    if not SIMULATION and name_interface_clone is not None:
        cloneResult.value = False

    # Stop MPC running in a parallel process
    if controller.enable_multiprocessing:
        print("Stopping parallel process")
        controller.mpc_wrapper.stop_parallel_loop()
    # controller.view.stop()  # Stop viewer

    # DAMPING TO GET ON THE GROUND PROGRESSIVELY *********************
    t = 0.0
    t_max = 2.5
    while ((not device.hardware.IsTimeout()) and (t < t_max)):

        device.UpdateMeasurment()  # Retrieve data from IMU and Motion capture

        # Set desired quantities for the actuators
        device.SetDesiredJointPDgains(np.zeros(12), 0.1 * np.ones(12))
        device.SetDesiredJointPosition(np.zeros(12))
        device.SetDesiredJointVelocity(np.zeros(12))
        device.SetDesiredJointTorque(np.zeros(12))

        # Send command to the robot
        device.SendCommand(WaitEndOfCycle=True)
        if ((device.cpt % 1000) == 0):
            device.Print()

        t += DT

    # FINAL SHUTDOWN *************************************************

    # Whatever happened we send 0 torques to the motors.
    device.SetDesiredJointTorque([0]*nb_motors)
    device.SendCommand(WaitEndOfCycle=True)

    if device.hardware.IsTimeout():
        print("Masterboard timeout detected.")
        print("Either the masterboard has been shut down or there has been a connection issue with the cable/wifi.")
    device.hardware.Stop()  # Shut down the interface between the computer and the master board

    # Plot estimated computation time for each step for the control architecture
    from matplotlib import pyplot as plt
    plt.figure()
    plt.plot(controller.t_list_filter[1:], 'r+')
    plt.plot(controller.t_list_planner[1:], 'g+')
    plt.plot(controller.t_list_mpc[1:], 'b+')
    plt.plot(controller.t_list_wbc[1:], '+', color="violet")
    plt.plot(controller.t_list_loop[1:], 'k+')
    plt.plot(controller.t_list_InvKin[1:], 'o', color="darkgreen")
    plt.plot(controller.t_list_QPWBC[1:], 'o', color="royalblue")
    plt.legend(["Estimator", "Planner", "MPC", "WBC", "Whole loop", "InvKin", "QP WBC"])
    plt.title("Loop time [s]")
    plt.show(block=True)

    # Plot recorded data
    if PLOTTING:
        loggerControl.plotAll(loggerSensors)

    # Save the logs of the Logger object
    if LOGGING:
        loggerControl.saveAll(loggerSensors)
        print("Log saved")

    if SIMULATION and enable_pyb_GUI:
        # Disconnect the PyBullet server (also close the GUI)
        device.Stop()

    if controller.myController.error:
        if (controller.error_flag == 1):
            print("-- POSITION LIMIT ERROR --")
        elif (controller.error_flag == 2):
            print("-- VELOCITY TOO HIGH ERROR --")
        elif (controller.error_flag == 3):
            print("-- FEEDFORWARD TORQUES TOO HIGH ERROR --")
        print(controller.error_value)

    print("End of script")
    quit()


def main():
    """Main function
    """

    parser = argparse.ArgumentParser(description='Playback trajectory to show the extent of solo12 workspace.')
    parser.add_argument('-i',
                        '--interface',
                        required=True,
                        help='Name of the interface (use ifconfig in a terminal), for instance "enp1s0"')
    parser.add_argument('-c',
                        '--clone',
                        required=False,
                        help='Name of the clone interface that will reproduce the movement of the first one \
                              (use ifconfig in a terminal), for instance "enp1s0"')

    control_loop(parser.parse_args().interface, parser.parse_args().clone)


if __name__ == "__main__":
    main()
