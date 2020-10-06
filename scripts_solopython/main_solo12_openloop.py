# coding: utf8

import os
import sys
sys.path.insert(0, './mpctsid')

from utils.logger import Logger
import tsid as tsid
import pinocchio as pin
import argparse
import numpy as np
from mpctsid.Estimator import Estimator
from mpctsid.Controller import Controller
from utils.viewerClient import viewerClient, NonBlockingViewerFromRobot

SIMULATION = True
LOGGING = False

if SIMULATION:
    from mpctsid.utils_mpc import PyBulletSimulator
else:
    from pynput import keyboard
    from solo12 import Solo12
    from utils.qualisysClient import QualisysClient

DT = 0.002

key_pressed = False


def on_press(key):
    """Wait for a specific key press on the keyboard

    Args:
        key (keyboard.Key): the key we want to wait for
    """
    global key_pressed
    try:
        if key == keyboard.Key.enter:
            key_pressed = True
            # Stop listener
            return False
    except AttributeError:
        print('Unknown key {0} pressed'.format(key))


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
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    print("Put the robot on the floor and press Enter")
    while not key_pressed:
        device.UpdateMeasurment()
        for motor in range(device.nb_motors):
            ref = Kp_pos*(pos[motor] - device.hardware.GetMotor(motor).GetPosition() -
                          Kd_pos*device.hardware.GetMotor(motor).GetVelocity())
            ref = min(imax, max(-imax, ref))
            device.hardware.GetMotor(motor).SetCurrentReference(ref)
        device.SendCommand(WaitEndOfCycle=True)

    print("Start the motion.")


def mcapi_playback(name_interface):
    """Main function that calibrates the robot, get it into a default waiting position then launch
    the main control loop once the user has pressed the Enter key

    Args:
        name_interface (string): name of the interface that is used to communicate with the robot
    """

    #########################################
    # PARAMETERS OF THE MPC-TSID CONTROLLER #
    #########################################

    envID = 0  # Identifier of the environment to choose in which one the simulation will happen
    velID = 0  # Identifier of the reference velocity profile to choose which one will be sent to the robot

    dt_tsid = 0.0020  # Time step of TSID
    dt_mpc = 0.02  # Time step of the MPC
    k_mpc = int(dt_mpc / dt_tsid)  # dt is dt_tsid, defined in the TSID controller script
    t = 0.0  # Time
    n_periods = 1  # Number of periods in the prediction horizon
    T_gait = 0.64  # Duration of one gait period
    N_SIMULATION = 50000  # number of simulated TSID time steps

    # Which MPC solver you want to use
    # True to have PA's MPC, to False to have Thomas's MPC
    type_MPC = True

    # Whether PyBullet feedback is enabled or not
    pyb_feedback = False

    # Whether we are working with solo8 or not
    on_solo8 = False

    # If True the ground is flat, otherwise it has bumps
    use_flat_plane = True

    # If we are using a predefined reference velocity (True) or a joystick (False)
    predefined_vel = False

    # Enable or disable PyBullet GUI
    enable_pyb_GUI = True

    # Default position after calibration
    q_init = np.array([0.0, 0.8, -1.6, 0, 0.8, -1.6, 0, -0.8, 1.6, 0, -0.8, 1.6])

    # Run a scenario and retrieve data thanks to the logger
    controller = Controller(q_init, envID, velID, dt_tsid, dt_mpc, k_mpc, t, n_periods, T_gait, N_SIMULATION, type_MPC,
                            pyb_feedback, on_solo8, use_flat_plane, predefined_vel, enable_pyb_GUI)

    ####

    if SIMULATION:
        device = PyBulletSimulator()
        qc = None
    else:
        device = Solo12(name_interface, dt=DT)
        qc = QualisysClient(ip="140.93.16.160", body_id=0)

    if LOGGING:
        logger = Logger(device, qualisys=qc, logSize=N_SIMULATION)

    # Number of motors
    nb_motors = device.nb_motors
    q_viewer = np.array((7 + nb_motors) * [0., ])

    # Initiate communication with the device and calibrate encoders
    if SIMULATION:
        device.Init(calibrateEncoders=True, q_init=q_init, envID=envID,
                    use_flat_plane=use_flat_plane, enable_pyb_GUI=enable_pyb_GUI, dt=dt_tsid)
    else:
        device.Init(calibrateEncoders=True, q_init=q_init)

        # Wait for Enter input before starting the control loop
        put_on_the_floor(device, q_init)

    # CONTROL LOOP ***************************************************
    t = 0.0
    t_max = (N_SIMULATION-2) * dt_tsid
    while ((not device.hardware.IsTimeout()) and (t < t_max)):

        device.UpdateMeasurment()  # Retrieve data from IMU and Motion capture

        # Desired torques
        tau = controller.compute(device)
        # print(tau[0:3].ravel())
        tau = tau.ravel()

        # Set desired torques for the actuators
        device.SetKp(controller.result.P)
        device.SetKd(controller.result.D)
        device.SetQdes(controller.result.q_des)
        device.SetVdes(controller.result.v_des)
        device.SetTauFF(controller.result.tau_ff)

        # Call logger
        if LOGGING:
            logger.sample(device, qualisys=qc)

        # Send command to the robot
        device.SendCommand(WaitEndOfCycle=True)
        if ((device.cpt % 1000) == 0):
            device.Print()

        t += DT

    # ****************************************************************

    # Stop MPC running in a parallel process
    if controller.enable_multiprocessing:
        print("Stopping parallel process")
        controller.mpc_wrapper.stop_parallel_loop()

    # Whatever happened we send 0 torques to the motors.
    device.SetDesiredJointTorque([0]*nb_motors)
    device.SendCommand(WaitEndOfCycle=True)

    if device.hardware.IsTimeout():
        print("Masterboard timeout detected.")
        print("Either the masterboard has been shut down or there has been a connection issue with the cable/wifi.")
    device.hardware.Stop()  # Shut down the interface between the computer and the master board

    # Save the logs of the Logger object
    if LOGGING:
        logger.saveAll()
        print("Log saved")

    if SIMULATION and enable_pyb_GUI:
        # Disconnect the PyBullet server (also close the GUI)
        device.Stop()

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

    mcapi_playback(parser.parse_args().interface)


if __name__ == "__main__":
    main()
