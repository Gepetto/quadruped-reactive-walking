# coding: utf8

import os
import sys
sys.path.insert(0, './mpctsid')

from utils.logger import Logger
import argparse
import numpy as np
from utils.viewerClient import viewerClient, NonBlockingViewerFromRobot
import threading

SIMULATION = False
LOGGING = True

if SIMULATION:
    from mpctsid.utils_mpc import PyBulletSimulator
else:
    # from pynput import keyboard
    from solo12 import Solo12
    from utils.qualisysClient import QualisysClient

DT = 0.001

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

def get_input():
    global key_pressed
    keystrk=input('Put the robot on the floor and press Enter \n')
    # thread doesn't continue until key is pressed
    key_pressed=True

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
    i=threading.Thread(target=get_input)
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



def mcapi_playback(name_interface):
    """Main function that calibrates the robot, get it into a default waiting position then launch
    the main control loop once the user has pressed the Enter key

    Args:
        name_interface (string): name of the interface that is used to communicate with the robot
    """
    name_replay = "/home/odri/git/thomasCbrs/log_eval/test_3/06_nl/"
    # name_replay = "/home/odri/git/thomasCbrs/log_eval/vmax_nl/"
    # replay_q = np.loadtxt(name_replay + "_q.dat", delimiter=" ")
    # replay_v = np.loadtxt(name_replay + "_v.dat", delimiter=" ")
    # replay_tau = np.loadtxt(name_replay + "_tau.dat", delimiter=" ")
    qtsid_full = np.load(name_replay + "qtsid.npy" , allow_pickle = True)
    vtsid_full = np.load(name_replay + "vtsid.npy" , allow_pickle = True)
    tau_ff = np.load(name_replay + "torques_ff.npy" , allow_pickle = True)
    replay_q = qtsid_full[7:,:].transpose()
    replay_v = vtsid_full[6:,:].transpose()
    replay_tau = tau_ff.transpose()

    N_SIMULATION = replay_q.shape[0]

    # Default position after calibration
    # q_init = replay_q[0, 1:]
    q_init = replay_q[0, :]

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

    # Initiate communication with the device and calibrate encoders
    if SIMULATION:
        device.Init(calibrateEncoders=True, q_init=q_init, envID=0,
                    use_flat_plane=True, enable_pyb_GUI=True, dt=DT)
    else:
        device.Init(calibrateEncoders=True, q_init=q_init)

        # Wait for Enter input before starting the control loop
        put_on_the_floor(device, q_init)

    # CONTROL LOOP ***************************************************
    t = 0.0
    t_max = (N_SIMULATION-1) * DT
    i = 0

    P = 7 * np.ones(12)
    D = 0.5 * np.ones(12)
    q_des = np.zeros(12)
    v_des = np.zeros(12)
    tau_ff = np.zeros(12)

    while ((not device.hardware.IsTimeout()) and (t < t_max)):

        device.UpdateMeasurment()  # Retrieve data from IMU and Motion capture

        # Set desired quantities for the actuators
        device.SetDesiredJointPDgains(P, D)
        # device.SetDesiredJointPosition(replay_q[i, 1:])
        # device.SetDesiredJointVelocity(replay_v[i, 1:])
        # device.SetDesiredJointTorque(replay_tau[i, 1:])
        device.SetDesiredJointPosition(replay_q[i, :])
        device.SetDesiredJointVelocity(replay_v[i, :])
        device.SetDesiredJointTorque(replay_tau[i, :])

        # Call logger
        if LOGGING:
            logger.sample(device, qualisys=qc)

        # Send command to the robot
        device.SendCommand(WaitEndOfCycle=True)
        if ((device.cpt % 1000) == 0):
            device.Print()

        t += DT
        i += 1

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

    # Save the logs of the Logger object
    if LOGGING:
        logger.saveAll()
        print("Log saved")

    if SIMULATION:
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
