# coding: utf8

"""from utils.logger import Logger
import tsid as tsid
import pinocchio as pin
import argparse
import numpy as np
import mpctsid.test_bug
# from mpctsid.Estimator import Estimator
#from mpctsid.Controller import Controller
from utils.viewerClient import viewerClient, NonBlockingViewerFromRobot"""
"""import os
import sys
sys.path.insert(0, './mpctsid')"""


"""SIMULATION = True
LOGGING = False

if SIMULATION:
    from mpctsid.utils_mpc import PyBulletSimulator
else:
    from pynput import keyboard
    from solo12 import Solo12
    from utils.qualisysClient import QualisysClient"""
#from matplotlib import pyplot as plt
# from utils_mpc import PyBulletSimulator
import numpy as np
import argparse
# from solo12 import Solo12
# from pynput import keyboard

# from utils.logger import Logger
# from utils.qualisysClient import QualisysClient

"""import os
import sys
sys.path.insert(0, './mpctsid')"""

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
        pos[motor] = q_init[device.motorToUrdf[motor]] * \
            device.gearRatioSigned[motor]
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
    from IPython import embed
    import pybullet as pyb
    import pybullet_data

    # Start the client for PyBullet
    pyb.connect(pyb.GUI)
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    robotStartPos = [0, 0, 0.235+0.0045]  # +0.2]
    robotStartOrientation = pyb.getQuaternionFromEuler([0.0, 0.0, 0.0])  # -np.pi/2
    pyb.setAdditionalSearchPath("/opt/openrobots/share/example-robot-data/robots/solo_description/robots")
    robotId = pyb.loadURDF("solo12.urdf", robotStartPos, robotStartOrientation)
    
    print("PASS")
    embed()


    #########################################
    # PARAMETERS OF THE MPC-TSID CONTROLLER #
    #########################################

    envID = 0  # Identifier of the environment to choose in which one the simulation will happen
    velID = 0  # Identifier of the reference velocity profile to choose which one will be sent to the robot

    dt_tsid = 0.0020  # Time step of TSID
    dt_mpc = 0.02  # Time step of the MPC
    # dt is dt_tsid, defined in the TSID controller script
    k_mpc = int(dt_mpc / dt_tsid)
    t = 0.0  # Time
    n_periods = 1  # Number of periods in the prediction horizon
    T_gait = 0.64  # Duration of one gait period
    N_SIMULATION = 20000  # number of simulated TSID time steps

    # If True the ground is flat, otherwise it has bumps
    use_flat_plane = True

    # Enable or disable PyBullet GUI
    enable_pyb_GUI = True

    # Default position after calibration
    q_init = np.array([0.0, 0.8, -1.6, 0, 0.8, -1.6,
                       0, -0.8, 1.6, 0, -0.8, 1.6])

    ####

    # Create device object
    # device = Solo12(name_interface, dt=DT)
    device = PyBulletSimulator()

    # qc = QualisysClient(ip="140.93.16.160", body_id=0)  # QualisysClient
    # logger = Logger(device, qualisys=qc)  # Logger object
    nb_motors = device.nb_motors

    # Calibrate encoders
    #device.Init(calibrateEncoders=True, q_init=q_init)
    device.Init(calibrateEncoders=True, q_init=q_init, envID=envID,
                use_flat_plane=use_flat_plane, enable_pyb_GUI=enable_pyb_GUI, dt=dt_tsid)

    # Wait for Enter input before starting the control loop
    # put_on_the_floor(device, q_init)

    # CONTROL LOOP ***************************************************
    t = 0.0
    t_max = (N_SIMULATION-2) * dt_tsid
    while ((not device.hardware.IsTimeout()) and (t < t_max)):

        device.UpdateMeasurment()  # Retrieve data from IMU and Motion capture

        # Desired torques
        tau = np.zeros(12)

        # Set desired torques for the actuators
        device.SetDesiredJointTorque(tau)

        # Call logger
        # logger.sample(device, qualisys=qc)

        # Send command to the robot
        device.SendCommand(WaitEndOfCycle=True)
        if ((device.cpt % 1000) == 0):
            device.Print()

        t += DT

    # ****************************************************************

    # Whatever happened we send 0 torques to the motors.
    device.SetDesiredJointTorque([0]*nb_motors)
    device.SendCommand(WaitEndOfCycle=True)

    if device.hardware.IsTimeout():
        print("Masterboard timeout detected.")
        print("Either the masterboard has been shut down or there has been a connection issue with the cable/wifi.")
    # Shut down the interface between the computer and the master board
    device.hardware.Stop()

    # Save the logs of the Logger object
    # logger.saveAll()


def main():
    """Main function
    """

    parser = argparse.ArgumentParser(
        description='Playback trajectory to show the extent of solo12 workspace.')
    parser.add_argument('-i',
                        '--interface',
                        required=True,
                        help='Name of the interface (use ifconfig in a terminal), for instance "enp1s0"')

    mcapi_playback(parser.parse_args().interface)


if __name__ == "__main__":
    main()
