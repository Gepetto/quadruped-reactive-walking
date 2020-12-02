# coding: utf8

from utils.logger import Logger
import tsid as tsid
import pinocchio as pin
import argparse
import numpy as np
from mpctsid.Estimator import Estimator
from utils.viewerClient import viewerClient, NonBlockingViewerFromRobot
import os
import sys
sys.path.insert(0, './mpctsid')

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

    if SIMULATION:
        device = PyBulletSimulator()
        qc = None
    else:
        device = Solo12(name_interface, dt=DT)
        qc = QualisysClient(ip="140.93.16.160", body_id=0)

    if LOGGING:
        logger = Logger(device, qualisys=qc)

    # Number of motors
    nb_motors = device.nb_motors
    q_viewer = np.array((7 + nb_motors) * [0., ])

    # Gepetto-gui
    v = viewerClient()
    v.display(q_viewer)

    # PyBullet GUI
    enable_pyb_GUI = True

    # Maximum duration of the demonstration
    t_max = 300.0

    # Default position after calibration
    q_init = np.array([0, 0.8, -1.6, 0, 0.8, -1.6, 0, -0.8, 1.6, 0, -0.8, 1.6])

    # Create Estimator object
    estimator = Estimator(DT, np.int(t_max/DT))

    # Set the paths where the urdf and srdf file of the robot are registered
    modelPath = "/opt/openrobots/share/example-robot-data/robots"
    urdf = modelPath + "/solo_description/robots/solo12.urdf"
    vector = pin.StdVec_StdString()
    vector.extend(item for item in modelPath)

    # Create the robot wrapper from the urdf model (which has no free flyer) and add a free flyer
    robot = tsid.RobotWrapper(urdf, vector, pin.JointModelFreeFlyer(), False)
    model = robot.model()

    # Creation of the Invverse Dynamics HQP problem using the robot
    # accelerations (base + joints) and the contact forces
    invdyn = tsid.InverseDynamicsFormulationAccForce("tsid", robot, False)

    # Compute the problem data with a solver based on EiQuadProg
    invdyn.computeProblemData(0.0, np.hstack(
        (np.zeros(7), q_init)), np.zeros(18))

    # Initiate communication with the device and calibrate encoders
    if SIMULATION:
        device.Init(calibrateEncoders=True, q_init=q_init, envID=0,
                    use_flat_plane=True, enable_pyb_GUI=enable_pyb_GUI, dt=DT)
    else:
        device.Init(calibrateEncoders=True, q_init=q_init)

        # Wait for Enter input before starting the control loop
        put_on_the_floor(device, q_init)

    # CONTROL LOOP ***************************************************
    t = 0.0
    k = 0

    while ((not device.hardware.IsTimeout()) and (t < t_max)):

        device.UpdateMeasurment()  # Retrieve data from IMU and Motion capture

        # Run estimator with hind left leg touching the ground
        estimator.run_filter(k, np.array(
            [0, 0, 1, 0]), device, invdyn.data(), model)

        # Zero desired torques
        tau = np.zeros(12)

        # Set desired torques for the actuators
        device.SetDesiredJointTorque(tau)

        # Call logger
        if LOGGING:
            logger.sample(device, qualisys=qc, estimator=estimator)

        # Send command to the robot
        device.SendCommand(WaitEndOfCycle=True)
        if ((device.cpt % 100) == 0):
            device.Print()

        # Gepetto GUI
        if k > 0:
            pos = np.array(estimator.data.oMf[26].translation).ravel()
            q_viewer[0:3] = np.array(
                [-pos[0], -pos[1], estimator.FK_h])  # Position
            q_viewer[3:7] = estimator.q_FK[3:7, 0]  # Orientation
            q_viewer[7:] = estimator.q_FK[7:, 0]  # Encoders
            v.display(q_viewer)

        t += DT
        k += 1

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
    if LOGGING:
        logger.saveAll()

    if SIMULATION and enable_pyb_GUI:
        # Disconnect the PyBullet server (also close the GUI)
        device.Stop()

    print("End of script")
    quit()


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
