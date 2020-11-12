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
import threading

SIMULATION = True
LOGGING = False

if SIMULATION:
    from mpctsid.utils_mpc import PyBulletSimulator
else:
    # from pynput import keyboard
    from solo12 import Solo12
    from utils.qualisysClient import QualisysClient

DT = 0.0020

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

    #########################################
    # PARAMETERS OF THE MPC-TSID CONTROLLER #
    #########################################

    envID = 0  # Identifier of the environment to choose in which one the simulation will happen
    velID = 5  # Identifier of the reference velocity profile to choose which one will be sent to the robot

    dt_tsid = 0.0020  # Time step of TSID
    dt_mpc = 0.02  # Time step of the MPC
    k_mpc = int(dt_mpc / dt_tsid)  # dt is dt_tsid, defined in the TSID controller script
    t = 0.0  # Time
    n_periods = 1  # Number of periods in the prediction horizon
    T_gait = 0.32  # Duration of one gait period
    N_SIMULATION = 10000  # number of simulated TSID time steps

    # Which MPC solver you want to use
    # True to have PA's MPC, to False to have Thomas's MPC
    type_MPC = True

    # Whether PyBullet feedback is enabled or not
    pyb_feedback = True

    # Whether we are working with solo8 or not
    on_solo8 = False

    # If True the ground is flat, otherwise it has bumps
    use_flat_plane = False

    # If we are using a predefined reference velocity (True) or a joystick (False)
    predefined_vel = True

    # Enable or disable PyBullet GUI
    enable_pyb_GUI = True
    if not SIMULATION:
        enable_pyb_GUI = False

    # Default position after calibration
    q_init = np.array([0.0, 0.8, -1.6, 0, 0.8, -1.6, 0, -0.8, +1.6, 0, -0.8, +1.6])

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
    # view = viewerClient()
    # view.display(q_viewer.copy())

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

    log_v_est = np.zeros((6, N_SIMULATION))
    log_v_tsid = np.zeros((6, N_SIMULATION))
    kk = 0
    cpt_frames = 0
    import pybullet as pyb

    while ((not device.hardware.IsTimeout()) and (t < t_max)):

        device.UpdateMeasurment()  # Retrieve data from IMU and Motion capture

        # Desired torques
        controller.compute(device)
        # print(tau[0:3].ravel())
        # tau = tau.ravel()

        """RPY_test = pin.rpy.matrixToRpy(pin.Quaternion(np.array([device.baseOrientation]).transpose()).toRotationMatrix())
        print("Orientation robot: ", RPY_test)"""

        # Check that the initial position of actuators is not too far from the
        # desired position of actuators to avoid breaking the robot
        if (t == 0.0):
            if np.max(np.abs(controller.result.q_des - device.q_mes)) > 0.15:
                print("DIFFERENCE: ", controller.result.q_des - device.q_mes)
                break

        # Set desired quantities for the actuators
        device.SetDesiredJointPDgains(controller.result.P, controller.result.D)
        device.SetDesiredJointPosition(controller.result.q_des)
        device.SetDesiredJointVelocity(controller.result.v_des)
        device.SetDesiredJointTorque(controller.result.tau_ff.ravel())

        # Call logger
        if LOGGING:
            logger.sample(device, qualisys=qc, estimator=controller.estimator)

        # view.display(controller.q[:, 0:1])

        """if kk >= 1500:
            if kk == 1500:
                print("FORCE")
            device.pyb_sim.apply_external_force(kk, 1500, 1000, np.array([0.0, +6.0, 0.0]), np.zeros((3,)))"""

        # view.display(controller.q)

        # Send command to the robot
        for i in range(1):
            device.SendCommand(WaitEndOfCycle=True)
        """if ((device.cpt % 1000) == 0):
            device.Print()"""

        # log_v_est[:, kk] = controller.estimator.v_filt[:6, 0]
        # log_v_tsid[:, kk] = controller.b_v[:6, 0]
        # kk += 1

        t += DT
        
        import os
        from matplotlib import pyplot as plt
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

        cpt_frames += 1



    # ****************************************************************

    # Stop MPC running in a parallel process
    if controller.enable_multiprocessing:
        print("Stopping parallel process")
        controller.mpc_wrapper.stop_parallel_loop()

    """from matplotlib import pyplot as plt
    index = [1, 3, 5, 2, 4, 6]
    lgd = ["Linear vel X", "Linear vel Y", "Linear vel Z", "Angular vel Roll", "Angular vel Pitch", "Angular vel Yaw"]
    plt.figure()
    for i in range(6):
        plt.subplot(3, 2, index[i])
        plt.plot(log_v_est[i, :], "r", linewidth=2)
        plt.plot(log_v_tsid[i, :], "b", linewidth=2)
        plt.ylabel(lgd[i])
    plt.show(block=True)"""

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

    #controller.estimator.plot_graphs()

    """import matplotlib.pylab as plt
    plt.figure()
    plt.plot(controller.t_list_filter[1:], 'r+')
    plt.plot(controller.t_list_planner[1:], 'g+')
    plt.plot(controller.t_list_mpc[1:], 'b+')
    plt.plot(controller.t_list_wbc[1:], '+', color="violet")
    plt.plot(controller.t_list_loop[1:], 'k+')
    plt.plot(controller.t_list_InvKin[1:], 'o', color="darkgreen")
    plt.plot(controller.t_list_QPWBC[1:], 'o', color="royalblue")
    plt.plot(controller.t_list_intlog[1:], 'o', color="darkgoldenrod")
    plt.legend(["Estimator", "Planner", "MPC", "WBC", "Whole loop", "InvKin", "QP WBC", "Integ + Log"])
    plt.title("Loop time [s]")
    plt.show(block=True)"""

    """import matplotlib.pylab as plt
    N = len(controller.log_tmp2)
    t_range = np.array([k*0.002 for k in range(N)])
    plt.figure()
    plt.plot(t_range, controller.log_tmp1, 'b')
    plt.plot(t_range, controller.log_tmp2, 'r')
    plt.plot(t_range, controller.log_tmp3, 'g')
    plt.plot(t_range, controller.log_tmp4, 'g')
    # plt.show(block=True)"""

    # controller.myController.saveAll(fileName="push_pyb_with_ff", log_date=False)
    if LOGGING:
        controller.myController.saveAll(fileName="data_control", log_date=True)
    print("-- Controller log saved --")
    controller.myController.show_logs()

    # Save the logs of the Logger object
    if LOGGING:
        logger.saveAll()
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

    mcapi_playback(parser.parse_args().interface)


if __name__ == "__main__":
    main()
