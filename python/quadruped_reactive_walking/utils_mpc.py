import math
import numpy as np

from example_robot_data import load

import Joystick
import Logger
import Estimator
import pinocchio as pin


######################################
# RPY / Quaternion / Rotation matrix #
######################################


def getQuaternion(rpy):
    """Roll Pitch Yaw (3 x 1) to Quaternion (4 x 1)"""

    c = np.cos(rpy*0.5)
    s = np.sin(rpy*0.5)
    cy = c[2, 0]
    sy = s[2, 0]
    cp = c[1, 0]
    sp = s[1, 0]
    cr = c[0, 0]
    sr = s[0, 0]

    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr

    return np.array([[x, y, z, w]]).transpose()


def quaternionToRPY(quat):
    """Quaternion (4 x 0) to Roll Pitch Yaw (3 x 1)"""

    qx = quat[0]
    qy = quat[1]
    qz = quat[2]
    qw = quat[3]

    rotateXa0 = 2.0*(qy*qz + qw*qx)
    rotateXa1 = qw*qw - qx*qx - qy*qy + qz*qz
    rotateX = 0.0

    if (rotateXa0 != 0.0) and (rotateXa1 != 0.0):
        rotateX = np.arctan2(rotateXa0, rotateXa1)

    rotateYa0 = -2.0*(qx*qz - qw*qy)
    rotateY = 0.0
    if (rotateYa0 >= 1.0):
        rotateY = np.pi/2.0
    elif (rotateYa0 <= -1.0):
        rotateY = -np.pi/2.0
    else:
        rotateY = np.arcsin(rotateYa0)

    rotateZa0 = 2.0*(qx*qy + qw*qz)
    rotateZa1 = qw*qw + qx*qx - qy*qy - qz*qz
    rotateZ = 0.0
    if (rotateZa0 != 0.0) and (rotateZa1 != 0.0):
        rotateZ = np.arctan2(rotateZa0, rotateZa1)

    return np.array([[rotateX], [rotateY], [rotateZ]])


def EulerToQuaternion(roll_pitch_yaw):
    """Roll Pitch Yaw to Quaternion"""

    roll, pitch, yaw = roll_pitch_yaw
    sr = math.sin(roll/2.)
    cr = math.cos(roll/2.)
    sp = math.sin(pitch/2.)
    cp = math.cos(pitch/2.)
    sy = math.sin(yaw/2.)
    cy = math.cos(yaw/2.)
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return [qx, qy, qz, qw]

##################
# Initialisation #
##################


def init_robot(q_init, enable_viewer):
    """Load the solo model and initialize the Gepetto viewer if it is enabled

    Args:
        q_init (array): the default position of the robot actuators
        enable_viewer (bool): if the Gepetto viewer is enabled or not
    """

    # Load robot model and data
    # Initialisation of the Gepetto viewer
    print(enable_viewer)
    solo = load('solo12', display=enable_viewer)
    q = solo.q0.reshape((-1, 1))
    q[7:, 0] = q_init

    
    """if enable_viewer:
        solo.initViewer(loadModel=True)
        if ('viewer' in solo.viz.__dict__):
            solo.viewer.gui.addFloor('world/floor')
            solo.viewer.gui.setRefreshIsSynchronous(False)"""
    if enable_viewer:
        solo.display(q)
    print("PASS")

    # Initialisation of model quantities
    pin.centerOfMass(solo.model, solo.data, q, np.zeros((18, 1)))
    pin.updateFramePlacements(solo.model, solo.data)
    pin.crba(solo.model, solo.data, solo.q0)

    # Initialisation of the position of footsteps
    fsteps_init = np.zeros((3, 4))
    indexes = [10, 18, 26, 34]
    for i in range(4):
        fsteps_init[:, i] = solo.data.oMf[indexes[i]].translation
    h_init = (solo.data.oMf[1].translation - solo.data.oMf[indexes[0]].translation)[2]
    fsteps_init[2, :] = 0.0

    return solo, fsteps_init, h_init


def init_objects(dt_tsid, dt_mpc, k_max_loop, k_mpc, T_mpc, type_MPC, predefined, h_init, kf_enabled):
    """Create several objects that are used in the control loop

    Args:
        dt_tsid (float): time step of TSID
        dt_mpc (float): time step of the MPC
        k_max_loop (int): maximum number of iterations of the simulation
        k_mpc (int): number of tsid iterations for one iteration of the mpc
        T_mpc (float): duration of mpc prediction horizon
        type_MPC (bool): which MPC you want to use (PA's or Thomas')
        predefined (bool): if we are using a predefined reference velocity (True) or a joystick (False)
        h_init (float): initial height of the robot base
        kf_enabled (bool): complementary filter (False) or kalman filter (True)
    """

    # Create Joystick object
    joystick = Joystick.Joystick(predefined)

    # Create logger object
    logger = Logger.Logger(k_max_loop, dt_tsid, dt_mpc, k_mpc, T_mpc, type_MPC)

    # Create Estimator object
    estimator = Estimator.Estimator(dt_tsid, k_max_loop, h_init, kf_enabled)

    return joystick, logger, estimator


def display_all(solo, k, sequencer, fstep_planner, ftraj_gen, mpc):
    """Update various objects in the Gepetto viewer: the Solo robot as well as debug spheres

    Args:
        k (int): current iteration of the simulation
        sequencer (object): ContactSequencer object
        fstep_planner (object): FootstepPlanner object
        ftraj_gen (object): FootTrajectoryGenerator object
        mpc (object): MpcSolver object
    """

    # Display non-locked target footholds with green spheres (gepetto gui)
    fstep_planner.update_viewer(solo.viewer, (k == 0))

    # Display locked target footholds with red spheres (gepetto gui)
    # Display desired 3D position of feet with magenta spheres (gepetto gui)
    ftraj_gen.update_viewer(solo.viewer, (k == 0))

    # Display reference trajectory, predicted trajectory, desired contact forces, current velocity
    # mpc.update_viewer(solo.viewer, (k == 0), sequencer)
    # mpc.plot_graphs(sequencer)

    qu_pinocchio = np.array(solo.q0).flatten()
    qu_pinocchio[0:3] = mpc.q_w[0:3, 0]
    qu_pinocchio[3:7] = getQuaternion(np.array([mpc.q_w[3:6, 0]])).flatten()
    # Refresh the gepetto viewer display
    solo.display(qu_pinocchio)


def getSkew(a):
    """Returns the skew matrix of a 3 by 1 column vector

    Keyword arguments:
    a -- the column vector
    """
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=a.dtype)
