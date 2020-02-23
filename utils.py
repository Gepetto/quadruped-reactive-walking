import math
import numpy as np

import robots_loader  # Gepetto viewer

import Joystick
import ContactSequencer
import FootstepPlanner
import FootTrajectoryGenerator
import MpcSolver
import MPC

##########################
# ROTATION MATRIX TO RPY #
##########################

# Taken from https://www.learnopencv.com/rotation-matrix-to-euler-angles/

# Checks if a matrix is a valid rotation matrix.


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):

    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

###########################################################
#  Roll Pitch Yaw (3 x 1) to Quaternion function (4 x 1)  #
###########################################################


def getQuaternion(rpy):
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

##################
# Initialisation #
##################


def init_viewer():
    solo = robots_loader.loadSolo(False)
    solo.initDisplay(loadModel=True)
    solo.viewer.gui.addFloor('world/floor')
    solo.display(solo.q0)

    return solo

from IPython import embed

def init_objects(dt, k_max_loop):

    dt = 0.15
    # Create Joystick object
    joystick = Joystick.Joystick()

    # Create contact sequencer object
    sequencer = ContactSequencer.ContactSequencer(dt)

    # Create footstep planner object
    fstep_planner = FootstepPlanner.FootstepPlanner(dt)

    # Create trajectory generator object
    ftraj_gen = FootTrajectoryGenerator.FootTrajectoryGenerator(dt)

    # Create MPC solver object
    mpc = MpcSolver.MpcSolver(dt, sequencer, k_max_loop)

    # Create the new version of the MPC solver object
    mpc_v2 = MPC.MPC(dt, sequencer)
    mpc_v2.update_v_ref(joystick)
    mpc_v2.run(0, sequencer, fstep_planner, ftraj_gen)
    embed()
    return joystick, sequencer, fstep_planner, ftraj_gen, mpc


def display_all(solo, k, sequencer, fstep_planner, ftraj_gen, mpc):
    # Display non-locked target footholds with green spheres (gepetto gui)
    fstep_planner.update_viewer(solo.viewer, (k == 0))

    # Display locked target footholds with red spheres (gepetto gui)
    # Display desired 3D position of feet with magenta spheres (gepetto gui)
    ftraj_gen.update_viewer(solo.viewer, (k == 0))

    # Display reference trajectory, predicted trajectory, desired contact forces, current velocity
    mpc.update_viewer(solo.viewer, (k == 0), sequencer)

    qu_pinocchio = np.array(solo.q0).flatten()
    qu_pinocchio[0:3] = mpc.q_w[0:3, 0]
    # TODO: Check why orientation of q_w and qu are different
    # qu_pinocchio[3:7, 0:1] = getQuaternion(settings.q_w[3:6, 0:1])
    qu_pinocchio[3:7] = getQuaternion(np.matrix([mpc.q_next[3:6, 0]]).T).flatten()

    # Refresh the gepetto viewer display
    solo.display(qu_pinocchio)


def getSkew(a):
    """Returns the skew matrix of a 3 by 1 column vector

    Keyword arguments:
    a -- the column vector
    """
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=a.dtype)

