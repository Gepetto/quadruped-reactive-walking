import math
import numpy as np

from example_robot_data.robots_loader import Solo12Loader
import pinocchio as pin


##################
# Initialisation #
##################


def init_robot(q_init, params, enable_viewer):
    """Load the solo model and initialize the Gepetto viewer if it is enabled

    Args:
        q_init (array): the default position of the robot actuators
        params (object): store parameters
        enable_viewer (bool): if the Gepetto viewer is enabled or not
    """

    # Load robot model and data
    # Initialisation of the Gepetto viewer
    Solo12Loader.free_flyer = True
    solo = Solo12Loader().robot  # TODO:enable_viewer
    q = solo.q0.reshape((-1, 1))

    # Initialisation of the position of footsteps to be under the shoulder
    # There is a lateral offset of around 7 centimeters
    fsteps_under_shoulders = np.zeros((3, 4))
    indexes = [10, 18, 26, 34]  # Feet indexes
    q[7:, 0] = 0.0
    pin.framesForwardKinematics(solo.model, solo.data, q)
    for i in range(4):
        fsteps_under_shoulders[:, i] = solo.data.oMf[indexes[i]].translation
    fsteps_under_shoulders[2, :] = 0.0  # Z component does not matter

    # Initial angular positions of actuators
    q[7:, 0] = q_init

    """if enable_viewer:
        solo.initViewer(loadModel=True)
        if ('viewer' in solo.viz.__dict__):
            solo.viewer.gui.addFloor('world/floor')
            solo.viewer.gui.setRefreshIsSynchronous(False)
    if enable_viewer:
        solo.display(q)"""

    # Initialisation of model quantities
    pin.centerOfMass(solo.model, solo.data, q, np.zeros((18, 1)))
    pin.updateFramePlacements(solo.model, solo.data)
    pin.crba(solo.model, solo.data, solo.q0)

    # Initialisation of the position of footsteps
    fsteps_init = np.zeros((3, 4))
    indexes = [10, 18, 26, 34]  # Feet indexes
    for i in range(4):
        fsteps_init[:, i] = solo.data.oMf[indexes[i]].translation
    h_init = 0.0
    for i in range(4):
        h_tmp = (solo.data.oMf[1].translation - solo.data.oMf[indexes[i]].translation)[2]
        if h_tmp > h_init:
            h_init = h_tmp

    # Assumption that all feet are initially in contact on a flat ground
    fsteps_init[2, :] = 0.0

    # Initialisation of the position of shoulders
    shoulders_init = np.zeros((3, 4))
    indexes = [4, 12, 20, 28]  # Shoulder indexes
    for i in range(4):
        shoulders_init[:, i] = solo.data.oMf[indexes[i]].translation

    # Saving data
    params.h_ref = h_init
    params.mass = solo.data.mass[0]  # Mass of the whole urdf model (also = to Ycrb[1].mass)
    params.I_mat = solo.data.Ycrb[1].inertia.ravel().tolist()  # Composite rigid body inertia in q_init position

    for i in range(4):
        for j in range(3):
            params.shoulders[3*i+j] = shoulders_init[j, i]
            params.footsteps_init[3*i+j] = fsteps_init[j, i]
            params.footsteps_under_shoulders[3*i+j] = fsteps_under_shoulders[j, i]

    return solo

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
