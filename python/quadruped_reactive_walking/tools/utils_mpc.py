from example_robot_data import load
import numpy as np
import pinocchio as pin


def init_robot(q_init, params):
    """Load the solo model and initialize the Gepetto viewer if it is enabled

    Args:
        q_init (array): the default position of the robot actuators
        params (object): store parameters
    """
    # Load robot model and data
    # Initialisation of the Gepetto viewer
    solo = load("solo12")
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

    if params.enable_corba_viewer:
        solo.initViewer(loadModel=True)
        if "viewer" in solo.viz.__dict__:
            solo.viewer.gui.addFloor("world/floor")
            solo.viewer.gui.setRefreshIsSynchronous(False)
        solo.display(q)

    # Initialisation of model quantities
    pin.centerOfMass(solo.model, solo.data, q, np.zeros((18, 1)))
    pin.updateFramePlacements(solo.model, solo.data)
    pin.crba(solo.model, solo.data, solo.q0)

    # Initialisation of the position of footsteps
    fsteps_init = np.zeros((3, 4))
    indexes = [
        solo.model.getFrameId("FL_FOOT"),
        solo.model.getFrameId("FR_FOOT"),
        solo.model.getFrameId("HL_FOOT"),
        solo.model.getFrameId("HR_FOOT"),
    ]
    for i in range(4):
        fsteps_init[:, i] = solo.data.oMf[indexes[i]].translation
    h_init = 0.0
    for i in range(4):
        h_tmp = (solo.data.oMf[1].translation - solo.data.oMf[indexes[i]].translation)[
            2
        ]
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
    params.mass = solo.data.mass[
        0
    ]  # Mass of the whole urdf model (also = to Ycrb[1].mass)
    params.I_mat = (
        solo.data.Ycrb[1].inertia.ravel().tolist()
    )  # Composite rigid body inertia in q_init position
    params.CoM_offset = (solo.data.com[0][:3] - q[0:3, 0]).tolist()
    params.CoM_offset[1] = 0.0

    for i in range(4):
        for j in range(3):
            params.shoulders[3 * i + j] = shoulders_init[j, i]
            params.footsteps_init[3 * i + j] = fsteps_init[j, i]
            params.footsteps_under_shoulders[3 * i + j] = fsteps_init[
                j, i
            ]  # Use initial feet pos as reference

    return solo
