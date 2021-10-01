import numpy as np
import utils_mpc
from example_robot_data.robots_loader import Solo12Loader
import pinocchio as pin
import libquadruped_reactive_walking as lqrw

def check_task_contact():

    # Load robot model and data
    # Initialisation of the Gepetto viewer
    Solo12Loader.free_flyer = True
    solo = Solo12Loader().robot
    q = solo.q0.reshape((-1, 1))
    dq = np.zeros((18, 1))

    solo.initViewer(loadModel=True)
    if ('viewer' in solo.viz.__dict__):
        solo.viewer.gui.addFloor('world/floor')
        solo.viewer.gui.setRefreshIsSynchronous(False)
    solo.display(q)

    params = lqrw.Params()  # Object that holds all controller parameters
    wbcWrapper = lqrw.WbcWrapper()
    wbcWrapper.initialize(params)

    feet_p_cmd = np.array([[0.1946, 0.1946, -0.1946, -0.1946],
                           [0.16891, -0.16891, 0.16891, -0.16891],
                           [0.0191028, 0.0191028, 0.0191028, 0.0191028]])
    feet_v_cmd = np.zeros((3, 4))
    feet_a_cmd = np.zeros((3, 4))
    xgoals = np.zeros((12, 1))

    # Run InvKin + WBC QP
    wbcWrapper.compute(q, dq, np.zeros(12), np.array([[0.0, 1.0, 1.0, 0.0]]),
                       feet_p_cmd, feet_v_cmd, feet_a_cmd, xgoals)

    

if __name__ == "__main__":

    check_task_contact()
