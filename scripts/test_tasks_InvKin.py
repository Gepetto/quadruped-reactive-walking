import numpy as np
import utils_mpc
from example_robot_data.robots_loader import Solo12Loader
import pinocchio as pin
import libquadruped_reactive_walking as lqrw
from IPython import embed

def check_task_tracking():

    params = lqrw.Params()  # Object that holds all controller parameters
    params.w_tasks = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # Keep only contact task
    dt = params.dt_wbc
    h_ref = 0.244749

    # Load robot model and data
    # Initialisation of the Gepetto viewer
    Solo12Loader.free_flyer = True
    solo = Solo12Loader().robot
    q = solo.q0.reshape((-1, 1))
    q[2, 0] = h_ref
    q[7:, 0] = params.q_init
    dq = np.zeros((18, 1))

    solo.initViewer(loadModel=True)
    if ('viewer' in solo.viz.__dict__):
        solo.viewer.gui.addFloor('world/floor')
        solo.viewer.gui.setRefreshIsSynchronous(False)
    solo.display(q)

    wbcWrapper = lqrw.WbcWrapper()
    wbcWrapper.initialize(params)

    feet_p_cmd0 = np.array([[0.1946, 0.1946, -0.1946, -0.1946],
                           [0.16891, -0.16891, 0.16891, -0.16891],
                           [0.0191028, 0.0191028, 0.0191028, 0.0191028]])
    feet_p_cmd0[2, :] = - h_ref
    feet_p_cmd = np.zeros((3, 4))
    feet_v_cmd = np.zeros((3, 4))
    feet_a_cmd = np.zeros((3, 4))
    xgoals = np.zeros((12, 1))
    q_wbc = np.zeros((18, 1))

    k = 0
    N = 50000
    while k < N:

        # Rotation from world to base frame
        oRb = pin.Quaternion(q[3:7, 0:1]).toRotationMatrix()

        # Roll pitch yaw vector
        RPY = pin.rpy.matrixToRpy(oRb)

        # Rotation from horizontal to base frame
        hRb = pin.rpy.rpyToMatrix(RPY[0], RPY[1], 0.0)

        # Target positions of feet
        feet_p_cmd = feet_p_cmd0.copy()
        feet_v_cmd = np.zeros((3, 4))
        feet_a_cmd = np.zeros((3, 4))
        feet_p_cmd[0, 0] = 0.1946 + 0.04 * np.sin(2 * np.pi * 0.25 * k * dt)
        feet_p_cmd[1, 1] = -0.16891 + 0.04 * np.sin(2 * np.pi * 0.25 * k * dt)
        feet_p_cmd[2, 2] = - h_ref + 0.04 * np.sin(2 * np.pi * 0.25 * k * dt)
        feet_v_cmd[0, 0] = 0.04 * 2 * np.pi * 0.25 * np.cos(2 * np.pi * 0.25 * k * dt)
        feet_v_cmd[1, 1] = 0.04 * 2 * np.pi * 0.25 * np.cos(2 * np.pi * 0.25 * k * dt)
        feet_v_cmd[2, 2] = 0.04 * 2 * np.pi * 0.25 * np.cos(2 * np.pi * 0.25 * k * dt)
        feet_a_cmd[0, 0] = -0.04 * (2 * np.pi * 0.25)**2 * np.sin(2 * np.pi * 0.25 * k * dt)
        feet_a_cmd[1, 1] = -0.04 * (2 * np.pi * 0.25)**2 * np.sin(2 * np.pi * 0.25 * k * dt)
        feet_a_cmd[2, 2] = -0.04 * (2 * np.pi * 0.25)**2 * np.sin(2 * np.pi * 0.25 * k * dt)

        # Express feet so that they follow base orientation
        feet_p_cmd = hRb @ feet_p_cmd
        feet_v_cmd = hRb @ feet_v_cmd
        feet_a_cmd = hRb @ feet_a_cmd

        # Goal is 20 degrees in pitch
        xgoals[4, 0] = - 20.0 / 57

        # Make q_wbc vector
        q_wbc[:3, 0] = np.zeros(3)  # Position
        q_wbc[3, 0] = RPY[0]  # Roll
        q_wbc[4, 0] = RPY[1]  # Pitch
        q_wbc[5, 0] = 0.0  # Yaw
        q_wbc[6:, 0] = q[7:, 0]  # Actuators

        # Run InvKin + WBC QP
        wbcWrapper.compute(q_wbc, dq, np.zeros(12), np.array([[0.0, 0.0, 0.0, 1.0]]),
                           feet_p_cmd, feet_v_cmd, feet_a_cmd, xgoals)

        # Velocity integration
        dq[0:3, 0:1] += dt * oRb.transpose() @ wbcWrapper.ddq_cmd[0:3].reshape((-1, 1))
        dq[3:6, 0:1] += dt * oRb.transpose() @ wbcWrapper.ddq_cmd[3:6].reshape((-1, 1))
        dq[6:, 0] += dt * wbcWrapper.ddq_cmd[6:]

        # Position integration
        q[:, 0] = pin.integrate(solo.model, q, dt * dq)

        k += 1

        if (k % 10 == 0):
            solo.display(q)


def check_task_contact():

    params = lqrw.Params()  # Object that holds all controller parameters
    params.w_tasks = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # Keep only contact task
    dt = params.dt_wbc
    h_ref = 0.244749

    # Load robot model and data
    # Initialisation of the Gepetto viewer
    Solo12Loader.free_flyer = True
    solo = Solo12Loader().robot
    q = solo.q0.reshape((-1, 1))
    q[2, 0] = h_ref
    q[7:, 0] = params.q_init
    dq = np.zeros((18, 1))

    solo.initViewer(loadModel=True)
    if ('viewer' in solo.viz.__dict__):
        solo.viewer.gui.addFloor('world/floor')
        solo.viewer.gui.setRefreshIsSynchronous(False)
    solo.display(q)

    wbcWrapper = lqrw.WbcWrapper()
    wbcWrapper.initialize(params)

    feet_p_cmd0 = np.array([[0.1946, 0.1946, -0.1946, -0.1946],
                           [0.16891, -0.16891, 0.16891, -0.16891],
                           [0.0191028, 0.0191028, 0.0191028, 0.0191028]])
    feet_p_cmd0[2, :] = - h_ref
    feet_p_cmd = np.zeros((3, 4))
    feet_v_cmd = np.zeros((3, 4))
    feet_a_cmd = np.zeros((3, 4))
    xgoals = np.zeros((12, 1))
    q_wbc = np.zeros((18, 1))

    k = 0
    N = 10000
    freq = 0.2
    log_f = np.zeros((N, 3, 4))
    log_fcmd = np.zeros((N, 3, 4))
    dx = 0.0
    dy = 0.0
    dw = 0.0
    while k < N:

        # Displacement
        sig = np.sign(np.sin(2 * np.pi * freq * k * dt))
        V = 0.02
        W = 10.0 / 57
        dx += dt * V * sig
        dy += dt * V * sig
        dw += dt * W * sig
        q[0, 0] = dx  # 0.04 * np.cos(2 * np.pi * freq * k * dt)
        q[1, 0] = dy  # 0.04 * np.sin(2 * np.pi * freq * k * dt)
        RPY = pin.rpy.matrixToRpy(pin.Quaternion(q[3:7, 0:1]).toRotationMatrix())
        RPY[2] = dw
        q[3:7, 0] = pin.Quaternion(pin.rpy.rpyToMatrix(RPY[0], RPY[1], RPY[2])).coeffs()

        # Rotation from world to base frame
        oRb = pin.Quaternion(q[3:7, 0:1]).toRotationMatrix()

        # Roll pitch yaw vector
        RPY = pin.rpy.matrixToRpy(oRb)

        # Rotation from horizontal to base frame
        oRh = pin.rpy.rpyToMatrix(0.0, 0.0, RPY[2])
        hRb = pin.rpy.rpyToMatrix(RPY[0], RPY[1], 0.0)

        # Target positions of feet
        feet_p_cmd = feet_p_cmd0.copy()
        feet_v_cmd = np.zeros((3, 4))
        feet_a_cmd = np.zeros((3, 4))

        # Feet in the air move with the base
        """feet_p_cmd[0, 1] += 0.04 * np.cos(2 * np.pi * freq * k * dt)
        feet_p_cmd[1, 1] += 0.04 * np.sin(2 * np.pi * freq * k * dt)
        feet_p_cmd[0, 2] += 0.04 * np.cos(2 * np.pi * freq * k * dt)
        feet_p_cmd[1, 2] += 0.04 * np.sin(2 * np.pi * freq * k * dt)"""

        """feet_v_cmd[0, 1] = -0.04 * 2 * np.pi * freq * np.sin(2 * np.pi * freq * k * dt)
        feet_v_cmd[1, 1] = 0.04 * 2 * np.pi * freq * np.cos(2 * np.pi * freq * k * dt)
        feet_v_cmd[0, 2] = -0.04 * 2 * np.pi * freq * np.sin(2 * np.pi * freq * k * dt)
        feet_v_cmd[1, 2] = 0.04 * 2 * np.pi * freq * np.cos(2 * np.pi * freq * k * dt)

        feet_a_cmd[0, 1] = -0.04 * (2 * np.pi * freq)**2 * np.cos(2 * np.pi * freq * k * dt)
        feet_a_cmd[1, 1] = -0.04 * (2 * np.pi * freq)**2 * np.sin(2 * np.pi * freq * k * dt)
        feet_a_cmd[0, 2] = -0.04 * (2 * np.pi * freq)**2 * np.cos(2 * np.pi * freq * k * dt)
        feet_a_cmd[1, 2] = -0.04 * (2 * np.pi * freq)**2 * np.sin(2 * np.pi * freq * k * dt)

        # Feet in contact do not move with the base
        feet_p_cmd[0, 0] -= 0.04 * np.cos(2 * np.pi * freq * k * dt)
        feet_p_cmd[1, 0] -= 0.04 * np.sin(2 * np.pi * freq * k * dt)
        feet_p_cmd[0, 3] -= 0.04 * np.cos(2 * np.pi * freq * k * dt)
        feet_p_cmd[1, 3] -= 0.04 * np.sin(2 * np.pi * freq * k * dt)"""

        feet_v_cmd[0, 1] = V * sig
        feet_v_cmd[1, 1] = V * sig
        feet_v_cmd[0, 2] = V * sig
        feet_v_cmd[1, 2] = V * sig

        for i in [0, 3]:
            feet_p_cmd[0, i] -= dx
            feet_p_cmd[1, i] -= dy
            feet_p_cmd[:, i:(i+1)] = oRh.transpose() @ feet_p_cmd[:, i:(i+1)]
            feet_v_cmd[:, i:(i+1)] = oRh.transpose() @ feet_v_cmd[:, i:(i+1)]
            feet_a_cmd[:, i:(i+1)] = oRh.transpose() @ feet_a_cmd[:, i:(i+1)]

        # Express feet so that they follow base orientation
        feet_p_cmd = hRb @ feet_p_cmd
        feet_v_cmd = hRb @ feet_v_cmd
        feet_a_cmd = hRb @ feet_a_cmd

        # Make q_wbc vector
        q_wbc[:3, 0] = np.zeros(3)  # Position
        q_wbc[3, 0] = RPY[0]  # Roll
        q_wbc[4, 0] = RPY[1]  # Pitch
        q_wbc[5, 0] = 0.0  # Yaw
        q_wbc[6:, 0] = q[7:, 0]  # Actuators

        xgoals[6, 0] = V * sig
        xgoals[7, 0] = V * sig
        xgoals[11, 0] = W * sig

        # Run InvKin + WBC QP
        wbcWrapper.compute(q_wbc, dq, np.zeros(12), np.array([[1.0, 0.0, 0.0, 1.0]]),
                           feet_p_cmd, feet_v_cmd, feet_a_cmd, xgoals)

        # Logging
        """log_f_cmd = feet_p_cmd0.copy()
        log_f_cmd[0, 1] += 0.04 * np.cos(2 * np.pi * freq * k * dt)
        log_f_cmd[1, 1] += 0.04 * np.sin(2 * np.pi * freq * k * dt)
        log_f_cmd[0, 2] += 0.04 * np.cos(2 * np.pi * freq * k * dt)
        log_f_cmd[1, 2] += 0.04 * np.sin(2 * np.pi * freq * k * dt)"""
        log_fcmd[k] = wbcWrapper.feet_pos_target
        log_fcmd[k, 0, :] += q[0, 0]
        log_fcmd[k, 1, :] += q[1, 0]
        log_fcmd[k, :, :] = oRh @ log_fcmd[k, :, :]
        log_f[k] = wbcWrapper.feet_pos
        log_f[k, 0, :] += q[0, 0]
        log_f[k, 1, :] += q[1, 0]
        log_f[k, :, :] = oRh @ log_f[k, :, :]

        # Velocity integration
        dq[0:3, 0:1] += dt * oRb.transpose() @ wbcWrapper.ddq_cmd[0:3].reshape((-1, 1))
        dq[3:6, 0:1] += dt * oRb.transpose() @ wbcWrapper.ddq_cmd[3:6].reshape((-1, 1))
        dq[6:, 0] += dt * wbcWrapper.ddq_cmd[6:]

        # Position integration
        q[:, 0] = pin.integrate(solo.model, q, dt * dq)

        k += 1

        if (k % 10 == 0):
            solo.display(q)

    from matplotlib import pyplot as plt
    for j in range(3):
        plt.figure()
        for i in range(4):
            if i == 0:
                ax0 = plt.subplot(2, 2, i+1)
            else:
                plt.subplot(2, 2, i+1, sharex=ax0)

            plt.plot(log_f[:, j, i], "b", linewidth=3)
            plt.plot(log_fcmd[:, j, i], "r", linewidth=3, linestyle="--")

            plt.legend(["Mes", "Ref"], prop={'size': 8})

    plt.show(block=True)

if __name__ == "__main__":

    # check_task_tracking()
    check_task_contact()
