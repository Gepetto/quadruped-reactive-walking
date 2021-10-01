import numpy as np
import utils_mpc
from example_robot_data.robots_loader import Solo12Loader
import pinocchio as pin

class Test:

    def __init__(self, params):

        self.k = 0
        self.k_mpc = 20
        self.dt_wbc = params.dt_wbc
        q_init = np.array(params.q_init.tolist())

        # Initialisation of the solo model/data and of the Gepetto viewer
        self.solo = utils_mpc.init_robot(q_init, params)

        self.h_ref = params.h_ref

        self.wbcWrapper = lqrw.WbcWrapper()
        self.wbcWrapper.initialize(params)

        self.gait = lqrw.Gait()
        self.gait.initialize(params)

        self.footstepPlanner = lqrw.FootstepPlanner()
        self.footstepPlanner.initialize(params, self.gait)

        self.footTrajectoryGenerator = lqrw.FootTrajectoryGenerator()
        self.footTrajectoryGenerator.initialize(params, self.gait)

        self.q = np.zeros((19, 1))  # Orientation part is in roll pitch yaw
        self.q[0:7, 0] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.q[7:, 0] = q_init
        self.h_v = np.zeros((18, 1))
        self.h_v_ref = np.zeros((18, 1))

        self.q_wbc = np.zeros((18, 1))
        self.dq_wbc = np.zeros((18, 1))
        self.xgoals = np.zeros((12, 1))
        self.xgoals[2, 0] = self.h_ref

        self.ddq_cmd = np.zeros((18, 1))

        # Load robot model and data
        """Solo12Loader.free_flyer = True
        self.solo = Solo12Loader().robot"""

    def run(self):

        # Update gait
        self.gait.updateGait(self.k, self.k_mpc, 0)
        cgait = self.gait.getCurrentGait()

        # Compute target footstep based on current and reference velocities
        qftps = np.zeros((18, 1))
        qftps[:3, 0] = self.q[:3, 0]
        RPY = pin.rpy.matrixToRpy(pin.Quaternion(self.q[3:7, 0:1]).toRotationMatrix())
        qftps[3:6, 0] = RPY
        qftps[6:, 0] = self.q[7:, 0]
        o_targetFootstep = self.footstepPlanner.updateFootsteps(self.k % self.k_mpc == 0 and self.k != 0,
                                                                int(self.k_mpc - self.k % self.k_mpc),
                                                                qftps[:, 0],
                                                                self.h_v[0:6, 0:1].copy(),
                                                                self.h_v_ref[0:6, 0:1])

        # Update pos, vel and acc references for feet
        self.footTrajectoryGenerator.update(self.k, o_targetFootstep)

        # Update configuration vector for wbc
        self.q_wbc[3, 0] = RPY[0]  # Roll
        self.q_wbc[4, 0] = RPY[1]  # Pitch
        self.q_wbc[6:, 0] = self.q[7:, 0]  # Joints

        # Update velocity vector for wbc
        self.dq_wbc[:, 0] = self.h_v[:, 0].copy()

        # Feet command position, velocity and acceleration in base frame
        oRh = pin.Quaternion(self.q[3:7, 0:1]).toRotationMatrix()
        oTh = self.q[:3, 0:1]

        self.feet_a_cmd = self.footTrajectoryGenerator.getFootAccelerationBaseFrame(
            oRh.transpose(), np.zeros((3, 1)), np.zeros((3, 1)))
        self.feet_v_cmd = self.footTrajectoryGenerator.getFootVelocityBaseFrame(
            oRh.transpose(), np.zeros((3, 1)), np.zeros((3, 1)))
        self.feet_p_cmd = self.footTrajectoryGenerator.getFootPositionBaseFrame(
            oRh.transpose(), oTh + np.array([[0.0], [0.0], [self.h_ref]]))

        # Desired position, orientation and velocities of the base
        self.xgoals[[0, 1, 5], 0] = np.zeros((3,))
        self.xgoals[2:5, 0] = [0.0, 0.0, 0.0]

        self.xgoals[6:, 0] = self.h_v_ref[:6, 0]  # Velocities (in horizontal frame!)

        cgait = self.gait.getCurrentGait()

        """if self.k == 50:
            from IPython import embed
            embed()"""

        # Run InvKin + WBC QP
        self.wbcWrapper.compute(self.q_wbc, self.dq_wbc,
                                np.zeros((12, 1)), np.array([cgait[0, :]]),
                                self.feet_p_cmd,
                                self.feet_v_cmd,
                                self.feet_a_cmd,
                                self.xgoals)

        # Quantities sent to the control board
        self.h_v[:, 0] += self.dt_wbc * self.wbcWrapper.ddq_cmd[:]
        self.q[:, 0] = pin.integrate(self.solo.model, self.q, self.dt_wbc * self.h_v)

        self.k += 1


if __name__ == "__main__":

    import libquadruped_reactive_walking as lqrw
    import pinocchio as pin

    params = lqrw.Params()  # Object that holds all controller parameters
    params.enable_corba_viewer = True
    test = Test(params)
    q_display = np.zeros((19, 1))

    test.h_v[0, 0] = 0.6
    test.h_v_ref[0, 0] = 0.7

    N = 10000
    log_q = np.zeros((N, 19))
    log_q_wbc = np.zeros((N, 18))
    log_dq = np.zeros((N, 18))
    log_ddq_cmd = np.zeros((N, 18))
    log_ddq_with_delta = np.zeros((N, 18))
    for i in range(N):
        test.run()

        log_ddq_cmd[i] = test.wbcWrapper.ddq_cmd
        log_dq[i] = test.h_v[:, 0]
        log_q[i] = test.q[:, 0]
        log_q_wbc[i] = test.q_wbc[:, 0]

        print("#####-> ", test.h_v[0, 0])

        """test.q[2, 0] = test.b_des[2]
        test.q[3:6, 0] = pin.rpy.matrixToRpy(pin.Quaternion(test.b_des[3:7].reshape((-1, 1))).toRotationMatrix())
        test.q[6:, 0] = test.q_des[:]"""

        # Display robot in Gepetto corba viewer
        if (i % 10 == 0 and i > 0):
            q_display[:3, 0] = test.q[0:3, 0]
            q_display[3:7, 0] = test.q[3:7, 0]
            q_display[7:, 0] = test.q[7:, 0]
            test.solo.display(q_display)

            """from IPython import embed
            embed()"""
        """test.q[0, 0] += params.dt_wbc * test.h_v[0, 0]
        test.q[1, 0] = 0.0
        test.q[5, 0] = 0.0"""

    index6 = [1, 3, 5, 2, 4, 6]

    from matplotlib import pyplot as plt
    t_range = np.array([k*0.001 for k in range(N)])

    lgd = ["Acc X", "Acc Y", "Acc Z", "Acc Roll", "Acc Pitch", "Acc Yaw"]
    plt.figure()
    for i in range(6):
        if i == 0:
            ax0 = plt.subplot(3, 2, index6[i])
        else:
            plt.subplot(3, 2, index6[i], sharex=ax0)

        plt.plot(t_range, log_ddq_cmd[:, i], "b", linewidth=3)

        plt.legend(["ddq_cmd"], prop={'size': 8})
        plt.ylabel(lgd[i])

    plt.show(block=True)

    lgd = ["Pos X", "Pos Y", "Pos Z", "Roll", "Pitch", "Yaw"]
    plt.figure()
    for i in range(6):
        if i == 0:
            ax0 = plt.subplot(3, 2, index6[i])
        else:
            plt.subplot(3, 2, index6[i], sharex=ax0)

        if i < 3:
            plt.plot(t_range, log_q[:, i], "b", linewidth=3)
        else:
            plt.plot(t_range, log_q_wbc[:, i], "b", linewidth=3)

        plt.legend(["Robot state"], prop={'size': 8})
        plt.ylabel(lgd[i])

    lgd = ["Vel X", "Vel Y", "Vel Z", "Vel Roll", "Vel Pitch", "Vel Yaw"]
    plt.figure()
    for i in range(6):
        if i == 0:
            ax0 = plt.subplot(3, 2, index6[i])
        else:
            plt.subplot(3, 2, index6[i], sharex=ax0)

        plt.plot(t_range, log_dq[:, i], "b", linewidth=3)

        plt.legend(["Robot state"], prop={'size': 8})
        plt.ylabel(lgd[i])

    plt.show(block=True)
    from IPython import embed
    embed()

