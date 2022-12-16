import unittest
import numpy as np
import pinocchio as pin

# Import classes to test
import quadruped_reactive_walking as lqrw

# Tune numpy output
np.set_printoptions(precision=6, linewidth=300)


class TestInvKin(unittest.TestCase):
    def setUp(self):

        # Object that holds all controller parameters
        self.params = lqrw.Params()

        # Set parameters to overwrite user's values
        self.params.dt_wbc = 0.001
        self.params.dt_mpc = 0.02
        self.k_mpc = int(self.params.dt_mpc / self.params.dt_wbc)
        self.params.N_SIMULATION = 1000
        self.params.perfect_estimator = False
        self.params.solo3D = False

        # Parameters of FootTrajectoryGenerator
        self.params.max_height = 0.05  # Apex height of the swinging trajectory [m]
        self.params.lock_time = 0.04  # Target lock before the touchdown [s]
        # Duration during which feet move only along Z when taking off and landing
        self.params.vert_time = 0.03

        q_init = [
            0.0,
            0.764,
            -1.407,
            0.0,
            0.76407,
            -1.4,
            0.0,
            0.76407,
            -1.407,
            0.0,
            0.764,
            -1.407,
        ]
        for i in range(len(q_init)):
            self.params.q_init[i] = q_init[i]
        self.params.N_periods = 1
        gait = [12, 1, 0, 0, 1, 12, 0, 1, 1, 0]
        for i in range(len(gait)):
            self.params.gait_vec[i] = gait[i]

        # Force refresh of gait matrix
        self.params.convert_gait_vec()

        # Initialization of params
        self.params.initialize()

        # Create WBC Wrapper class and initialize it
        self.wbcWrapper = lqrw.WbcWrapper()
        self.wbcWrapper.initialize(self.params)

    def tearDown(self):
        pass

    def test_base_tracking(self):

        DISPLAY = True

        # Load robot model and data
        from example_robot_data.robots_loader import Solo12Loader

        Solo12Loader.free_flyer = True
        solo = Solo12Loader().robot
        q = solo.q0.reshape((-1, 1))
        q[2, 0] = self.params.h_ref
        q[7:, 0] = self.params.q_init
        dq = np.zeros((18, 1))

        # Get foot indexes
        # BASE_ID = solo.model.getFrameId("base_link")
        foot_ids = [
            solo.model.getFrameId("FL_FOOT"),
            solo.model.getFrameId("FR_FOOT"),
            solo.model.getFrameId("HL_FOOT"),
            solo.model.getFrameId("HR_FOOT"),
        ]

        # Initialization of viewer
        if DISPLAY:
            solo.initViewer(loadModel=True)
            if "viewer" in solo.viz.__dict__:
                solo.viewer.gui.addFloor("world/floor")
                solo.viewer.gui.setRefreshIsSynchronous(False)
            solo.display(q)

        # References for tracking tasks
        feet_p_cmd0 = np.array(
            [
                [0.1946, 0.1946, -0.1946, -0.1946],
                [0.16891, -0.16891, 0.16891, -0.16891],
                [0.0191028, 0.0191028, 0.0191028, 0.0191028],
            ]
        )
        feet_p_cmd0[2, :] = 0.05  # - self.params.h_ref

        xgoals = np.zeros((12, 1))
        q_wbc = np.zeros((18, 1))
        dq_wbc = np.zeros((18, 1))
        posb = np.zeros((6, 1))
        velb = np.zeros((6, 1))
        posf = np.zeros((3, 4))
        velf = np.zeros((3, 4))
        accf = np.zeros((3, 4))

        q_cmd = np.zeros((19, 1))
        q_cmd[7:, 0] = self.params.q_init
        dq_cmd = np.zeros((18, 1))

        k = 0
        N = 800
        log_vel_cmd = np.zeros((N, 3, 4))
        log_pos_cmd = np.zeros((N, 3, 4))
        log_dq = np.zeros((N, 18))
        log_q = np.zeros((N, 19))

        log_ddq_cmd = np.zeros((N, 18))
        log_dq_cmd = np.zeros((N, 18))
        log_q_cmd = np.zeros((N, 19))

        log_pos_b = np.zeros((N, 6))
        log_vel_b = np.zeros((N, 6))
        log_pos_f = np.zeros((N, 3, 4))
        log_vel_f = np.zeros((N, 3, 4))
        log_acc_f = np.zeros((N, 3, 4))

        pos_b_cmd = np.zeros((6, 1))
        vel_b_cmd = np.zeros((6, 1))
        pos_f_cmd = np.zeros((3, 4))
        vel_f_cmd = np.zeros((3, 4))
        acc_f_cmd = np.zeros((3, 4))

        base_p_cmd = np.zeros((6, 1))
        base_v_cmd = np.zeros((6, 1))
        feet_p_cmd = np.zeros((3, 4))
        feet_v_cmd = np.zeros((3, 4))
        feet_a_cmd = np.zeros((3, 4))

        log_pos_b_cmd = np.zeros((N, 6))
        log_vel_b_cmd = np.zeros((N, 6))
        log_pos_f_cmd = np.zeros((N, 3, 4))
        log_vel_f_cmd = np.zeros((N, 3, 4))
        log_acc_f_cmd = np.zeros((N, 3, 4))
        log_base_p_cmd = np.zeros((N, 6))
        log_base_v_cmd = np.zeros((N, 6))
        log_feet_p_cmd = np.zeros((N, 3, 4))
        log_feet_v_cmd = np.zeros((N, 3, 4))
        log_feet_a_cmd = np.zeros((N, 3, 4))

        vy = 0.02
        wyaw = 0.0 / 57.0
        # x_ref = 0.0
        y_ref = 0.0
        yaw_ref = 0.0
        while k < N:
            # Position in base frame
            sig = np.sign(np.sin(2 * np.pi * 0.1 * k * self.params.dt_wbc))
            y_ref += vy * self.params.dt_wbc * sig
            yaw_ref += wyaw * self.params.dt_wbc * sig

            # Rotation from world to base frame
            oRb = pin.Quaternion(q[3:7, 0:1]).toRotationMatrix()

            # Roll pitch yaw vector
            RPY = pin.rpy.matrixToRpy(oRb)

            # Rotation between frames
            hRb = pin.rpy.rpyToMatrix(RPY[0], RPY[1], 0.0)
            oRh = pin.rpy.rpyToMatrix(0.0, 0.0, RPY[2])
            oTh = np.array([[q[0, 0]], [q[1, 0]], [self.params.h_ref]])

            # Target positions of feet in base frame
            o_feet_p_cmd = feet_p_cmd0.copy()
            o_feet_v_cmd = np.zeros((3, 4))
            o_feet_a_cmd = np.zeros((3, 4))
            o_feet_p_cmd[0, 0] += 0.04 * np.sin(
                2 * np.pi * 0.25 * k * self.params.dt_wbc
            )
            o_feet_p_cmd[1, 1] += 0.04 * np.sin(
                2 * np.pi * 0.25 * k * self.params.dt_wbc
            )
            o_feet_p_cmd[2, 2] += 0.04 * np.sin(
                2 * np.pi * 0.25 * k * self.params.dt_wbc
            )
            o_feet_v_cmd[0, 0] += (
                0.04
                * 2
                * np.pi
                * 0.25
                * np.cos(2 * np.pi * 0.25 * k * self.params.dt_wbc)
            )
            o_feet_v_cmd[1, 1] += (
                0.04
                * 2
                * np.pi
                * 0.25
                * np.cos(2 * np.pi * 0.25 * k * self.params.dt_wbc)
            )
            o_feet_v_cmd[2, 2] += (
                0.04
                * 2
                * np.pi
                * 0.25
                * np.cos(2 * np.pi * 0.25 * k * self.params.dt_wbc)
            )
            o_feet_a_cmd[0, 0] += (
                -0.04
                * (2 * np.pi * 0.25) ** 2
                * np.sin(2 * np.pi * 0.25 * k * self.params.dt_wbc)
            )
            o_feet_a_cmd[1, 1] += (
                -0.04
                * (2 * np.pi * 0.25) ** 2
                * np.sin(2 * np.pi * 0.25 * k * self.params.dt_wbc)
            )
            o_feet_a_cmd[2, 2] += (
                -0.04
                * (2 * np.pi * 0.25) ** 2
                * np.sin(2 * np.pi * 0.25 * k * self.params.dt_wbc)
            )

            # Second test
            amp_x = 0.1
            amp_z = 0.05
            freq = 3
            o_feet_p_cmd = feet_p_cmd0.copy()
            o_feet_v_cmd = np.zeros((3, 4))
            o_feet_a_cmd = np.zeros((3, 4))
            o_feet_p_cmd[0, 0] += amp_x * np.sin(
                2 * np.pi * freq * k * self.params.dt_wbc
            )
            o_feet_p_cmd[0, 2] += amp_x * np.sin(
                2 * np.pi * freq * k * self.params.dt_wbc + np.pi
            )
            o_feet_v_cmd[0, 0] += (
                amp_x
                * 2
                * np.pi
                * freq
                * np.cos(2 * np.pi * freq * k * self.params.dt_wbc)
            )
            o_feet_v_cmd[0, 2] += (
                amp_x
                * 2
                * np.pi
                * freq
                * np.cos(2 * np.pi * freq * k * self.params.dt_wbc + np.pi)
            )
            o_feet_a_cmd[0, 0] += (
                -amp_x
                * (2 * np.pi * freq) ** 2
                * np.sin(2 * np.pi * freq * k * self.params.dt_wbc)
            )
            o_feet_a_cmd[0, 2] += (
                -amp_x
                * (2 * np.pi * freq) ** 2
                * np.sin(2 * np.pi * freq * k * self.params.dt_wbc + np.pi)
            )

            o_feet_p_cmd[2, 0] += amp_z * np.sin(
                2 * np.pi * freq * k * self.params.dt_wbc
            )
            o_feet_p_cmd[2, 2] += amp_z * np.sin(
                2 * np.pi * freq * k * self.params.dt_wbc + np.pi
            )
            o_feet_v_cmd[2, 0] += (
                amp_z
                * 2
                * np.pi
                * freq
                * np.cos(2 * np.pi * freq * k * self.params.dt_wbc)
            )
            o_feet_v_cmd[2, 2] += (
                amp_z
                * 2
                * np.pi
                * freq
                * np.cos(2 * np.pi * freq * k * self.params.dt_wbc + np.pi)
            )
            o_feet_a_cmd[2, 0] += (
                -amp_z
                * (2 * np.pi * freq) ** 2
                * np.sin(2 * np.pi * freq * k * self.params.dt_wbc)
            )
            o_feet_a_cmd[2, 2] += (
                -amp_z
                * (2 * np.pi * freq) ** 2
                * np.sin(2 * np.pi * freq * k * self.params.dt_wbc + np.pi)
            )

            # Target positions of feet in horizontal frame
            feet_p_cmd = oRh.transpose() @ (o_feet_p_cmd - oTh)
            feet_v_cmd = oRh.transpose() @ o_feet_v_cmd
            feet_a_cmd = oRh.transpose() @ o_feet_a_cmd

            # Target positions of feet in base frame
            # feet_p_cmd = oRh.transpose() @ (feet_p_cmd0.copy() - oTh)

            # for j in range(4):
            # feet_v_cmd[0, j] = (
            # -np.cos(np.arctan(feet_p_cmd0[0, j] / feet_p_cmd0[1, j]))
            # * np.sqrt(0.1946**2 + 0.16891**2)
            # * wyaw
            # )
            # feet_v_cmd[1, j] = (
            # vy
            # + np.sin(np.arctan(feet_p_cmd0[0, j] / feet_p_cmd0[1, j]))
            # * np.sqrt(0.1946**2 + 0.16891**2)
            # * wyaw
            # )

            # Express feet so that they follow base orientation
            # feet_p_cmd = hRb @ feet_p_cmd
            # feet_v_cmd = hRb @ feet_v_cmd
            # feet_a_cmd = hRb @ feet_a_cmd

            # Goal is 20 degrees in pitch, lateral velocity, yaw velocity
            xgoals[4, 0] = -10.0 / 57
            xgoals[6, 0] = 0.1
            # * k * self.params.dt_wbc
            # 0.04 * np.sin(2 * np.pi * 0.25 * k * self.params.dt_wbc)
            xgoals[7, 0] = vy * sig
            xgoals[11, 0] = wyaw * sig

            # Make q_wbc vector
            q_wbc[:3, 0] = np.zeros(3)  # Position
            q_wbc[3, 0] = RPY[0]  # Roll
            q_wbc[4, 0] = RPY[1]  # Pitch
            q_wbc[5, 0] = 0.0  # Yaw
            q_wbc[6:, 0] = q_cmd[7:, 0]  # Actuators

            # Make dq_wbc vector
            dq_wbc[:6, 0] = dq[:6, 0]  # Base
            dq_wbc[6:, 0] = dq_cmd[6:, 0]  # Actuators

            # Save base reference
            base_p_cmd[:, 0] = xgoals[:6, 0]
            base_v_cmd[:, 0] = xgoals[6:, 0]

            # Run InvKin + WBC QP
            self.wbcWrapper.compute(
                q_wbc,
                dq_wbc,
                np.zeros(12),
                np.array([[0.0, 0.0, 0.0, 1.0]]),
                feet_p_cmd,
                feet_v_cmd,
                feet_a_cmd,
                xgoals,
            )

            # Check acceleration output from IK
            q_tmp = np.zeros((19, 1))
            q_tmp[:3, 0] = q_wbc[:3, 0]
            q_tmp[3:7, 0] = pin.Quaternion(
                pin.rpy.rpyToMatrix(q_wbc[3, 0], q_wbc[4, 0], q_wbc[5, 0])
            ).coeffs()
            q_tmp[7:, 0] = q_wbc[6:, 0]
            pin.forwardKinematics(
                solo.model,
                solo.data,
                q_tmp,
                dq_wbc,
                self.wbcWrapper.ddq_cmd.reshape((-1, 1)),
            )
            pin.updateFramePlacements(solo.model, solo.data)
            for i_ee in range(4):
                idx = int(foot_ids[i_ee])
                acc_f_cmd[:, i_ee] = pin.getFrameAcceleration(
                    solo.model, solo.data, int(idx), pin.LOCAL_WORLD_ALIGNED
                ).linear

            # Check velocity output from IK
            dq_cmd[:, 0] = self.wbcWrapper.dq_cmd
            pin.computeJointJacobians(solo.model, solo.data, q_tmp)
            pin.forwardKinematics(
                solo.model, solo.data, q_tmp, dq_cmd, np.zeros(solo.model.nv)
            )
            pin.updateFramePlacements(solo.model, solo.data)
            for i_ee in range(4):
                idx = int(foot_ids[i_ee])
                vel_f_cmd[:, i_ee] = pin.getFrameVelocity(
                    solo.model, solo.data, int(idx), pin.LOCAL_WORLD_ALIGNED
                ).linear
            vel_b_cmd[:, 0] = dq_cmd[:6, 0]

            # Check position output from IK
            q_cmd[:, 0] = self.wbcWrapper.q_cmd
            pin.forwardKinematics(
                solo.model,
                solo.data,
                q_cmd,
                np.zeros(solo.model.nv),
                np.zeros(solo.model.nv),
            )
            pin.updateFramePlacements(solo.model, solo.data)
            for i_ee in range(4):
                idx = int(foot_ids[i_ee])
                pos_f_cmd[:, i_ee] = solo.data.oMf[idx].translation
            pos_b_cmd[:3, 0] = q_cmd[:3, 0]
            pos_b_cmd[3:, 0] = pin.rpy.matrixToRpy(
                pin.Quaternion(q_cmd[3:7].reshape((-1, 1))).toRotationMatrix()
            )

            """from IPython import embed
            embed()"""

            # Velocity integration
            dq[0:3, 0:1] += (
                self.params.dt_wbc
                * hRb.transpose()
                @ self.wbcWrapper.ddq_cmd[0:3].reshape((-1, 1))
            )
            dq[3:6, 0:1] += (
                self.params.dt_wbc
                * hRb.transpose()
                @ self.wbcWrapper.ddq_cmd[3:6].reshape((-1, 1))
            )
            dq[6:, 0] += self.params.dt_wbc * self.wbcWrapper.ddq_cmd[6:]

            # Position integration
            q[:, 0] = pin.integrate(solo.model, q, self.params.dt_wbc * dq)

            # Check outputs from integration
            pin.computeJointJacobians(solo.model, solo.data, q)
            pin.forwardKinematics(solo.model, solo.data, q, dq, self.wbcWrapper.ddq_cmd)
            pin.updateFramePlacements(solo.model, solo.data)
            for i_ee in range(4):
                idx = int(foot_ids[i_ee])
                posf[:, i_ee] = solo.data.oMf[idx].translation
                velf[:, i_ee] = pin.getFrameVelocity(
                    solo.model, solo.data, int(idx), pin.LOCAL_WORLD_ALIGNED
                ).linear
                accf[:, i_ee] = pin.getFrameAcceleration(
                    solo.model, solo.data, int(idx), pin.LOCAL_WORLD_ALIGNED
                ).linear
            posb[:3, 0] = q[:3, 0]
            posb[3:, 0] = pin.rpy.matrixToRpy(
                pin.Quaternion(q[3:7].reshape((-1, 1))).toRotationMatrix()
            )
            velb[:, 0] = dq[:6, 0]

            oTh = np.array([[q[0, 0]], [q[1, 0]], [q[2, 0]]])
            RPY = pin.rpy.matrixToRpy(pin.Quaternion(q[3:7, 0]).toRotationMatrix())
            oRh = pin.rpy.rpyToMatrix(0.0, 0.0, RPY[2])

            if DISPLAY:
                log_vel_cmd[k, :, :] = np.zeros((3, 4))
                log_pos_cmd[k, :, :] = feet_p_cmd0
                log_dq[k, :] = dq[:, 0]
                log_q[k, :] = q[:, 0]

                # Commands (from integration)
                log_pos_b[k, :] = posb[:, 0]
                log_vel_b[k, :] = velb[:, 0]
                log_pos_f[k, :, :] = posf
                log_vel_f[k, :, :] = velf
                log_acc_f[k, :, :] = accf

                # Commands (from IK)
                log_ddq_cmd[k, :] = self.wbcWrapper.ddq_cmd
                log_dq_cmd[k, :] = dq_cmd[:, 0]
                log_q_cmd[k, :] = q_cmd[:, 0]
                log_pos_b_cmd[k, :] = pos_b_cmd[:, 0]
                log_vel_b_cmd[k, :] = vel_b_cmd[:, 0]
                log_pos_f_cmd[k, :, :] = pos_f_cmd
                log_vel_f_cmd[k, :, :] = vel_f_cmd
                log_acc_f_cmd[k, :, :] = acc_f_cmd

                # References
                log_base_p_cmd[k, :] = base_p_cmd[:, 0]
                log_base_v_cmd[k, :] = base_v_cmd[:, 0]
                log_feet_p_cmd[k, :, :] = feet_p_cmd
                log_feet_v_cmd[k, :, :] = feet_v_cmd
                log_feet_a_cmd[k, :, :] = feet_a_cmd

            # self.assertTrue(
            # np.allclose(oRh.transpose() @ (posf - oTh), feet_p_cmd, atol=2e-3),
            # "feet pos tracking is OK",
            # )
            # self.assertTrue(
            # np.allclose(oRh.transpose() @ velf, feet_v_cmd, atol=2e-3),
            # "feet vel tracking is OK",
            # )

            k += 1

            if True and DISPLAY and (k % 10 == 0):
                q_display = np.zeros((19, 1))
                q_display[:3, 0] = q[:3, 0]
                q_display[3:, 0] = q_tmp[3:, 0]
                solo.display(q_display)

                """from IPython import embed
                embed()"""

        return
        filename = "/home/palex/Documents/Travail/Presentations/2022_03_29_Comparisons/"
        N_test = "3"
        N_size = "18"
        lgd4 = ["FL", "FR", "HL", "HR"]
        lgd3 = [" x", " y", " z"]
        lgd6 = ["x", "y", "z", "roll", "pitch", "yaw"]
        if DISPLAY:
            index6 = [1, 3, 5, 2, 4, 6]
            index12 = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
            from matplotlib import pyplot as plt

            plt.figure()
            for i, j in enumerate([0, 1, 5]):
                plt.subplot(3, 1, i + 1)
                plt.plot(log_dq[:, j], "b")

            # RESULTS FROM INTEGRATION
            plt.figure()
            for i in range(12):
                plt.subplot(3, 4, index12[i])
                plt.plot(log_q[:, 7 + i], "r")
            plt.suptitle("Actuators position (from integration)")
            plt.figure()
            for i in range(12):
                plt.subplot(3, 4, index12[i])
                plt.plot(log_dq[:, 6 + i], "r")
            plt.suptitle("Actuators velocity (from integration)")
            plt.figure()
            mng = plt.get_current_fig_manager()
            mng.full_screen_toggle()
            for i in range(6):
                plt.subplot(3, 2, index6[i])
                plt.plot(log_base_p_cmd[:, i], "r")
                plt.plot(log_pos_b[:, i], "b")
                plt.legend(["Ref", "Cmd"])
                plt.ylabel(lgd6[i])
            plt.suptitle("Base position task (from integration)")
            plt.show(block=False)
            plt.savefig(
                filename + "base_pos_test_" + N_test + "_" + N_size + ".eps",
                dpi=150,
                bbox_inches="tight",
            )
            plt.figure()
            mng = plt.get_current_fig_manager()
            mng.full_screen_toggle()
            for i in range(6):
                plt.subplot(3, 2, index6[i])
                plt.plot(log_base_v_cmd[:, i], "r")
                plt.plot(log_vel_b[:, i], "b")
                plt.legend(["Ref", "Cmd"])
                plt.ylabel(lgd6[i])
            plt.suptitle("Base velocity task (from integration)")
            plt.show(block=False)
            plt.savefig(
                filename + "base_vel_test_" + N_test + "_" + N_size + ".eps",
                dpi=150,
                bbox_inches="tight",
            )
            plt.figure()
            for i in range(12):
                plt.subplot(3, 4, index12[i])
                plt.plot(log_feet_p_cmd[:, int(i % 3), int(i / 3)], "r")
                plt.plot(log_pos_f[:, int(i % 3), int(i / 3)], "b")
            plt.suptitle("Feet position task (from integration)")
            plt.figure()
            for i in range(12):
                plt.subplot(3, 4, index12[i])
                plt.plot(log_feet_v_cmd[:, int(i % 3), int(i / 3)], "r")
                plt.plot(log_vel_f[:, int(i % 3), int(i / 3)], "b")
            plt.suptitle("Feet velocity task (from integration)")
            plt.figure()
            for i in range(12):
                plt.subplot(3, 4, index12[i])
                plt.plot(log_feet_a_cmd[:, int(i % 3), int(i / 3)], "r")
                plt.plot(log_acc_f[:, int(i % 3), int(i / 3)], "b")
            plt.suptitle("Feet acceleration task (from integration)")

            # RESULTS FROM IK
            plt.figure()
            for i in range(12):
                plt.subplot(3, 4, index12[i])
                plt.plot(log_q_cmd[:, 7 + i], "r")
            plt.suptitle("Actuators position commands")
            plt.figure()
            for i in range(12):
                plt.subplot(3, 4, index12[i])
                plt.plot(log_dq_cmd[:, 6 + i], "r")
            plt.suptitle("Actuators velocity commands")
            plt.figure()
            for i in range(12):
                plt.subplot(3, 4, index12[i])
                plt.plot(log_ddq_cmd[:, 6 + i], "r")
            plt.suptitle("Actuators acceleration commands")
            plt.figure()
            mng = plt.get_current_fig_manager()
            mng.full_screen_toggle()
            for i in range(6):
                plt.subplot(3, 2, index6[i])
                plt.plot(log_base_p_cmd[:, i], "r")
                plt.plot(log_pos_b_cmd[:, i], "b")
                plt.legend(["Ref", "Cmd"])
                plt.ylabel(lgd6[i])
            plt.suptitle("Base position task (from IK)")
            plt.show(block=False)
            plt.figure()
            mng = plt.get_current_fig_manager()
            mng.full_screen_toggle()
            for i in range(6):
                plt.subplot(3, 2, index6[i])
                plt.plot(log_base_v_cmd[:, i], "r")
                plt.plot(log_vel_b_cmd[:, i], "b")
                plt.legend(["Ref", "Cmd"])
                plt.ylabel(lgd6[i])
            plt.suptitle("Base velocity task (from IK)")
            plt.show(block=False)
            plt.figure()
            mng = plt.get_current_fig_manager()
            mng.full_screen_toggle()
            for i in range(12):
                plt.subplot(3, 4, index12[i])
                plt.plot(log_feet_p_cmd[:, int(i % 3), int(i / 3)], "r")
                plt.plot(log_pos_f_cmd[:, int(i % 3), int(i / 3)], "b")
                plt.legend(["Ref", "Cmd"])
                plt.ylabel(lgd4[int(i / 3)] + lgd3[int(i % 3)])
            plt.suptitle("Feet position task (from IK)")
            plt.show(block=False)
            plt.savefig(
                filename + "feet_pos_test_" + N_test + "_" + N_size + ".eps",
                dpi=150,
                bbox_inches="tight",
            )
            plt.figure()
            mng = plt.get_current_fig_manager()
            mng.full_screen_toggle()
            for i in range(12):
                plt.subplot(3, 4, index12[i])
                plt.plot(log_feet_v_cmd[:, int(i % 3), int(i / 3)], "r")
                plt.plot(log_vel_f_cmd[:, int(i % 3), int(i / 3)], "b")
                plt.legend(["Ref", "Cmd"])
                plt.ylabel(lgd4[int(i / 3)] + lgd3[int(i % 3)])
            plt.suptitle("Feet velocity task (from IK)")
            plt.show(block=False)
            plt.savefig(
                filename + "feet_vel_test_" + N_test + "_" + N_size + ".eps",
                dpi=150,
                bbox_inches="tight",
            )
            plt.figure()
            mng = plt.get_current_fig_manager()
            mng.full_screen_toggle()
            for i in range(12):
                plt.subplot(3, 4, index12[i])
                plt.plot(log_feet_a_cmd[:, int(i % 3), int(i / 3)], "r")
                plt.plot(log_acc_f_cmd[:, int(i % 3), int(i / 3)], "b")
                plt.legend(["Ref", "Cmd"])
                plt.ylabel(lgd4[int(i / 3)] + lgd3[int(i % 3)])
            plt.suptitle("Feet acceleration task (from IK)")
            plt.show(block=False)
            plt.savefig(
                filename + "feet_acc_test_" + N_test + "_" + N_size + ".eps",
                dpi=150,
                bbox_inches="tight",
            )

    def test_task_tracking(self, ctc=0.0):
        return
        DISPLAY = False

        # Load robot model and data
        from example_robot_data.robots_loader import Solo12Loader

        Solo12Loader.free_flyer = True
        solo = Solo12Loader().robot
        q = solo.q0.reshape((-1, 1))
        q[2, 0] = self.params.h_ref
        q[7:, 0] = self.params.q_init
        dq = np.zeros((18, 1))

        # Get foot indexes
        # BASE_ID = solo.model.getFrameId("base_link")
        foot_ids = [
            solo.model.getFrameId("FL_FOOT"),
            solo.model.getFrameId("FR_FOOT"),
            solo.model.getFrameId("HL_FOOT"),
            solo.model.getFrameId("HR_FOOT"),
        ]

        # Initialization of viewer
        if DISPLAY:
            solo.initViewer(loadModel=True)
            if "viewer" in solo.viz.__dict__:
                solo.viewer.gui.addFloor("world/floor")
                solo.viewer.gui.setRefreshIsSynchronous(False)
            solo.display(q)

        # References for tracking tasks
        feet_p_cmd0 = np.array(
            [
                [0.1946, 0.1946, -0.1946, -0.1946],
                [0.16891, -0.16891, 0.16891, -0.16891],
                [0.0191028, 0.0191028, 0.0191028, 0.0191028],
            ]
        )
        feet_p_cmd0[2, :] = 0.05
        o_feet_p_cmd = np.zeros((3, 4))
        o_feet_v_cmd = np.zeros((3, 4))
        o_feet_a_cmd = np.zeros((3, 4))
        xgoals = np.zeros((12, 1))
        q_wbc = np.zeros((18, 1))
        posf = np.zeros((3, 4))
        velf = np.zeros((3, 4))

        k = 0
        N = 10000
        log_vel_cmd = np.zeros((N, 3, 4))
        log_vel_f = np.zeros((N, 3, 4))
        log_pos_cmd = np.zeros((N, 3, 4))
        log_pos_f = np.zeros((N, 3, 4))
        log_dq = np.zeros((N, 18))
        vy = 0.02
        wyaw = 5.0 / 57.0
        y_ref = 0.0
        yaw_ref = 0.0
        while k < N:

            # Position in base frame
            sig = np.sign(np.sin(2 * np.pi * 0.2 * k * self.params.dt_wbc))
            y_ref += vy * self.params.dt_wbc * sig
            yaw_ref += wyaw * self.params.dt_wbc * sig

            # Rotation from world to base frame
            oRb = pin.Quaternion(q[3:7, 0:1]).toRotationMatrix()

            # Roll pitch yaw vector
            RPY = pin.rpy.matrixToRpy(oRb)

            # Rotation between frames
            hRb = pin.rpy.rpyToMatrix(RPY[0], RPY[1], 0.0)
            oRh = pin.rpy.rpyToMatrix(0.0, 0.0, RPY[2])
            oTh = np.array([[q[0, 0]], [q[1, 0]], [q[2, 0]]])

            # Target positions of feet in base frame
            o_feet_p_cmd = feet_p_cmd0.copy()
            o_feet_v_cmd = np.zeros((3, 4))
            o_feet_a_cmd = np.zeros((3, 4))
            o_feet_p_cmd[0, 0] += 0.04 * np.sin(
                2 * np.pi * 0.25 * k * self.params.dt_wbc
            )
            o_feet_p_cmd[1, 1] += 0.04 * np.sin(
                2 * np.pi * 0.25 * k * self.params.dt_wbc
            )
            o_feet_p_cmd[2, 2] += 0.04 * np.sin(
                2 * np.pi * 0.25 * k * self.params.dt_wbc
            )
            o_feet_v_cmd[0, 0] += (
                0.04
                * 2
                * np.pi
                * 0.25
                * np.cos(2 * np.pi * 0.25 * k * self.params.dt_wbc)
            )
            o_feet_v_cmd[1, 1] += (
                0.04
                * 2
                * np.pi
                * 0.25
                * np.cos(2 * np.pi * 0.25 * k * self.params.dt_wbc)
            )
            o_feet_v_cmd[2, 2] += (
                0.04
                * 2
                * np.pi
                * 0.25
                * np.cos(2 * np.pi * 0.25 * k * self.params.dt_wbc)
            )
            o_feet_a_cmd[0, 0] += (
                -0.04
                * (2 * np.pi * 0.25) ** 2
                * np.sin(2 * np.pi * 0.25 * k * self.params.dt_wbc)
            )
            o_feet_a_cmd[1, 1] += (
                -0.04
                * (2 * np.pi * 0.25) ** 2
                * np.sin(2 * np.pi * 0.25 * k * self.params.dt_wbc)
            )
            o_feet_a_cmd[2, 2] += (
                -0.04
                * (2 * np.pi * 0.25) ** 2
                * np.sin(2 * np.pi * 0.25 * k * self.params.dt_wbc)
            )

            # Target positions of feet in horizontal frame
            feet_p_cmd = oRh.transpose() @ (o_feet_p_cmd - oTh)
            feet_v_cmd = oRh.transpose() @ o_feet_v_cmd
            feet_a_cmd = oRh.transpose() @ o_feet_a_cmd

            # Express feet so that they follow base orientation
            # feet_p_cmd = hRb @ feet_p_cmd
            # feet_v_cmd = hRb @ feet_v_cmd
            # feet_a_cmd = hRb @ feet_a_cmd

            # Goal is 20 degrees in pitch
            xgoals[4, 0] = -10.0 / 57

            # xgoals[1, 0] += 0.02 * sig * self.params.dt_wbc
            xgoals[7, 0] = vy * sig

            # xgoals[5, 0] += (10.0 / 57) * sig * self.params.dt_wbc
            xgoals[11, 0] = wyaw * sig

            # Make q_wbc vector
            q_wbc[:3, 0] = np.zeros(3)  # Position
            q_wbc[3, 0] = RPY[0]  # Roll
            q_wbc[4, 0] = RPY[1]  # Pitch
            q_wbc[5, 0] = 0.0  # Yaw
            q_wbc[6:, 0] = q[7:, 0]  # Actuators

            # Run InvKin + WBC QP
            self.wbcWrapper.compute(
                q_wbc,
                dq,
                np.zeros(12),
                np.array([[0.0, 0.0, 0.0, ctc]]),
                feet_p_cmd,
                feet_v_cmd,
                feet_a_cmd,
                xgoals,
            )

            # Velocity integration
            dq[0:3, 0:1] += (
                self.params.dt_wbc
                * hRb.transpose()
                @ self.wbcWrapper.ddq_cmd[0:3].reshape((-1, 1))
            )
            dq[3:6, 0:1] += (
                self.params.dt_wbc
                * hRb.transpose()
                @ self.wbcWrapper.ddq_cmd[3:6].reshape((-1, 1))
            )
            dq[6:, 0] += self.params.dt_wbc * self.wbcWrapper.ddq_cmd[6:]

            # Position integration
            q[:, 0] = pin.integrate(solo.model, q, self.params.dt_wbc * dq)

            # Checks
            pin.computeJointJacobians(solo.model, solo.data, q)
            pin.forwardKinematics(solo.model, solo.data, q, dq, np.zeros(solo.model.nv))
            pin.updateFramePlacements(solo.model, solo.data)
            # Get data required by IK with Pinocchio
            for i_ee in range(4):
                idx = int(foot_ids[i_ee])
                posf[:, i_ee] = solo.data.oMf[idx].translation
                velf[:, i_ee] = pin.getFrameVelocity(
                    solo.model, solo.data, int(idx), pin.LOCAL_WORLD_ALIGNED
                ).linear

            oTh = np.array([[q[0, 0]], [q[1, 0]], [q[2, 0]]])
            RPY = pin.rpy.matrixToRpy(pin.Quaternion(q[3:7, 0]).toRotationMatrix())
            oRh = pin.rpy.rpyToMatrix(0.0, 0.0, RPY[2])

            if DISPLAY:
                log_vel_cmd[k, :, :] = o_feet_v_cmd
                log_vel_f[k, :, :] = velf
                log_pos_cmd[k, :, :] = o_feet_p_cmd
                log_pos_f[k, :, :] = posf
                log_dq[k, :] = dq[:, 0]

            # self.assertTrue(
            # np.allclose(oRh.transpose() @ (posf - oTh), feet_p_cmd, atol=2e-3),
            # "feet pos tracking is OK",
            # )
            # self.assertTrue(
            # np.allclose(oRh.transpose() @ velf, feet_v_cmd, atol=2e-3),
            # "feet vel tracking is OK",
            # )

            k += 1

            if DISPLAY and (k % 10 == 0):
                solo.display(q)

        if DISPLAY:
            from matplotlib import pyplot as plt

            plt.figure()
            index12 = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
            for i in range(12):
                plt.subplot(3, 4, index12[i])
                plt.plot(log_pos_cmd[:, int(i % 3), int(i / 3)], "r")
                plt.plot(log_pos_f[:, int(i % 3), int(i / 3)], "b")
            plt.figure()
            for i in range(12):
                plt.subplot(3, 4, index12[i])
                plt.plot(log_vel_cmd[:, int(i % 3), int(i / 3)], "r")
                plt.plot(log_vel_f[:, int(i % 3), int(i / 3)], "b")
            plt.figure()
            for i, j in enumerate([0, 1, 5]):
                plt.subplot(3, 1, i + 1)
                plt.plot(log_dq[:, j], "b")
            plt.show()

    def test_task_contact(self):
        return
        self.test_task_tracking(ctc=1.0)

    def test_upper(self):
        self.assertEqual("foo".upper(), "FOO")


if __name__ == "__main__":
    unittest.main()

"""
# Feet command position, velocity and acceleration in base frame
# Use ideal base frame
self.feet_a_cmd = self.ftg.getFootAccelerationBaseFrame(
    get_oRh(k).transpose(), np.zeros((3, 1)), np.zeros((3, 1)))
self.feet_v_cmd = self.ftg.getFootVelocityBaseFrame(
    get_oRh(k).transpose(), np.zeros((3, 1)), np.zeros((3, 1)))
self.feet_p_cmd = self.ftg.getFootPositionBaseFrame(
    get_oRh(k).transpose(), get_oTh(k) + np.array([[0.0], [0.0], [self.params.h_ref]]))
"""
