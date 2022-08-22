import unittest
import numpy as np
import pinocchio as pin
import time
from example_robot_data.robots_loader import Solo12Loader

# Import classes to test
import quadruped_reactive_walking as lqrw

# Tune numpy output
np.set_printoptions(precision=6, linewidth=300)


class TestEstimator(unittest.TestCase):
    def setUp(self):

        # Object that holds all controller parameters
        self.params = lqrw.Params()

        # Set parameters to overwrite user's values
        self.params.dt_wbc = 0.001
        self.params.dt_mpc = 0.02
        self.params.N_SIMULATION = 500
        self.params.perfect_estimator = False
        self.params.solo3D = False
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

        # Create Estimator class and initialize it
        self.estimator = lqrw.Estimator()
        self.estimator.initialize(self.params)

    def tearDown(self):
        pass

    def test_frame_rotation(self):
        """
        Check rotation from world frame to base frame
        Check rotation from world frame to horizontal frame
        Check rotation from horizontal from to base frame
        Check yaw orientation in world frame
        """

        yaw_estim = 0.0
        for i in range(self.params.N_SIMULATION):

            # Random velocity reference (in horizontal frame)
            h_v_ref = np.random.random((6, 1))

            # Random roll pitch yaw
            RPY = np.random.random((3, 1))

            # Rotation in yaw is in perfect world
            yaw_estim += h_v_ref[5, 0] * self.params.dt_wbc
            oRh = pin.rpy.rpyToMatrix(0.0, 0.0, yaw_estim)
            hRb = pin.rpy.rpyToMatrix(RPY[0, 0], RPY[1, 0], 0.0)

            # Run filter
            self.estimator.run(
                self.params.gait,
                np.random.random((3, 4)),
                np.random.random((3, 1)),
                np.random.random((3, 1)),
                RPY,
                np.random.random((12, 1)),
                np.random.random((12, 1)),
                np.random.random((6, 1)),
                np.random.random((3, 1)),
            )

            # Update state
            self.estimator.update_reference_state(h_v_ref)

            # Test output values
            self.assertTrue(np.allclose(oRh, self.estimator.get_oRh()), "oRh is OK")
            self.assertTrue(np.allclose(hRb, self.estimator.get_hRb()), "hRb is OK")
            self.assertTrue(
                np.allclose(yaw_estim, self.estimator.get_q_reference()[5]),
                "yaw_estim is OK",
            )

    def test_motion_ideal(self):
        """
        Check reference motion by integration of reference velocity
        Check reference velocity in horizontal frame
        Check reference acceleration by derivation of reference velocity
        Check yaw orientation in world frame
        """

        # Velocity profile in x, y and yaw.
        # Position in x, y and yaw orientation are integrated.
        # Acceleration in x, y and yaw are derived.
        t = np.linspace(
            0,
            (self.params.N_SIMULATION - 1) * self.params.dt_wbc,
            self.params.N_SIMULATION,
        )
        vx = 1.0 * np.ones(self.params.N_SIMULATION)
        vy = np.cos(2 * np.pi * t) - 1.0
        wyaw = -(np.cos(2 * np.pi * t) - 1.0)
        x = np.cumsum(vx * self.params.dt_wbc)
        y = np.cumsum(vy * self.params.dt_wbc)
        yaw = np.cumsum(wyaw * self.params.dt_wbc)
        ax = np.diff(np.hstack((0.0, vx))) / self.params.dt_wbc
        ay = np.diff(np.hstack((0.0, vy))) / self.params.dt_wbc
        ayaw = np.hstack((0.0, np.diff(wyaw) / self.params.dt_wbc))

        # Check output of estimator
        h_v_ref = np.zeros((6, 1))
        h_a_ref = np.zeros((6, 1))
        for i in range(self.params.N_SIMULATION - 1):
            oRh = pin.rpy.rpyToMatrix(0.0, 0.0, yaw[i])
            h_v_ref[0:2, 0:1] = oRh[0:2, 0:2].transpose() @ np.array([[vx[i]], [vy[i]]])
            h_v_ref[5, 0] = wyaw[i]
            h_a_ref[0:2, 0:1] = oRh[0:2, 0:2].transpose() @ np.array([[ax[i]], [ay[i]]])
            h_a_ref[5, 0] = ayaw[i]

            # Run filter
            self.estimator.run(
                self.params.gait,
                np.random.random((3, 4)),
                np.random.random((3, 1)),
                np.random.random((3, 1)),
                np.random.random((3, 1)),
                np.random.random((12, 1)),
                np.random.random((12, 1)),
                np.random.random((6, 1)),
                np.random.random((3, 1)),
            )

            # Update state
            self.estimator.update_reference_state(h_v_ref)

            # Test output values
            self.assertTrue(
                np.allclose(np.array([x[i], y[i], 0.0]), self.estimator.get_oTh()),
                "oTh is OK",
            )
            self.assertTrue(
                np.allclose(h_v_ref.ravel(), self.estimator.get_base_vel_ref()),
                "h_v_ref is OK",
            )
            # self.assertTrue(np.allclose(h_a_ref.ravel(), self.estimator.get_base_acc_ref()), "h_a_ref is OK")
            self.assertTrue(
                np.allclose(yaw[i], self.estimator.get_q_reference()[5]),
                "yaw_estim is OK",
            )

    def test_estimation(self):
        """
        Check estimation of configuration vector by complementary filter
        Check estimation of velocity vector by complementary filter
        """

        # IMU measures linear acceleration, angular velocity and angular position
        # Acc in base frame at IMU loc
        lin_acc = np.random.random((3, self.params.N_SIMULATION))
        ang_vel = np.random.random((3, self.params.N_SIMULATION))
        ang_pos = np.cumsum(ang_vel * self.params.dt_wbc, axis=1)
        # IMU yaw angular position starts at 0 (hard reset to 0 for k = 0 and k = 1)
        ang_pos[2, 0] = 0.0
        ang_pos[2, 1:] -= ang_pos[2, 1]
        # Lever arm with IMU position to get velocity of the base
        # Acc in world frame at IMU loc
        oi_lin_acc = np.zeros((3, self.params.N_SIMULATION))
        for i in range(self.params.N_SIMULATION):
            oRb = pin.rpy.rpyToMatrix(ang_pos[0, i], ang_pos[1, i], ang_pos[2, i])
            # Acc in world frame at IMU loc
            oi_lin_acc[:, i : (i + 1)] = oRb @ lin_acc[:, i : (i + 1)]
        # Vel in world frame at IMU loc
        oi_lin_vel = np.cumsum(oi_lin_acc * self.params.dt_wbc, axis=1)
        # Vel in base frame at center trunk
        lin_vel = np.zeros((3, self.params.N_SIMULATION))
        # Vel in world frame at center trunk
        o_lin_vel = np.zeros((3, self.params.N_SIMULATION))
        for i in range(self.params.N_SIMULATION):
            oRb = pin.rpy.rpyToMatrix(ang_pos[0, i], ang_pos[1, i], ang_pos[2, i])
            lin_vel[:, i : (i + 1)] = oRb.transpose() @ oi_lin_vel[
                :, i : (i + 1)
            ] - np.cross(
                np.array([[0.1163], [0.0], [0.02]]), ang_vel[:, i : (i + 1)], axis=0
            )
            o_lin_vel[:, i : (i + 1)] = oRb @ lin_vel[:, i : (i + 1)]
        # Pos in world frame at center trunk
        lin_pos = np.cumsum(o_lin_vel * self.params.dt_wbc, axis=1)
        lin_pos[2, :] += self.params.h_ref
        # Quantities in horizontal frame
        h_lin_vel = np.zeros((3, self.params.N_SIMULATION))
        h_ang_vel = np.zeros((3, self.params.N_SIMULATION))
        for i in range(self.params.N_SIMULATION):
            hRb = pin.rpy.rpyToMatrix(ang_pos[0, i], ang_pos[1, i], 0.0)
            h_lin_vel[:, i : (i + 1)] = hRb @ lin_vel[:, i : (i + 1)]
            h_ang_vel[:, i : (i + 1)] = hRb @ ang_vel[:, i : (i + 1)]

        # Encoders measure position and velocity of the actuators
        q_12 = np.random.random((12, self.params.N_SIMULATION))
        v_12 = np.random.random((12, self.params.N_SIMULATION))

        for i in range(self.params.N_SIMULATION - 1):

            # Run filter
            self.estimator.run(
                np.zeros(self.params.gait.shape),
                np.random.random((3, 4)),
                lin_acc[:, i : (i + 1)],
                ang_vel[:, i : (i + 1)],
                ang_pos[:, i : (i + 1)],
                q_12[:, i : (i + 1)],
                v_12[:, i : (i + 1)],
                np.random.random((6, 1)),
                np.random.random((3, 1)),
            )

            # Update state
            self.estimator.update_reference_state(np.random.random((6, 1)))

            # Test output values
            q_filt = self.estimator.get_q_estimate()
            RPY_filt = pin.rpy.matrixToRpy(
                pin.Quaternion(q_filt[3:7].reshape((-1, 1))).toRotationMatrix()
            )
            v_filt = self.estimator.get_v_estimate()
            h_v_filt = self.estimator.get_h_v()

            # Velocity checks
            self.assertTrue(
                np.allclose(lin_vel[:, i], v_filt[:3]), "Linear velocity OK"
            )
            self.assertTrue(
                np.allclose(ang_vel[:, i], v_filt[3:6]), "Angular velocity OK"
            )
            self.assertTrue(np.allclose(v_12[:, i], v_filt[6:]), "Actuator velocity OK")
            self.assertTrue(
                np.allclose(h_lin_vel[:, i], h_v_filt[:3]),
                "Horizontal linear velocity OK",
            )
            self.assertTrue(
                np.allclose(h_ang_vel[:, i], h_v_filt[3:6]),
                "Horizontal angular velocity OK",
            )
            # Position checks
            self.assertTrue(
                np.allclose(lin_pos[:, i], q_filt[:3]), "Linear position OK"
            )
            self.assertTrue(
                np.allclose(ang_pos[:, i], RPY_filt.ravel()), "Angular position OK"
            )
            self.assertTrue(np.allclose(q_12[:, i], q_filt[7:]), "Actuator position OK")

    def test_estimation_windowed(self):
        """
        Check window filter over a gait period
        """

        # Time vector and gait period
        t = np.linspace(
            0,
            (self.params.N_SIMULATION - 1) * self.params.dt_wbc,
            self.params.N_SIMULATION,
        )
        T = self.params.dt_mpc * self.params.gait.shape[0] / self.params.N_periods

        # IMU measures linear acceleration, angular velocity and angular position
        lin_acc = np.array(
            [
                np.zeros(self.params.N_SIMULATION),
                np.sin(2 * np.pi * t / T),
                -np.sin(2 * np.pi * t / T),
            ]
        )
        ang_vel = np.zeros((3, 1))
        ang_pos = np.random.random((3, 1))
        hRb = pin.rpy.rpyToMatrix(ang_pos[0, 0], ang_pos[1, 0], 0.0)

        # IMU yaw angular position starts at 0 (hard reset to 0 for k = 0 and k = 1)
        ang_pos[2, 0] = 0.0

        # Encoders measure position and velocity of the actuators
        q_12 = np.random.random((12, self.params.N_SIMULATION))
        v_12 = np.random.random((12, self.params.N_SIMULATION))

        for i in range(self.params.N_SIMULATION - 1):

            # Run filter
            self.estimator.run(
                np.zeros(self.params.gait.shape),
                np.random.random((3, 4)),
                lin_acc[:, i : (i + 1)],
                ang_vel,
                ang_pos,
                q_12[:, i : (i + 1)],
                v_12[:, i : (i + 1)],
                np.random.random((6, 1)),
                np.random.random((3, 1)),
            )

            # Update state
            self.estimator.update_reference_state(np.random.random((6, 1)))

            # Test output values
            if i > int(T / self.params.dt_wbc) + 1:  # Wait one gait period
                self.assertTrue(
                    np.allclose(v_win, self.estimator.get_v_filtered()[:6]),
                    "Windowed velocity OK",
                )
                self.assertTrue(
                    np.allclose(h_v_win, self.estimator.get_h_v_filtered()),
                    "Horizontal windowed velocity OK",
                )

            # Save values for next loop
            v_win = self.estimator.get_v_filtered()[:6]
            h_v_win = self.estimator.get_h_v_filtered()
            self.assertTrue(
                np.allclose(
                    h_v_win[:3].reshape((-1, 1)), hRb @ v_win[:3].reshape((-1, 1))
                ),
                "Horizontal lin rotation OK",
            )
            self.assertTrue(
                np.allclose(
                    h_v_win[3:].reshape((-1, 1)), hRb @ v_win[3:].reshape((-1, 1))
                ),
                "Horizontal ang rotation OK",
            )

    def test_forward_kinematics(self):
        """
        Check that estimation with FK and IMU is working
        Check that estimation with FK and IMU does not drift when there is IMU noise
        Check that estimation drifts with only the IMU when there is IMU noise
        """

        # Tolerance of the estimation [m/s]
        atol_esti = 1e-2

        # Parameters of the Invkin
        l = 0.1946 * 2
        L = 0.14695 * 2
        h = self.params.h_ref
        q_init = [0.0, 0.7, -1.4, -0.0, 0.7, -1.4, 0.0, 0.7, -1.4, -0.0, 0.7, -1.4]

        # Load robot model and data
        # Initialisation of the Gepetto viewer
        Solo12Loader.free_flyer = False
        solo = Solo12Loader().robot
        q_12 = solo.q0.reshape((-1, 1))
        q_12[:, 0] = q_init  # Initial angular positions of actuators

        # Get foot indexes
        BASE_ID = solo.model.getFrameId("base_link")
        foot_ids = [
            solo.model.getFrameId("FL_FOOT"),
            solo.model.getFrameId("FR_FOOT"),
            solo.model.getFrameId("HL_FOOT"),
            solo.model.getFrameId("HR_FOOT"),
        ]

        # Init variables
        Jf = np.zeros((12, 12))
        posf = np.zeros((4, 3))
        posf_ref = np.array(
            [
                [l * 0.5, l * 0.5, -l * 0.5, -l * 0.5],
                [L * 0.5, -L * 0.5, L * 0.5, -L * 0.5],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        log_alpha = np.zeros(3 * self.params.N_SIMULATION)

        def run_IK(pos_base, q, oRb):

            b_posf_ref = oRb.transpose() @ (posf_ref - pos_base)
            pfeet_err = np.ones((3, 4))
            while np.any(np.abs(pfeet_err) > 0.0001):

                # Update model and data of the robot
                pin.computeJointJacobians(solo.model, solo.data, q)
                pin.forwardKinematics(
                    solo.model,
                    solo.data,
                    q,
                    np.zeros(solo.model.nv),
                    np.zeros(solo.model.nv),
                )
                pin.updateFramePlacements(solo.model, solo.data)

                # Get data required by IK with Pinocchio
                for i_ee in range(4):
                    idx = int(foot_ids[i_ee])
                    posf[i_ee, :] = solo.data.oMf[idx].translation
                    Jf[(3 * i_ee) : (3 * (i_ee + 1)), :] = pin.getFrameJacobian(
                        solo.model, solo.data, idx, pin.LOCAL_WORLD_ALIGNED
                    )[:3]

                # Compute errors
                pfeet_err = b_posf_ref - posf.transpose()

                # Loop
                q = pin.integrate(
                    solo.model,
                    q,
                    0.01 * np.linalg.pinv(Jf) @ pfeet_err.reshape((-1, 1), order="F"),
                )

            return q

        q_12 = run_IK(np.array([[0.0], [0.0], [h]]), q_12, np.eye(3))

        # Time vector and gait period
        t = np.linspace(
            0,
            (self.params.N_SIMULATION * 3 - 1) * self.params.dt_wbc,
            self.params.N_SIMULATION * 3,
        )
        T = self.params.dt_mpc * self.params.gait.shape[0] / self.params.N_periods

        # Trajectory of the base in position, velocity and acceleration
        osc = 0.01
        off = (2 * np.pi / T) * osc
        lin_acc = np.array(
            [
                np.zeros(self.params.N_SIMULATION * 3),
                (2 * np.pi / T) ** 2 * -osc * np.sin(2 * np.pi * t / T)
                + (2 * np.pi / T) ** 2 * osc * 0.1,
                (2 * np.pi / T) ** 2 * -osc * np.sin(2 * np.pi * t / T)
                + (2 * np.pi / T) ** 2 * osc * 0.1,
            ]
        )
        lin_vel = np.cumsum(lin_acc * self.params.dt_wbc, axis=1) + np.array(
            [[0.0], [off], [off]]
        )
        lin_pos = np.cumsum(lin_vel * self.params.dt_wbc, axis=1)
        lin_pos[2, :] += h

        ang_vel = np.array(
            [
                np.zeros(self.params.N_SIMULATION * 3),
                (2 * np.pi / T) * 0.07 * np.sin(2 * np.pi * t / T),
                (2 * np.pi / T) * 0.05 * np.sin(2 * np.pi * t / T),
            ]
        )
        ang_pos = np.cumsum(ang_vel * self.params.dt_wbc, axis=1)
        # IMU yaw angular position starts at 0 (hard reset to 0 for k = 0 and k = 1)
        ang_pos[2, 0] = 0.0
        ang_pos[2, 1:] -= ang_pos[2, 1]
        # Lever arm with IMU position to get velocity of the base
        # Acc in world frame at IMU loc
        oi_lin_acc = np.zeros((3, self.params.N_SIMULATION * 3))
        for i in range(self.params.N_SIMULATION * 3):
            oRb = pin.rpy.rpyToMatrix(ang_pos[0, i], ang_pos[1, i], ang_pos[2, i])
            # Acc in world frame at IMU loc
            oi_lin_acc[:, i : (i + 1)] = oRb @ lin_acc[:, i : (i + 1)]
        # Vel in world frame at IMU loc
        oi_lin_vel = np.cumsum(oi_lin_acc * self.params.dt_wbc, axis=1)
        # Vel in base frame at center trunk
        lin_vel = np.zeros((3, self.params.N_SIMULATION * 3))
        lin_vel_esti = np.zeros((3, self.params.N_SIMULATION * 3))
        # Vel in world frame at center trunk
        o_lin_vel = np.zeros((3, self.params.N_SIMULATION * 3))
        for i in range(self.params.N_SIMULATION * 3):
            oRb = pin.rpy.rpyToMatrix(ang_pos[0, i], ang_pos[1, i], ang_pos[2, i])
            lin_vel[:, i : (i + 1)] = oRb.transpose() @ oi_lin_vel[
                :, i : (i + 1)
            ] - np.cross(
                np.array([[0.1163], [0.0], [0.02]]), ang_vel[:, i : (i + 1)], axis=0
            )
            o_lin_vel[:, i : (i + 1)] = oRb @ lin_vel[:, i : (i + 1)]
        # Pos in world frame at center trunk
        lin_pos = np.cumsum(o_lin_vel * self.params.dt_wbc, axis=1)
        lin_pos[2, :] += self.params.h_ref

        ####
        # Testing IMU acc + FK without noise in IMU acceleration
        ####

        gait = self.params.gait.copy()

        # Loop with forward kinematics
        for i in range(self.params.N_SIMULATION):

            q_12_next = run_IK(
                lin_pos[:, i : (i + 1)],
                q_12,
                pin.rpy.rpyToMatrix(ang_pos[0, i], ang_pos[1, i], ang_pos[2, i]),
            )
            v_12 = (q_12_next - q_12) / self.params.dt_wbc
            q_12 = q_12_next

            # Gait evolution
            if i > 0 and i % int(self.params.dt_mpc / self.params.dt_wbc) == 0:
                gait = np.roll(gait, -1, axis=0)

            # Run filter
            self.estimator.run(
                gait,
                posf_ref,
                lin_acc[:, i : (i + 1)],
                ang_vel[:, i : (i + 1)],
                ang_pos[:, i : (i + 1)],
                q_12,
                v_12,
                np.random.random((6, 1)),
                np.random.random((3, 1)),
            )

            log_alpha[i] = self.estimator.get_alpha()

            # Update state
            self.estimator.update_reference_state(np.random.random((6, 1)))

            # Log estimated velocity
            lin_vel_esti[:, i] = self.estimator.get_v_estimate()[:3]

            # Test output values
            if i > int(0.5 * T / self.params.dt_wbc) + 1:  # Wait half a gait period
                self.assertTrue(
                    np.allclose(
                        lin_vel[:3, i],
                        lin_vel_esti[:, i],
                        atol=atol_esti,
                    ),
                    "Estimated velocity OK",
                )

        ####
        # Adding noise to acceleration and using only IMU acc, it should fail due to drift
        ####

        # Loop with forward kinematics
        for i in range(self.params.N_SIMULATION, 2 * self.params.N_SIMULATION):

            q_12_next = run_IK(
                lin_pos[:, i : (i + 1)],
                q_12,
                pin.rpy.rpyToMatrix(ang_pos[0, i], ang_pos[1, i], ang_pos[2, i]),
            )
            v_12 = (q_12_next - q_12) / self.params.dt_wbc
            q_12 = q_12_next

            # Run filter
            self.estimator.run(
                np.zeros(self.params.gait.shape),
                posf_ref,
                lin_acc[:, i : (i + 1)] + 3e-2 * np.random.random((3, 1)),
                ang_vel[:, i : (i + 1)],
                ang_pos[:, i : (i + 1)],
                q_12,
                v_12,
                np.random.random((6, 1)),
                np.random.random((3, 1)),
            )

            log_alpha[i] = self.estimator.get_alpha()

            # Update state
            self.estimator.update_reference_state(np.random.random((6, 1)))

            # Log estimated velocity
            lin_vel_esti[:, i] = self.estimator.get_v_estimate()[:3]

        # Test output values at the end of the loop (it should drift)
        self.assertTrue(
            not np.allclose(
                lin_vel[:3, i], self.estimator.get_v_estimate()[:3], atol=atol_esti
            ),
            "Drift of estimated velocity OK",
        )

        ####
        # Adding noise to acceleration and using IMU acc + FK, drift should be compensated by FK
        ####

        gait = self.params.gait.copy()

        # Loop with forward kinematics
        for i in range(2 * self.params.N_SIMULATION, 3 * self.params.N_SIMULATION):

            q_12_next = run_IK(
                lin_pos[:, i : (i + 1)],
                q_12,
                pin.rpy.rpyToMatrix(ang_pos[0, i], ang_pos[1, i], ang_pos[2, i]),
            )
            v_12 = (q_12_next - q_12) / self.params.dt_wbc
            q_12 = q_12_next

            # Gait evolution
            if i > 0 and i % int(self.params.dt_mpc / self.params.dt_wbc) == 0:
                gait = np.roll(gait, -1, axis=0)

            # Run filter
            self.estimator.run(
                gait,
                posf_ref,
                lin_acc[:, i : (i + 1)] + 3e-2 * np.random.random((3, 1)),
                ang_vel[:, i : (i + 1)],
                ang_pos[:, i : (i + 1)],
                q_12,
                v_12,
                np.random.random((6, 1)),
                np.random.random((3, 1)),
            )

            log_alpha[i] = self.estimator.get_alpha()

            # Update state
            self.estimator.update_reference_state(np.random.random((6, 1)))

            # Log estimated velocity
            lin_vel_esti[:, i] = self.estimator.get_v_estimate()[:3]

        # Test output values at the end of the loop (drift should have been compensated)
        self.assertTrue(
            np.allclose(
                lin_vel[:3, i], self.estimator.get_v_estimate()[:3], atol=atol_esti
            ),
            "Drift compensation OK",
        )

        """
        # Figures to understand what happens
        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(t, log_alpha)
        plt.figure()
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plt.plot(t, lin_acc[i, :] + np.random.random((3 * self.params.N_SIMULATION)))
            plt.plot(t, lin_acc[i, :], "k") 
        plt.figure()
        ylabels = ["Vel X", "Vel Y", "Vel Z"]
        for i in range(3):
            if i == 0:
                ax0 = plt.subplot(3, 1, i + 1)
            else:
                plt.subplot(3, 1, i + 1, sharex=ax0)
            plt.plot(t, lin_vel[i, :], "r")
            plt.plot(t, lin_vel_esti[i, :], "b")
            plt.legend(["Expected", "Estimated"])
            plt.ylabel(ylabels[i])
            if i == 2:
                plt.xlabel("Time [s]")
        plt.show(block=True)
        print("Refe: ", lin_vel[:3, i])
        print("Esti: ", self.estimator.get_v_estimate()[:3])
        """

    def test_upper(self):
        self.assertEqual("foo".upper(), "FOO")


if __name__ == "__main__":
    unittest.main()
