import unittest
import numpy as np
import time
import quadruped_reactive_walking as lqrw

# Import classes to test
from quadruped_reactive_walking import MPC_Wrapper

np.set_printoptions(precision=3, linewidth=300)


class TestMPC(unittest.TestCase):
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
        self.params.enable_multiprocessing = True
        q_init = np.zeros((18, 1))
        q_init[2] = self.params.h_ref
        q_init[6:, 0] = np.array(
            [0.0, 0.7, -1.4, -0.0, 0.7, -1.4, 0.0, -0.7, +1.4, -0.0, -0.7, +1.4]
        )
        for i in range(12):
            self.params.q_init[i] = q_init[6 + i, 0]
        self.params.N_periods = 1
        gait = [12, 0, 1, 1, 0, 12, 1, 0, 0, 1]
        for i in range(len(gait)):
            self.params.gait_vec[i] = gait[i]

        # Force refresh of gait matrix
        self.params.convert_gait_vec()

        # Initialization of params
        self.params.initialize()

        # Inialization of the MPC wrapper
        self.mpc_wrapper = MPC_Wrapper.MPC_Wrapper(self.params, q_init.ravel())

        # Misc arrays
        self.xref = np.zeros((12, 1 + self.params.gait.shape[0]))
        self.gait = np.zeros((self.params.gait.shape[0], 4))
        self.fsteps = np.zeros((self.params.gait.shape[0], 12))

    def tearDown(self):
        self.mpc_wrapper.stop_parallel_loop()  # Stop MPC running in parallel process

    def test_fourstance_immobile(self):
        """
        Four feet in stance phase. Check if robot stays centered.
        """

        x_test = np.zeros((12, 100))
        f_test = np.zeros((12, 100))

        # All feet in contact
        self.gait[:, :] = 1.0

        # Stay at reference height
        self.xref[2, :] = self.params.h_ref

        # Feet at default positions
        for i in range(24):
            self.fsteps[i, :] = np.array(
                [
                    0.195,
                    0.147,
                    0.0,
                    0.195,
                    -0.147,
                    0.0,
                    -0.195,
                    0.147,
                    0.0,
                    -0.195,
                    -0.147,
                    0.0,
                ]
            )

        # First iteration returns [0.0, 0.0, 8.0] for all feet in contact
        self.mpc_wrapper.solve(0, self.xref, self.fsteps, self.gait, np.zeros((3, 4)))
        while not self.mpc_wrapper.newResult.value:
            time.sleep(0.002)
        x_f_mpc, cost_mpc = self.mpc_wrapper.get_latest_result()

        # Second iteration should return [0.0, 0.0, mass/4] for all feet
        for i in range(1, 100):
            self.mpc_wrapper.solve(
                i, self.xref, self.fsteps, self.gait, np.zeros((3, 4))
            )
            while not self.mpc_wrapper.newResult.value:
                time.sleep(0.002)
            # Retrieve last result of MPC
            (
                x_f_mpc,
                cost_mpc,
            ) = self.mpc_wrapper.get_latest_result()
            x_test[:, i] = x_f_mpc[:12, 0]
            f_test[:, i] = x_f_mpc[12:, 0]

            # Update current state of the robot
            self.xref[:, 0] = x_f_mpc[:12, 0].copy()

        self.assertTrue(
            np.allclose(
                x_f_mpc[12:, 0],
                np.array([x_f_mpc[12, 0], x_f_mpc[13, 0], x_f_mpc[14, 0]] * 4),
            ),
            "All feet forces are equal.",
        )
        self.assertTrue(
            np.allclose(x_f_mpc[:12, 0], self.xref[:, 1], atol=1e-3),
            "Close to reference state",
        )
        self.assertTrue(
            np.allclose(
                x_f_mpc[12:, 0], np.array([0.0, 0.0, 9.81 * self.params.mass / 4] * 4)
            ),
            "Feet forces are equal to theorical value.",
        )

    def test_fourstance_not_centered(self):
        """
        Four feet in stance phase. Check if robot gets back to centered.
        """

        # All feet in contact
        self.gait[:, :] = 1.0

        # Stay at reference height
        self.xref[2, :] = self.params.h_ref

        # Feet at default positions
        for i in range(24):
            self.fsteps[i, :] = np.array(
                [
                    0.195,
                    0.147,
                    0.0,
                    0.195,
                    -0.147,
                    0.0,
                    -0.195,
                    0.147,
                    0.0,
                    -0.195,
                    -0.147,
                    0.0,
                ]
            )

        # Non centered state
        self.xref[:, 0] = np.array(
            [0.05, 0.05, 0.2, 0.1, 0.1, 0.1, 0.01, 0.01, 0.04, 0.4, 0.4, 0.4]
        )

        # First iteration
        self.mpc_wrapper.solve(0, self.xref, self.fsteps, self.gait, np.zeros((3, 4)))
        while not self.mpc_wrapper.newResult.value:
            time.sleep(0.002)
        x_f_mpc, cost_mpc = self.mpc_wrapper.get_latest_result()

        # Run the mpc
        for i in range(1, 500):
            self.mpc_wrapper.solve(
                i, self.xref, self.fsteps, self.gait, np.zeros((3, 4))
            )
            while not self.mpc_wrapper.newResult.value:
                time.sleep(0.002)
            # Retrieve last result of MPC
            (
                x_f_mpc,
                cost_mpc,
            ) = self.mpc_wrapper.get_latest_result()
            # Update current state of the robot
            self.xref[:, 0] = x_f_mpc[:12, 0].copy()

        self.assertTrue(
            np.allclose(
                x_f_mpc[12:, 0],
                np.array([x_f_mpc[12, 0], x_f_mpc[13, 0], x_f_mpc[14, 0]] * 4),
            ),
            "All feet forces are equal.",
        )
        self.assertTrue(
            np.allclose(x_f_mpc[:12, 0], self.xref[:, 1], atol=1e-3),
            "Close to reference state",
        )

    def test_twostance_centered(self):
        """
        Two feet in stance phase at a given time (FL-HR and FR-HL). Check if robot stays centered.
        """

        # Default trotting gait
        self.gait = self.params.gait.copy()
        self.fsteps[:12, :] = np.array(
            [0.195, 0.147, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.195, -0.147, 0.0]
        )
        self.fsteps[12:, :] = np.array(
            [0.0, 0.0, 0.0, 0.195, -0.147, 0.0, -0.195, 0.147, 0.0, 0.0, 0.0, 0.0]
        )

        # Age gait by one iteration
        self.gait = np.roll(self.gait, -1, axis=0)
        self.fsteps = np.roll(self.fsteps, -1, axis=0)

        # Stay at reference height
        self.xref[2, :] = self.params.h_ref

        # Run the mpc
        for i in range(0, 500):
            self.mpc_wrapper.solve(
                i, self.xref, self.fsteps, self.gait, np.zeros((3, 4))
            )
            while not self.mpc_wrapper.newResult.value:
                time.sleep(0.002)
            # Retrieve last result of MPC
            (
                x_f_mpc,
                cost_mpc,
            ) = self.mpc_wrapper.get_latest_result()

            # Age gait by one iteration
            self.gait = np.roll(self.gait, -1, axis=0)
            self.fsteps = np.roll(self.fsteps, -1, axis=0)

            if i > 0:
                # Update current state of the robot
                self.xref[:, 0] = x_f_mpc[:12, 0].copy()

        self.assertTrue(
            np.allclose(x_f_mpc[:12, 0], self.xref[:, 1], atol=1e-2),
            "Close to reference state",
        )

    def test_twostance_not_centered(self):
        """
        Two feet in stance phase at a given time (FL-HR and FR-HL). Check if robot gets back to centered.
        """

        N = 1000
        x_test = np.zeros((12, N))
        t_test = [i * self.params.dt_mpc for i in range(N)]
        # [2.0, 2.0, 10.0, 0.25, 0.25, 10.0, 0.2, 0.2, 0.2, 0.0, 0.0, 0.3]
        osqp_w_states = [4.0, 4.0, 1.0, 0.1, 0.1, 0.1, 0.4, 0.4, 1.0, 0.0, 0.0, 0.1]
        osqp_w_states = [
            10.0,
            10.0,
            10.0,
            0.001,
            0.001,
            0.001,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        for i in range(3):
            osqp_w_states[6 + i] = 2.0 * np.sqrt(osqp_w_states[i])
        for i in range(len(osqp_w_states)):
            self.params.osqp_w_states[i] = osqp_w_states[i]

        # Default trotting gait
        self.gait = self.params.gait.copy()
        self.fsteps[:12, :] = np.array(
            [0.195, 0.147, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.195, -0.147, 0.0]
        )
        self.fsteps[12:, :] = np.array(
            [0.0, 0.0, 0.0, 0.195, -0.147, 0.0, -0.195, 0.147, 0.0, 0.0, 0.0, 0.0]
        )

        # Age gait by one iteration
        self.gait = np.roll(self.gait, -1, axis=0)
        self.fsteps = np.roll(self.fsteps, -1, axis=0)

        # Stay at reference height
        self.xref[2, :] = self.params.h_ref

        # Non centered state
        self.xref[:, 0] = np.array(
            [0.05, 0.05, 0.2, 0.1, 0.1, 0.1, 0.01, 0.01, 0.04, 0.4, 0.4, 0.4]
        )

        # Run the mpc
        for i in range(0, N):
            self.mpc_wrapper.solve(
                i, self.xref, self.fsteps, self.gait, np.zeros((3, 4))
            )
            while not self.mpc_wrapper.newResult.value:
                time.sleep(0.002)
            # Retrieve last result of MPC
            (
                x_f_mpc,
                cost_mpc,
            ) = self.mpc_wrapper.get_latest_result()

            # Age gait by one iteration
            self.gait = np.roll(self.gait, -1, axis=0)
            self.fsteps = np.roll(self.fsteps, -1, axis=0)

            if i > 0:
                # Update current state of the robot
                self.xref[:, 0] = x_f_mpc[:12, 0].copy()
                x_test[:, i] = x_f_mpc[:12, 0].copy()

        """from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(t_test, x_test[:3, :].transpose())
        plt.figure()
        plt.plot(x_test[6:9, :].transpose())
        plt.show()

        print(x_f_mpc[:12, 0] - self.xref[:, 1])"""

        self.assertTrue(
            np.allclose(x_f_mpc[:12, 0], self.xref[:, 1], atol=1e-2),
            "Close to reference state",
        )

    def test_twostance_consistent(self):
        """
        Two feet in stance phase at a given time (FL-HR and FR-HL). Check if MPC outputs consistent results.
        """

        # Default trotting gait
        self.gait = self.params.gait.copy()
        self.fsteps[:12, :] = np.array(
            [0.195, 0.147, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.195, -0.147, 0.0]
        )
        self.fsteps[12:, :] = np.array(
            [0.0, 0.0, 0.0, 0.195, -0.147, 0.0, -0.195, 0.147, 0.0, 0.0, 0.0, 0.0]
        )

        # Age gait by one iteration
        self.gait = np.roll(self.gait, -1, axis=0)
        self.fsteps = np.roll(self.fsteps, -1, axis=0)

        # Stay at reference height while going forward
        Vx = 0.05
        self.xref[0, :] = (
            Vx * self.params.dt_mpc * np.array([i for i in range(self.xref.shape[1])])
        )
        self.xref[2, :] = self.params.h_ref
        self.xref[6, :] = Vx

        # Memory
        x_f_mpc_mem = np.zeros((12,))

        # Create fsteps matrix
        shoulders = np.zeros((3, 4))
        shoulders[0, :] = [0.1946, 0.1946, -0.1946, -0.1946]
        shoulders[1, :] = [0.14695, -0.14695, 0.14695, -0.14695]
        shoulders[2, :] = np.zeros(4)

        # Run the mpc
        N = 50
        log_x_f_mpc = np.zeros((24, self.xref.shape[1] - 1, N))
        for i in range(0, N):
            offset = np.repeat((self.fsteps[:, ::3] != 0), 3, axis=1) * np.tile(
                np.array([Vx * self.params.dt_mpc * i, 0.0, 0.0] * 4),
                (self.fsteps.shape[0], 1),
            )
            self.mpc_wrapper.solve(
                i, self.xref, self.fsteps - offset, self.gait, shoulders
            )
            while not self.mpc_wrapper.newResult.value:
                time.sleep(0.002)
            # Retrieve last result of MPC
            (
                x_f_mpc,
                cost_mpc,
            ) = self.mpc_wrapper.get_latest_result()
            x_f_mpc = x_f_mpc[:24, :]

            # Logging
            log_x_f_mpc[:, :, i] = x_f_mpc

            # Age gait by one iteration
            self.gait = np.roll(self.gait, -1, axis=0)
            self.fsteps = np.roll(self.fsteps, -1, axis=0)

            if i > 0:
                self.xref[0, 0] = 0.0
                self.xref[1, 0] = 0.0
                self.xref[2:5, 0] = x_f_mpc[2:5, 0]
                self.xref[5, 0] = 0.0
                self.xref[6:, 0] = x_f_mpc[6:12, 0]

            if i > 2:
                # Do not monitor x, y, yaw since they are reset at each iteration
                self.assertTrue(
                    np.allclose(
                        x_f_mpc[2:5, 0], x_f_mpc_mem[2:5], atol=1e-3, rtol=1e-1
                    ),
                    "Output not consistent",
                )
                self.assertTrue(
                    np.allclose(
                        x_f_mpc[6:9, 0], x_f_mpc_mem[6:9], atol=1e-3, rtol=1e-1
                    ),
                    "Output not consistent",
                )
                self.assertTrue(
                    np.allclose(
                        x_f_mpc[9:12, 0], x_f_mpc_mem[9:12], atol=1e-1, rtol=1e-1
                    ),
                    "Output not consistent",
                )
            if i > 0:
                x_f_mpc_mem = x_f_mpc[:12, 1]

        """
        index6 = [1, 3, 5, 2, 4, 6]
        index12 = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
        titles = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
        t_range = np.linspace(0.0, self.params.dt_mpc * (self.xref.shape[1]-1), self.xref.shape[1])
        import matplotlib.pyplot as plt
        plt.figure()
        for i in range(6):
            plt.subplot(3, 2, index6[i])
            for j in range(N):
                h1, = plt.plot(t_range[1:] + self.params.dt_mpc * j, log_x_f_mpc[i+6, :, j], linewidth=2)
            h3, = plt.plot(t_range, self.xref[i+6, :], "b", linestyle="--", marker='x', color="k", linewidth=2)
            plt.xlabel("Time [s]")
            if i in [0, 1]:
                m1 = np.min(log_x_f_mpc[6:8, :, :])
                M1 = np.max(log_x_f_mpc[6:8, :, :])
                plt.ylim([m1 - 0.01, M1 + 0.01])
            elif i in [3, 4]:
                m1 = np.min(log_x_f_mpc[9:11, :, :])
                M1 = np.max(log_x_f_mpc[9:11, :, :])
                plt.ylim([m1 - 0.01, M1 + 0.01])
            plt.title("Predicted trajectory for velocity in " + titles[i])
        plt.suptitle("Analysis of trajectories of linear and angular velocities computed by the MPC")
        plt.show()
        """

    def test_fourstance_jump(self):
        """
        Four feet in stance phase before jumping.
        """

        return

        # Initial position of joints
        q_init = np.zeros((18, 1))
        q_init[6:, 0] = np.array(
            [0.0, 0.7, -1.4, -0.0, 0.7, -1.4, 0.0, -0.7, +1.4, -0.0, -0.7, +1.4]
        )
        for i in range(12):
            self.params.q_init[i] = q_init[6 + i, 0]

        # Jumping gait during 0.48s
        gait = [18, 1, 1, 1, 1, 12, 0, 0, 0, 0, 18, 1, 1, 1, 1]
        for i in range(len(gait)):
            self.params.gait_vec[i] = gait[i]

        # Force refresh of gait matrix
        self.params.convert_gait_vec()

        # Initialization of params
        self.params.initialize()

        # Inialization of the MPC wrapper
        self.mpc_wrapper = MPC_Wrapper.MPC_Wrapper(self.params, q_init.ravel())

        # Misc arrays
        self.xref = np.zeros((12, 1 + self.params.gait.shape[0]))
        self.gait = np.zeros((self.params.gait.shape[0], 4))
        self.fsteps = np.zeros((self.params.gait.shape[0], 12))

        # Default trotting gait
        self.gait = self.params.gait.copy()
        self.fsteps[:18, :] = np.array(
            [
                0.195,
                0.147,
                0.195,
                0.195,
                -0.147,
                0.0,
                -0.195,
                0.147,
                0.0,
                -0.195,
                -0.147,
                0.0,
            ]
        )
        self.fsteps[30:, :] = np.array(
            [
                0.195,
                0.147,
                0.195,
                0.195,
                -0.147,
                0.0,
                -0.195,
                0.147,
                0.0,
                -0.195,
                -0.147,
                0.0,
            ]
        )

        # Jump
        Tz = 0.24
        maxHeight_ = 0.1
        Az = np.zeros((4, 1))
        Az[0, 0] = -maxHeight_ / ((Tz / 2) ** 3 * (Tz - Tz / 2) ** 3)
        Az[1, 0] = (3 * Tz * maxHeight_) / ((Tz / 2) ** 3 * (Tz - Tz / 2) ** 3)
        Az[2, 0] = -(3 * Tz**2 * maxHeight_) / ((Tz / 2) ** 3 * (Tz - Tz / 2) ** 3)
        Az[3, 0] = (Tz**3 * maxHeight_) / ((Tz / 2) ** 3 * (Tz - Tz / 2) ** 3)

        self.xref[2, 0] = self.params.h_ref
        for i in range(1, self.xref.shape[1]):
            dtz = ((i * 20) % 360) * self.params.dt_wbc
            if (i * 20) <= 360 or (i * 20) > (360 + 240):
                self.xref[2, i] = self.params.h_ref
                self.xref[8, i] = 0.0
            else:
                self.xref[2, i] = (
                    self.params.h_ref
                    + Az[3, 0] * dtz**3
                    + Az[2, 0] * dtz**4
                    + Az[1, 0] * dtz**5
                    + Az[0, 0] * dtz**6
                )
                self.xref[8, i] = (
                    3 * Az[3, 0] * dtz**2
                    + 4 * Az[2, 0] * dtz**3
                    + 5 * Az[1, 0] * dtz**4
                    + 6 * Az[0, 0] * dtz**5
                )

        # Create fsteps matrix
        shoulders = np.zeros((3, 4))
        shoulders[0, :] = [0.1946, 0.1946, -0.1946, -0.1946]
        shoulders[1, :] = [0.14695, -0.14695, 0.14695, -0.14695]
        shoulders[2, :] = np.zeros(4)

        # Run the mpc
        for i in range(2):
            self.mpc_wrapper.solve(i, self.xref, self.fsteps, self.gait, shoulders)
            while not self.mpc_wrapper.newResult.value:
                time.sleep(0.002)
            x_f_mpc, cost_mpc = self.mpc_wrapper.get_latest_result()[
                :24, :
            ]  # Retrieve last result of MPC
            x_f_mpc = x_f_mpc[:24, :]

        # Check result
        self.assertTrue(
            not np.any(x_f_mpc[[0, 1, 3, 4, 5, 6, 7, 9, 10, 11], :] > 1e-5),
            "Abnormal deviation",
        )
        self.assertTrue(
            np.max(x_f_mpc[2, :]) > maxHeight_ * 0.75, "Jump not high enough"
        )

        # Plot result
        from matplotlib import pyplot as plt

        index6 = [1, 3, 5, 2, 4, 6]
        index12 = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
        titles = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]

        t_range = np.linspace(
            0.0, self.params.dt_mpc * (self.xref.shape[1] - 1), self.xref.shape[1]
        )
        plt.figure()
        for i in range(6):
            plt.subplot(3, 2, index6[i])
            (h1,) = plt.plot(t_range[1:], x_f_mpc[i, :], linewidth=2)
            (h3,) = plt.plot(
                t_range,
                self.xref[i, :],
                "b",
                linestyle="--",
                marker="x",
                color="g",
                linewidth=2,
            )
            plt.xlabel("Time [s]")
            plt.legend([h1, h3], ["OSQP", "Ref"])
            plt.title("Predicted trajectory for " + titles[i])
        plt.suptitle(
            "Analysis of trajectories in position and orientation computed by the MPC"
        )

        plt.figure()
        for i in range(6):
            plt.subplot(3, 2, index6[i])
            (h1,) = plt.plot(t_range[1:], x_f_mpc[i + 6, :], linewidth=2)
            (h3,) = plt.plot(
                t_range,
                self.xref[i + 6, :],
                "b",
                linestyle="--",
                marker="x",
                color="k",
                linewidth=2,
            )
            plt.xlabel("Time [s]")
            plt.title("Predicted trajectory for velocity in " + titles[i])
        plt.suptitle(
            "Analysis of trajectories of linear and angular velocities computed by the MPC"
        )

        lgd1 = ["Ctct force X", "Ctct force Y", "Ctct force Z"]
        lgd2 = ["FL", "FR", "HL", "HR"]
        plt.figure()
        for i in range(12):
            if i == 0:
                ax0 = plt.subplot(3, 4, index12[i])
            else:
                plt.subplot(3, 4, index12[i], sharex=ax0)
            (h1,) = plt.plot(t_range[1:], x_f_mpc[i + 12, :], "g", linewidth=3)
            plt.xlabel("Time [s]")
            plt.ylabel(lgd1[i % 3] + " " + lgd2[int(i / 3)] + " [N]")

            if i % 3 != 2:
                idx = np.array([0, 1, 3, 4, 6, 7, 9, 10]) + 12
            else:
                idx = np.array([2, 5, 8, 11]) + 12
            m1 = np.min(x_f_mpc[idx, :])
            M1 = np.max(x_f_mpc[idx, :])
            plt.ylim([m1 - 0.2, M1 + 0.2])

            plt.legend([h1], ["Classic"])
        plt.suptitle("Contact forces (MPC command)")

        plt.show()

    def test_upper(self):
        self.assertEqual("foo".upper(), "FOO")


if __name__ == "__main__":
    unittest.main()
