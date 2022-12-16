import unittest
import numpy as np

# import pinocchio as pin

# Import classes to test
import quadruped_reactive_walking as lqrw

# Tune numpy output
np.set_printoptions(precision=6, linewidth=300)


class TestFootTrajectoryGenerator(unittest.TestCase):
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

        # Create Gait class and initialize it
        self.gait = lqrw.Gait()
        self.gait.initialize(self.params)

        # Create FootTrajectoryGenerator class and initialize it
        self.ftg = lqrw.FootTrajectoryGenerator()
        self.ftg.initialize(self.params, self.gait)

    def tearDown(self):
        pass

    def test_non_moving(self):
        """
        Check footsteps targets in a basic non-moving situation
        """

        # Footsteps should be at the vertical of shoulders in a non-moving situation
        ref = np.array(self.params.footsteps_under_shoulders.tolist()).reshape(
            (3, 4), order="F"
        )

        for k in range(500):
            # Update gait
            self.gait.update(k, 0)

            # Update pos, vel and acc references for feet
            self.ftg.update(k, ref)

            # Target in ideal world
            o_tgt = self.ftg.get_target_position()
            self.assertTrue(np.allclose(ref, o_tgt), "o_tgt is OK")

    def test_non_moving_sin(self):
        """
        Check footsteps in a basic non-moving situation with a sinus reference
        Check limits conditions (correct pos, 0 vel, 0 acc)
        """

        # Moving reference
        t = np.round(
            np.linspace(
                0,
                (self.params.N_SIMULATION - 1) * self.params.dt_wbc,
                self.params.N_SIMULATION,
            ),
            3,
        )
        x = np.random.random() * np.sin(2 * np.pi * t)
        y = np.random.random() * np.sin(2 * np.pi * t)
        z = np.zeros(len(t))
        osc = np.array([x, y, z])

        # Footsteps should be at the vertical of shoulders in a non-moving situation
        ref = np.array(self.params.footsteps_under_shoulders.tolist()).reshape(
            (3, 4), order="F"
        )
        ref_lock = ref.copy()
        t_lock = np.round(
            self.params.T_gait * 0.5
            - self.params.lock_time
            - self.params.vert_time
            - self.params.dt_wbc,
            3,
        )
        # N_ref = self.params.gait.shape[0]

        for k in range(self.params.N_SIMULATION - 1):
            # print("#### ", k)
            # print("t[k]: ", t[k])

            # Update gait
            self.gait.update(k, 0)
            cgait = self.gait.matrix

            # Update pos, vel and acc references for feet
            self.ftg.update(k, ref + osc[:, k : (k + 1)])

            # Target in ideal world
            o_tgt = self.ftg.get_target_position()
            o_pos = self.ftg.get_foot_position()
            o_vel = self.ftg.get_foot_velocity()
            o_acc = self.ftg.get_foot_acceleration()

            # print("Ref:\n", ref + osc[:, k:(k+1)])
            # print("o_tgt:\n", o_tgt)
            # print("o_pos:\n", o_pos)

            for j in range(4):
                # Check target location on the ground
                if cgait[0, j] == 1:
                    self.assertTrue(
                        np.allclose(ref_lock[:, j], o_tgt[:, j]), "o_tgt is OK"
                    )
                else:
                    tmod = np.round(t[k] % (self.params.T_gait * 0.5), 3)
                    if tmod == t_lock:
                        # Lock before touchdown
                        ref_lock[:, j] = ref[:, j] + osc[:, k]
                        self.assertTrue(
                            np.allclose(ref_lock[:, j], o_tgt[:, j]), "o_tgt is OK"
                        )
                    elif tmod > t_lock:
                        self.assertTrue(
                            np.allclose(ref_lock[:, j], o_tgt[:, j]), "o_tgt is OK"
                        )
                    else:
                        self.assertTrue(
                            np.allclose(ref[:, j] + osc[:, k], o_tgt[:, j]),
                            "o_tgt is OK",
                        )

                # Check 3D output of generator (pos, vel, acc)
                if cgait[0, j] == 1:
                    self.assertTrue(
                        np.allclose(o_tgt[:, j], o_pos[:, j]), "o_pos is OK"
                    )
                    self.assertTrue(o_pos[2, j] < 1e-8, "o_pos is OK")
                    self.assertTrue(
                        np.allclose(np.zeros(3), o_vel[:, j]), "o_vel is OK"
                    )
                    self.assertTrue(
                        np.allclose(np.zeros(3), o_acc[:, j]), "o_acc is OK"
                    )
                else:
                    tmod = np.round(t[k] % (self.params.T_gait * 0.5), 3)
                    if (tmod < self.params.vert_time) or (
                        tmod
                        >= np.round(self.params.T_gait * 0.5 - self.params.vert_time, 3)
                    ):
                        self.assertTrue(
                            np.allclose(ref_lock[:2, j], o_pos[:2, j]), "o_pos is OK"
                        )
                        self.assertTrue(
                            np.allclose(np.zeros(2), o_vel[:2, j]), "o_vel is OK"
                        )
                        self.assertTrue(
                            np.allclose(np.zeros(2), o_acc[:2, j]), "o_acc is OK"
                        )
                    """
                    if tmod == 0 or tmod == np.round(
                        self.params.T_gait * 0.5 - self.params.dt_wbc, 3
                    ):
                        self.assertTrue(o_pos[2, j] < 1e-8, "o_pos is OK")
                        self.assertTrue(o_vel[2, j] < 1e-8, "o_vel is OK")
                        self.assertTrue(o_acc[2, j] < 1e-8, "o_acc is OK")
                    """

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
