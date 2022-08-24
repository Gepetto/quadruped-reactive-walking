import unittest
import numpy as np
from numpy.lib.function_base import gradient
import pinocchio as pin
import time
from example_robot_data.robots_loader import Solo12Loader

# Import classes to test
import quadruped_reactive_walking as lqrw

# Tune numpy output
np.set_printoptions(precision=6, linewidth=300)


class TestGait(unittest.TestCase):
    def setUp(self):

        # Object that holds all controller parameters
        self.params = lqrw.Params()

        # Set parameters to overwrite user's values
        self.params.dt_wbc = 0.001
        self.params.dt_mpc = 0.02
        self.k_mpc = int(self.params.dt_mpc / self.params.dt_wbc)
        self.params.N_SIMULATION = 1000
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

    def tearDown(self):
        pass

    def test_initial(self):
        """
        Check that the past, current and desired gait matrices are correctly set up and returned
        """

        pgait = self.params.gait.copy()

        # Check initial state
        self.assertTrue(
            np.allclose(pgait, self.gait.get_past_gait()), "Initial past gait is OK"
        )
        self.assertTrue(
            np.allclose(pgait, self.gait.matrix), "Initial current gait is OK"
        )
        self.assertTrue(
            np.allclose(pgait, self.gait.get_desired_gait()),
            "Initial desired gait is OK",
        )

    def test_roll(self):
        """Check if roll is properly applied with a transfer from desired gait to current gait and from
        current gait to past gait
        """

        o_pgait = np.round(np.random.random(self.params.gait.shape))
        o_cgait = np.round(np.random.random(self.params.gait.shape))
        o_dgait = np.round(np.random.random(self.params.gait.shape))
        self.gait.set_past_gait(o_pgait)
        self.gait.set_current_gait(o_cgait)
        self.gait.set_desired_gait(o_dgait)

        # Check that set functions are working
        self.assertTrue(
            np.allclose(o_pgait, self.gait.get_past_gait()), "set_past_gait is OK"
        )
        self.assertTrue(
            np.allclose(o_cgait, self.gait.matrix), "set_current_gait is OK"
        )
        self.assertTrue(
            np.allclose(o_dgait, self.gait.get_desired_gait()), "set_desired_gait is OK"
        )

        # Oldify
        for i in range(12 * self.k_mpc + 1):
            self.gait.update(i, 0)

        # Check new values in gait matrices
        pgait = self.gait.get_past_gait()
        cgait = self.gait.matrix
        dgait = self.gait.get_desired_gait()
        self.assertTrue(
            np.allclose(o_pgait[12:, :], pgait[:12, :]), "First half past gait is OK"
        )
        self.assertTrue(
            np.allclose(o_cgait[:12, :], pgait[12:, :]), "Second half past gait is OK"
        )
        self.assertTrue(
            np.allclose(o_cgait[12:, :], cgait[:12, :]), "First half current gait is OK"
        )
        self.assertTrue(
            np.allclose(o_dgait[:12, :], cgait[12:, :]),
            "Second half current gait is OK",
        )
        self.assertTrue(
            np.allclose(o_dgait[12:, :], dgait[:12, :]), "First half desired gait is OK"
        )
        self.assertTrue(
            np.allclose(o_dgait[:12, :], dgait[12:, :]),
            "Second half desired gait is OK",
        )

    def test_phase_duration(self):
        """
        Check if getPhaseDuration computes phase duration by seeking into past gait and desired gait
        when necessary
        Check if getRemainingTime returns the remaining time of the last computed phase
        """

        for i in range(16 * self.k_mpc):
            self.gait.update(i, 0)

        cgait = self.gait.matrix

        for i in range(cgait.shape[0]):
            for j in range(4):
                self.assertTrue(
                    np.allclose(
                        self.params.T_gait * 0.5, self.gait.get_phase_duration(i, j)
                    ),
                    "phaseDuration is OK",
                )
                self.assertTrue(
                    np.allclose(
                        self.params.T_gait * 0.5
                        - self.params.dt_mpc * (12 - (i - 9) % 12),
                        self.gait.get_elapsed_time(i, j),
                    ),
                    "elapsedTime is OK",
                )
                self.assertTrue(
                    np.allclose(
                        self.params.dt_mpc * (12 - (i - 9) % 12),
                        self.gait.get_remaining_time(i, j),
                    ),
                    "remainingTime is OK",
                )

    def test_upper(self):
        self.assertEqual("foo".upper(), "FOO")


if __name__ == "__main__":
    unittest.main()
