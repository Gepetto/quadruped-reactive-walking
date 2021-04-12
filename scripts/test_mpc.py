import unittest
import numpy as np
import time

# Import classes to test
import MPC_Wrapper
np.set_printoptions(precision=3, linewidth=300)


class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)


class TestMPC(unittest.TestCase):

    def setUp(self):

        self.mass = 2.50000279
        type_MPC = True
        dt_wbc = 0.002
        dt_mpc = 0.02
        k_mpc = int(dt_mpc / dt_wbc)
        T_mpc = 0.32
        T_gait = 0.32
        q_init = np.zeros((19, 1))
        q_init[0:7, 0] = np.array([0.0, 0.0, 0.24474949993103629, 0.0, 0.0, 0.0, 1.0])
        q_init[7:, 0] = np.array([0.0, 0.7, -1.4, -0.0, 0.7, -1.4, 0.0, -0.7, +1.4, -0.0, -0.7, +1.4])
        enable_multiprocessing = True
        self.mpc_wrapper = MPC_Wrapper.MPC_Wrapper(type_MPC, dt_mpc, np.int(T_mpc/dt_mpc),
                                                   k_mpc, T_mpc, q_init, enable_multiprocessing)

        self.planner = MPC_Wrapper.Dummy()
        n_steps = np.int(T_gait/dt_mpc)
        self.planner.xref = np.zeros((12, 1 + n_steps))
        self.planner.gait = np.zeros((20, 5))
        self.planner.fsteps = np.zeros((20, 13))

    def tearDown(self):
        self.mpc_wrapper.stop_parallel_loop()  # Stop MPC running in parallel process

    def test_fourstance_immobile(self):

        self.planner.xref[2, :] = 0.24474949993103629
        self.planner.gait[0, :] = np.array([16, 1, 1, 1, 1])
        self.planner.fsteps[0, :] = np.array([16, 0.195, 0.147, 0., 0.195, -0.147, 0., -0.195, 0.147, 0., -0.195, -0.147, 0.])
        """planner.fsteps[0, :] = np.array([7, 0.195, 0.147, 0., 0., 0., 0., 0., 0., 0., -0.195, -0.147, 0.])
        planner.fsteps[1, :] = np.array([8, 0., 0., 0., 0.195, -0.147, 0., -0.195, 0.147, 0., 0., 0., 0.])
        planner.fsteps[2, :] = np.array([1, 0.195, 0.147, 0., 0., 0., 0., 0., 0., 0., -0.195, -0.147, 0.])"""

        # First iteration returns [0.0, 0.0, 8.0] for all feet in contact
        self.mpc_wrapper.solve(0, self.planner)
        while not self.mpc_wrapper.newResult.value:
            time.sleep(0.002)
        x_f_mpc = self.mpc_wrapper.get_latest_result()

        # Second iteration should return [0.0, 0.0, mass/4] for all feet
        for i in range(1, 100):
            self.mpc_wrapper.solve(i, self.planner)
            while not self.mpc_wrapper.newResult.value:
                time.sleep(0.002)
            x_f_mpc = self.mpc_wrapper.get_latest_result()  # Retrieve last result of MPC
            # print("X: ", x_f_mpc[:12, 0:6])
            # print("F: ", x_f_mpc[12:, 0:6])

            self.planner.xref[:, 0] = x_f_mpc[:12, 0].copy()  # Update current state of the robot

        # print(9.81*self.mass/4)
        # print(x_f_mpc[12:, 0])
        # print(x_f_mpc[:12, 0] - self.planner.xref[:, 1])
        self.assertTrue(np.allclose(x_f_mpc[12:, 0], np.array([x_f_mpc[12, 0], x_f_mpc[13, 0], x_f_mpc[14, 0]] * 4)), "All feet forces are equal.")
        self.assertTrue(np.allclose(x_f_mpc[:12, 0], self.planner.xref[:, 1], atol=1e-3), "Close to reference state")
        # self.assertTrue(np.allclose(x_f_mpc[12:, 0], np.array([0.0, 0.0, 9.81*self.mass/4] * 4)), "Feet forces are equal to theorical value.")

    def test_fourstance_not_centered(self):
        self.planner.xref[2, :] = 0.24474949993103629
        self.planner.gait[0, :] = np.array([16, 1, 1, 1, 1])
        self.planner.fsteps[0, :] = np.array([16, 0.195, 0.147, 0., 0.195, -0.147, 0., -0.195, 0.147, 0., -0.195, -0.147, 0.])

        # Non centered state
        self.planner.xref[:, 0] = np.array([0.05, 0.05, 0.2, 0.1, 0.1, 0.1, 0.01, 0.01, 0.04, 0.4, 0.4, 0.4])

        # First iteration
        self.mpc_wrapper.solve(0, self.planner)
        while not self.mpc_wrapper.newResult.value:
            time.sleep(0.002)
        x_f_mpc = self.mpc_wrapper.get_latest_result()

        # Run the mpc during 500 iterations
        for i in range(1,500):
            self.mpc_wrapper.solve(i, self.planner)
            while not self.mpc_wrapper.newResult.value:
                time.sleep(0.002)
            x_f_mpc = self.mpc_wrapper.get_latest_result()  # Retrieve last result of MPC
            self.planner.xref[:, 0] = x_f_mpc[:12, 0].copy()  # Update current state of the robot

        self.assertTrue(np.allclose(x_f_mpc[12:, 0], np.array([x_f_mpc[12, 0], x_f_mpc[13, 0], x_f_mpc[14, 0]] * 4)), "All feet forces are equal.")
        self.assertTrue(np.allclose(x_f_mpc[:12, 0], self.planner.xref[:, 1], atol=1e-3), "Close to reference state")

    def roll(self, gait, fsteps):
        if(gait[0, 0]==1):
            gait[2, 0] += 1
            fsteps[2, 0] += 1
            gait[0, :] = gait[1, :].copy()
            gait[1, :] = gait[2, :].copy()
            fsteps[0, :] = fsteps[1, :].copy()
            fsteps[1, :] = fsteps[2, :].copy()
            gait[2, :] *= 0.0
            fsteps[2, :] *= 0.0
        elif(gait[0, 0]==8):
            gait[0, 0] -= 1
            fsteps[0, 0] -= 1
            gait[2, 0] = 1
            fsteps[2, 0] = 1
            gait[2, 1:] = gait[0, 1:].copy()
            fsteps[2, 1:] = fsteps[0, 1:].copy()
        else:
            gait[0, 0] -= 1
            fsteps[0, 0] -= 1
            gait[2, 0] += 1
            fsteps[2, 0] += 1

    # Two feet in stance phase at a given time (FL-HR and FR-HL) with a 0.32s gait period. Check if robot stays centered.
    def test_twostance_centered(self):
        pair_1 = np.array([0.195, 0.147, 0., 0., 0., 0., 0., 0., 0., -0.195, -0.147, 0.])
        pair_2 = np.array([0., 0., 0., 0.195, -0.147, 0., -0.195, 0.147, 0., 0., 0., 0.])

        self.planner.xref[2, :] = 0.24474949993103629
        self.planner.gait[0, :] = np.array([7, 1, 0, 0, 1])
        self.planner.gait[1, :] = np.array([8, 0, 1, 1, 0])
        self.planner.gait[2, :] = np.array([1, 1, 0, 0, 1])
        self.planner.fsteps[0:3, 0] = np.array([7, 8, 1])
        self.planner.fsteps[0, 1:] = pair_1
        self.planner.fsteps[1, 1:] = pair_2
        self.planner.fsteps[2, 1:] = pair_1

        # Run the mpc during 500 iterations
        for i in range(0, 500):
            self.mpc_wrapper.solve(i, self.planner)
            while not self.mpc_wrapper.newResult.value:
                time.sleep(0.002)
            x_f_mpc = self.mpc_wrapper.get_latest_result()  # Retrieve last result of MPC
            self.roll(self.planner.gait, self.planner.fsteps)
            if(i > 0):
                self.planner.xref[:, 0] = x_f_mpc[:12, 0].copy()  # Update current state of the robot

        print(x_f_mpc[:12, 0])
        self.assertTrue(np.allclose(x_f_mpc[:12, 0], self.planner.xref[:, 1], atol=1e-2), "Close to reference state")

    # Two feet in stance phase at a given time (FL-HR and FR-HL) with a 0.32s gait period. Check if robot stays centered.
    def test_twostance_not_centered(self):
        pair_1 = np.array([0.195, 0.147, 0., 0., 0., 0., 0., 0., 0., -0.195, -0.147, 0.])
        pair_2 = np.array([0., 0., 0., 0.195, -0.147, 0., -0.195, 0.147, 0., 0., 0., 0.])

        self.planner.xref[2, :] = 0.24474949993103629
        self.planner.gait[0, :] = np.array([7, 1, 0, 0, 1])
        self.planner.gait[1, :] = np.array([8, 0, 1, 1, 0])
        self.planner.gait[2, :] = np.array([1, 1, 0, 0, 1])
        self.planner.fsteps[0:3, 0] = np.array([7, 8, 1])
        self.planner.fsteps[0, 1:] = pair_1
        self.planner.fsteps[1, 1:] = pair_2
        self.planner.fsteps[2, 1:] = pair_1

        # Non centered state
        self.planner.xref[:, 0] = np.array([0.05, 0.05, 0.2, 0.1, 0.1, 0.1, 0.01, 0.01, 0.04, 0.4, 0.4, 0.4])

        # Run the mpc during 500 iterations
        for i in range(0, 2000):
            self.mpc_wrapper.solve(i, self.planner)
            while not self.mpc_wrapper.newResult.value:
                time.sleep(0.002)
            x_f_mpc = self.mpc_wrapper.get_latest_result()  # Retrieve last result of MPC
            self.roll(self.planner.gait, self.planner.fsteps)
            if(i > 0):
                self.planner.xref[:, 0] = x_f_mpc[:12, 0].copy()  # Update current state of the robot

        print(x_f_mpc[:12, 0])
        self.assertTrue(np.allclose(x_f_mpc[:12, 0], self.planner.xref[:, 1], atol=1e-2), "Close to reference state")


    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')



if __name__ == '__main__':
    unittest.main()
