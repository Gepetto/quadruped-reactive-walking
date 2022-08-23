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


class TestFootstepPlanner(unittest.TestCase):

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
        q_init = [0.0, 0.764, -1.4, 0.0, 0.764, -1.4, 0.0, 0.764, -1.4, 0.0, 0.764, -1.4]
        for i in range(len(q_init)):
            self.params.q_init[i] = q_init[i]
        self.params.N_periods = 1
        gait = [12, 1, 0, 0, 1,
                12, 0, 1, 1, 0]
        for i in range(len(gait)):
            self.params.gait_vec[i] = gait[i]

        # Force refresh of gait matrix
        self.params.convert_gait_vec()

        # Initialization of params
        self.params.initialize()

        # Create Estimator class and initialize it
        self.estimator = lqrw.Estimator()
        self.estimator.initialize(self.params)

        # Create Gait class and initialize it
        self.gait = lqrw.Gait()
        self.gait.initialize(self.params)

        # Create FootstepPlanner class and initialize it
        self.footstepPlanner = lqrw.FootstepPlanner()
        self.footstepPlanner.initialize(self.params, self.gait)

        """
        # Load robot model and data
        # Initialisation of the Gepetto viewer
        Solo12Loader.free_flyer = True
        self.solo = Solo12Loader().robot
        q = self.solo.q0.reshape((-1, 1))

        # Initialisation of the position of footsteps to be under the shoulder
        # There is a lateral offset of around 7 centimeters
        pos_feet = np.zeros((3, 4))
        indexes = [self.solo.model.getFrameId('FL_FOOT'),
                   self.solo.model.getFrameId('FR_FOOT'),
                   self.solo.model.getFrameId('HL_FOOT'),
                   self.solo.model.getFrameId('HR_FOOT')]
        q[7:, 0] = np.array(q_init)
        pin.framesForwardKinematics(self.solo.model, self.solo.data, q)
        pin.updateFramePlacements(self.solo.model, self.solo.data)
        for i in range(4):
            pos_feet[:, i] = self.solo.data.oMf[indexes[i]].translation
        pos_feet[2, :] = 0.0  # Z component does not matter
        """

    def tearDown(self):
        pass

    def test_non_moving(self):
        """
        Check footsteps in a basic non-moving situation
        """

        # Footsteps should be at the vertical of shoulders in a non-moving situation
        ref = np.array(self.params.footsteps_under_shoulders.tolist()).reshape((3, 4), order='F')
        N_ref = self.params.gait.shape[0]

        # Configuration vector
        q = np.zeros((18, 1))
        q[2, 0] = self.params.h_ref
        q[6:, 0] = np.array(self.params.q_init)

        for k in range(500):
            # Update gait
            self.gait.update(k, 0)

            # Compute target footstep based on current and reference velocities
            o_targetFootstep = self.footstepPlanner.update_footsteps(k % self.k_mpc == 0 and k != 0,
                                                                    int(self.k_mpc - k % self.k_mpc),
                                                                    q, np.zeros((6, 1)),
                                                                    np.zeros((6, 1)),
                                                                    ref)
            # Same footsteps in horizontal frame
            h_targetFootstep = self.footstepPlanner.get_target_footsteps()

            # Pos of feet in stance phase + target of feet in swing phase (in horizontal frame)
            fsteps = self.footstepPlanner.get_footsteps()

            # Check footsteps locations
            if not np.allclose(ref, o_targetFootstep):
                print(ref)
                print(o_targetFootstep)
            self.assertTrue(np.allclose(ref, o_targetFootstep), "o_targetFootstep is OK")
            self.assertTrue(np.allclose(ref, h_targetFootstep), "h_targetFootstep is OK")
            self.assertTrue(np.allclose(np.tile(ref.ravel(order='F'), (N_ref, 1)) *
                                        np.repeat(self.gait.matrix, 3, axis=1), fsteps), "fsteps is OK")

    def test_moving_at_ref_forward(self):
        """
        Check footsteps when walking at reference velocity forwards
        """

        # Forward velocity
        v_x = 0.5

        # Footsteps should land in front of the shoulders
        under_shoulder = np.array(self.params.footsteps_under_shoulders.tolist()).reshape((3, 4), order='F')
        o_targetFootstep = under_shoulder.copy()
        targets = under_shoulder.copy()
        targets[0, :] += v_x * self.params.T_gait / 4
        N_ref = self.params.gait.shape[0]

        # Configuration vector
        q = np.zeros((18, 1))
        q[2, 0] = self.params.h_ref
        q[6:, 0] = np.array(self.params.q_init)

        # Velocity vectors
        v = np.zeros((6, 1))
        v[0, 0] = v_x
        v_ref = np.zeros((6, 1))
        v_ref[0, 0] = v_x

        for k in range(500):

            # Run estimator
            self.estimator.run(self.gait.matrix,
                                      np.random.random((3, 4)),
                                      np.random.random((3, 1)),
                                      np.random.random((3, 1)),
                                      np.random.random((3, 1)),
                                      np.random.random((12, 1)),
                                      np.random.random((12, 1)),
                                      np.random.random((6, 1)),
                                      np.random.random((3, 1)))

            # Update state
            self.estimator.update_reference_state(v_ref)

            # Robot moving in ideal world
            oTh = self.estimator.get_oTh()
            yaw = self.estimator.get_q_reference()[5]
            q[0:2, 0] = oTh[0:2]
            q[5, 0] = yaw

            # Update gait
            self.gait.update(k, 0)

            # Compute target footstep based on current and reference velocities
            o_targetFootstep = self.footstepPlanner.update_footsteps(k % self.k_mpc == 0 and k != 0,
                                                                    int(self.k_mpc - k % self.k_mpc),
                                                                    q, v, v_ref, o_targetFootstep)
            
            print(o_targetFootstep)

            # Same footsteps in horizontal frame
            h_targetFootstep = self.footstepPlanner.get_target_footsteps()

            # Pos of feet in stance phase + target of feet in swing phase (in horizontal frame)
            fsteps = self.footstepPlanner.get_footsteps()

            """print(k)
            print(fsteps[0:2, :])"""

            # Check footsteps locations
            # if (k % 20 == 0):
            #     print(o_targetFootstep)

            # Check each foot one after the other
            cgait = self.gait.matrix
            for j in range(4):
                phase = -1
                cpt_phase = 0
                for i in range(cgait.shape[0]):
                    if (cgait[i, j] == 0):
                        if (phase != 0):
                            if (phase != -1):
                                cpt_phase += 1
                            phase = 0
                        self.assertTrue(np.allclose(np.zeros(3), fsteps[i, (3*j):(3*(j+1))]), "fsteps swing is OK")
                    else:
                        if (phase != 1):
                            if (phase != -1):
                                cpt_phase += 1
                            phase = 1
                            if (cpt_phase == 0):  # Foot currently in stance phase
                                o_loc = under_shoulder[:, j] + \
                                    np.array([v_x * np.floor(k / 240) * 240 * self.params.dt_wbc, 0.0, 0.0])
                                if k >= 240:
                                    o_loc += np.array([v_x * self.params.T_gait * 0.25, 0.0, 0.0])

                            else:
                                o_loc = targets[:, j] + np.array([v_x * (np.floor(k / 240) *
                                                                         240 + 1 + cpt_phase * 240) * self.params.dt_wbc, 0.0, 0.0])
                            h_loc = o_loc - np.array([v_x * (k + 1) * self.params.dt_wbc, 0.0, 0.0])
                            #print("oloc:", o_loc)
                            #print("minu:", np.array([v_x * (k + 1) * self.params.dt_wbc, 0.0, 0.0]))
                        if (not np.allclose(h_loc, fsteps[i, (3*j):(3*(j+1))])):
                            print("---")
                            print("Status: ", cgait[0, :])
                            print("[", i, ", ", j, "]")
                            print(h_loc)
                            print(fsteps[i, (3*j):(3*(j+1))])
                            print(o_loc)
                            print(o_targetFootstep)
                            print("---")

                            from IPython import embed
                            embed()
                        self.assertTrue(np.allclose(h_loc, fsteps[i, (3*j):(3*(j+1))]), "fsteps stance is OK")
                        if (cpt_phase == 0):
                            self.assertTrue(np.allclose(h_loc, h_targetFootstep[:, j]), "h_target is OK")
                            self.assertTrue(np.allclose(o_loc, o_targetFootstep[:, j]), "o_target is OK")
    
    """if(not flag_first_swing):
        loc_stance[:] = under_shoulder[:, j]

    if (val)
        self.assertTrue(np.allclose(under_shoulder[:, j], h_targetFootstep[:, j]), "o_targetFootstep is OK")
        under_shoulder
    else:"""

    # phTime = self.gait.getPhaseDuration(0, j, 1)
    # remTime = self.gait.getRemainingTime()
    # self.assertTrue(0.24 == self.gait.getPhaseDuration(i, j, cgait[i, j]), "phaseDuration is OK")
    # self.assertTrue(12 - (i - 9) % 12 == self.gait.getRemainingTime(), "remainingTime is OK")

    """if not np.allclose(ref, o_targetFootstep):
        print(ref)
        print(o_targetFootstep)
    self.assertTrue(np.allclose(ref, o_targetFootstep), "o_targetFootstep is OK")
    self.assertTrue(np.allclose(ref, h_targetFootstep), "h_targetFootstep is OK")
    self.assertTrue(np.allclose(np.tile(ref.ravel(order='F'), (N_ref, 1)) *
                                np.repeat(self.gait.matrix, 3, axis=1), fsteps), "fsteps is OK")"""

    def test_moving_at_ref_turning(self):
        """
        Check footsteps when walking at reference velocity forwards and turning
        """

        # Forward velocity
        v_x = 0.5
        v_y = 0.4
        w_yaw = 1.3

        def get_oTh(k):
            if (w_yaw != 0):
                x = (v_x * np.sin(w_yaw * k * self.params.dt_wbc) + v_y *
                     (np.cos(w_yaw * k * self.params.dt_wbc) - 1.0)) / w_yaw
                y = (v_y * np.sin(w_yaw * k * self.params.dt_wbc) - v_x *
                     (np.cos(w_yaw * k * self.params.dt_wbc) - 1.0)) / w_yaw
            else:
                x = v_x * self.params.dt_wbc * k
                y = v_y * self.params.dt_wbc * k
            return np.array([[x], [y], [0.0]])

        def get_oTh_bis(k):
            k = int(k)
            k_range = np.linspace(0, k, k + 1)
            yaw = w_yaw * (k_range + 1) * self.params.dt_wbc
            oTh = np.zeros((3, 1))
            for i in range(k):
                Rz = pin.rpy.rpyToMatrix(0.0, 0.0, yaw[i])
                vRef = Rz @ np.array([[v_x], [v_y], [0.0]])
                oTh += vRef * self.params.dt_wbc
            return oTh

        def get_oRh(k):
            return pin.rpy.rpyToMatrix(0.0, 0.0, w_yaw * k * self.params.dt_wbc)

        # Footsteps should land in front of the shoulders
        under_shoulder = np.array(self.params.footsteps_under_shoulders.tolist()).reshape((3, 4), order='F')
        o_targetFootstep = under_shoulder.copy()
        targets = under_shoulder.copy()
        targets[0, :] += v_x * self.params.T_gait / 4 + 0.5 * np.sqrt(self.params.h_ref / 9.81) * (v_y * w_yaw)
        targets[1, :] += v_y * self.params.T_gait / 4 + 0.5 * np.sqrt(self.params.h_ref / 9.81) * (- v_x * w_yaw)
        N_ref = self.params.gait.shape[0]

        # Configuration vector
        q = np.zeros((18, 1))
        q[2, 0] = self.params.h_ref
        q[6:, 0] = np.array(self.params.q_init)

        # Velocity vectors
        v = np.zeros((6, 1))
        v[[0, 1, 5], 0] = [v_x, v_y, w_yaw]
        v_ref = np.zeros((6, 1))
        v_ref[[0, 1, 5], 0] = [v_x, v_y, w_yaw]

        # Check consistency over iterations
        memory_o_targetFootstep = np.zeros((3, 4))

        log_o_targetFootstep = np.zeros((500, 3, 4))
        log_h_targetFootstep = np.zeros((500, 3, 4))

        for k in range(500):

            # Run estimator
            self.estimator.run(self.gait.matrix,
                                      np.random.random((3, 4)),
                                      np.random.random((3, 1)),
                                      np.random.random((3, 1)),
                                      np.random.random((3, 1)),
                                      np.random.random((12, 1)),
                                      np.random.random((12, 1)),
                                      np.random.random((6, 1)),
                                      np.random.random((3, 1)))

            # Update state
            self.estimator.update_reference_state(v_ref)

            # Robot moving in ideal world
            oTh = get_oTh_bis(k + 1)
            oRh = get_oRh(k + 1)
            yaw = w_yaw * (k + 1) * self.params.dt_wbc

            # Compare with estimator
            """print(oTh.ravel())
            print(get_oTh_bis(k + 1).ravel())
            print(self.estimator.get_oTh())"""
            self.assertTrue(np.allclose(oTh.ravel(), self.estimator.get_oTh(), atol=1e-3), "oTh is OK")
            self.assertTrue(np.allclose(yaw, self.estimator.get_q_reference()[5]), "yaw is OK")

            """if k == 100:
                from IPython import embed
                embed()"""

            # oTh = self.estimator.get_oTh().reshape((3, 1))
            # yaw = self.estimator.get_q_reference()[5]
            # oRh = pin.rpy.rpyToMatrix(0.0, 0.0, yaw)
            q[0:2, 0] = self.estimator.get_oTh()[0:2]  # oTh[0:2, 0]
            q[5, 0] = self.estimator.get_q_reference()[5]

            # Update gait
            self.gait.update(k, 0)

            # Compute target footstep based on current and reference velocities
            o_targetFootstep = self.footstepPlanner.update_footsteps(k % self.k_mpc == 0 and k != 0,
                                                                    int(self.k_mpc - k % self.k_mpc),
                                                                    q, v, v_ref, o_targetFootstep)
            log_o_targetFootstep[k, :, :] = o_targetFootstep.copy()
            #print(k)
            #print(o_targetFootstep)

            # Same footsteps in horizontal frame
            h_targetFootstep = self.footstepPlanner.get_target_footsteps()
            log_h_targetFootstep[k, :, :] = h_targetFootstep.copy()

            np.set_printoptions(precision=8)
            """pos = np.array([[0], [-1]])
            dpos = np.array([[v_x * 0.001], [v_y * 0.001]])
            c = np.cos(w_yaw * 0.001)
            s = np.sin(w_yaw * 0.001)
            R = np.array([[c, s], [-s, c]])
            R @ (pos - dpos)
            print("DPOS: ", dpos)
            print("R ", R)
            from IPython import embed
            embed()"""

            mem = h_targetFootstep.copy()

            # Pos of feet in stance phase + target of feet in swing phase (in horizontal frame)
            fsteps = self.footstepPlanner.get_footsteps()

            # Check each foot one after the other
            cgait = self.gait.matrix
            for j in range(4):
                phase = -1
                cpt_phase = 0
                for i in range(cgait.shape[0]):
                    if (cgait[i, j] == 0):
                        if (phase != 0):
                            if (phase != -1):
                                cpt_phase += 1
                            phase = 0
                        self.assertTrue(np.allclose(np.zeros(3), fsteps[i, (3*j):(3*(j+1))]), "fsteps swing is OK")
                    else:
                        if (phase != 1):
                            if (phase != -1):
                                cpt_phase += 1
                            phase = 1
                            if (cpt_phase == 0):  # Foot currently in stance phase
                                n = np.floor(k / 240) * 240 + 1
                                o_loc = get_oRh(n) @ under_shoulder[:, j:(j+1)] + get_oTh_bis(n)
                                # print("FIRST")
                                if k >= 240:
                                    a = k % 240
                                    """if (k % 240 != 0 and a == 0):
                                        a = 20"""
                                    o_loc = get_oRh(n+a) @ (fsteps[0:1, (3*j):(3*(j+1))]).transpose() + get_oTh_bis(n+a)
                            else:
                                n = np.floor(k / 240) * 240 + 1 + cpt_phase * 240
                                o_loc = get_oRh(n) @ targets[:, j:(j+1)] + get_oTh_bis(n)
                            h_loc = get_oRh(k+1).transpose() @ (o_loc - get_oTh_bis(k+1))
                        # or not np.allclose(o_loc, o_targetFootstep[:, j:(j+1)], atol=1e-6)):

                        """if j == 1 and i == 0 and cgait[i, j] == 1.0:
                            print(o_loc.ravel(), "  |  ", o_targetFootstep[:, j:(j+1)].ravel())"""

                        if (not np.allclose(h_loc.ravel(), fsteps[i, (3*j):(3*(j+1))], atol=1e-3) 
                            or (cpt_phase <= 1 and not np.allclose(o_loc, o_targetFootstep[:, j:(j+1)], atol=1e-3))):
                            print("------ [", i, ", ", j, "]")
                            """print(h_loc)
                            print(fsteps[i, (3*j):(3*(j+1))])"""
                            print(o_loc)
                            print(o_targetFootstep[:, j:(j+1)])
                            from IPython import embed
                            embed()
                        self.assertTrue(np.allclose(
                            h_loc.ravel(), fsteps[i, (3*j):(3*(j+1))], atol=1e-3), "fsteps stance is OK")
                        if (cpt_phase <= 1):
                            self.assertTrue(np.allclose(
                                h_loc, h_targetFootstep[:, j:(j+1)], atol=1e-3), "h_target is OK")
                            self.assertTrue(np.allclose(
                                o_loc, o_targetFootstep[:, j:(j+1)], atol=1e-3), "o_target is OK")

                if (k == 0 or (cgait[0, j] != cgait[-1, j])):  # == 0 and cgait[-1, j] == 1)):
                    memory_o_targetFootstep[:, j] = o_targetFootstep[:, j]
                else:  # if cgait[0, j] == 0:
                    # print("Foot ", j, "in status ", cgait[0, j])
                    #print("Memory: ", memory_o_targetFootstep[:, j])
                    #print("Current: ", o_targetFootstep[:, j])
                    self.assertTrue(np.allclose(
                        memory_o_targetFootstep[:, j], o_targetFootstep[:, j], atol=1e-3), "o_target is consistent")

        """from matplotlib import pyplot as plt
        for j in range(3):
            plt.figure()
            for i in range(4):
                plt.plot(log_o_targetFootstep[:, j, i])
        plt.show(block=True)"""


    """
    # Compute target footstep based on current and reference velocities
    o_targetFootstep = self.footstepPlanner.update_footsteps(self.k % self.k_mpc == 0 and self.k != 0,
                                                            int(self.k_mpc - self.k % self.k_mpc),
                                                            self.q[:, 0], self.h_v_windowed[0:6, 0:1].copy(),
                                                            self.v_ref[0:6, 0:1])

    # Footsteps in horizontal frame
    fsteps = self.footstepPlanner.get_footsteps()
    get_target_footsteps()
    """

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')


if __name__ == '__main__':
    unittest.main()
