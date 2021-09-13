import numpy as np
import utils_mpc

class Test:

    def __init__(self, params):

        self.k = 0
        self.k_mpc = 20
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

        self.q = np.zeros((18, 1))  # Orientation part is in roll pitch yaw
        self.q[0:6, 0] = np.array([0.0, 0.0, self.h_ref, 0.0, 0.0, 0.0])
        self.q[6:, 0] = q_init
        self.h_v = np.zeros((18, 1))
        self.h_v_ref = np.zeros((18, 1))

        self.q_wbc = np.zeros((18, 1))
        self.dq_wbc = np.zeros((18, 1))
        self.xgoals = np.zeros((12, 1))
        self.xgoals[2, 0] = self.h_ref

    def run(self):

        # Update gait
        self.gait.updateGait(self.k, self.k_mpc, 0)
        cgait = self.gait.getCurrentGait()

        # Compute target footstep based on current and reference velocities
        o_targetFootstep = self.footstepPlanner.updateFootsteps(self.k % self.k_mpc == 0 and self.k != 0,
                                                                int(self.k_mpc - self.k % self.k_mpc),
                                                                self.q[:, 0],
                                                                self.h_v[0:6, 0:1].copy(),
                                                                self.h_v_ref[0:6, 0:1])

        # Update pos, vel and acc references for feet
        self.footTrajectoryGenerator.update(self.k, o_targetFootstep)

        # Update configuration vector for wbc
        self.q_wbc[:, 0] = self.q[:, 0].copy()
        self.q_wbc[2, 0] = self.h_ref  # Height

        # Update velocity vector for wbc
        self.dq_wbc[:6, 0] = self.h_v[:6, 0]  #  Velocities in base frame (not horizontal frame!)
        self.dq_wbc[6:, 0] = self.wbcWrapper.vdes[:]  # with reference angular velocities of previous loop

        # Feet command position, velocity and acceleration in base frame
        self.feet_a_cmd = self.footTrajectoryGenerator.getFootAcceleration()
        self.feet_v_cmd = self.footTrajectoryGenerator.getFootVelocity()
        self.feet_p_cmd = self.footTrajectoryGenerator.getFootPosition()

        # Desired position, orientation and velocities of the base
        if not self.gait.getIsStatic():
            self.xgoals[[0, 1, 5], 0] = self.q[[0, 1, 5], 0]
            self.xgoals[2:5, 0] = [self.h_ref, 0.0, 0.0]  #  Height (in horizontal frame!)
        else:
            self.xgoals[2:5, 0] += self.h_v_ref[2:5, 0] * self.dt_wbc

        self.xgoals[6:, 0] = self.h_v_ref[:6, 0]  # Velocities (in horizontal frame!)

        cgait = self.gait.getCurrentGait()

        """if self.k == 50:
            from IPython import embed
            embed()"""

        # Run InvKin + WBC QP
        self.wbcWrapper.compute(self.q_wbc, self.dq_wbc,
                                np.zeros((12, 1)), np.array([cgait[0, :]]),
                                self.footTrajectoryGenerator.getFootPosition(),
                                self.footTrajectoryGenerator.getFootVelocity(),
                                self.footTrajectoryGenerator.getFootAcceleration(),
                                self.xgoals)

        # Quantities sent to the control board
        #self.result.P = np.array(self.Kp_main.tolist() * 4)
        #self.result.D = np.array(self.Kd_main.tolist() * 4)
        self.b_des = self.wbcWrapper.bdes[:]
        self.q_des = self.wbcWrapper.qdes[:]
        self.v_des = self.wbcWrapper.vdes[:]
        #self.result.FF = self.Kff_main * np.ones(12)
        self.tau_ff = self.wbcWrapper.tau_ff

        self.k += 1


if __name__ == "__main__":

    import libquadruped_reactive_walking as lqrw
    import pinocchio as pin

    params = lqrw.Params()  # Object that holds all controller parameters
    params.enable_corba_viewer = True
    test = Test(params)
    q_display = np.zeros((19, 1))

    test.h_v[0, 0] = 0.1
    test.h_v_ref[0, 0] = 0.1
    for i in range(160):
        test.run()

        test.q[2, 0] = test.b_des[2]
        test.q[3:6, 0] = pin.rpy.matrixToRpy(pin.Quaternion(test.b_des[3:7].reshape((-1, 1))).toRotationMatrix())
        test.q[6:, 0] = test.q_des[:]

        # Display robot in Gepetto corba viewer
        if (i % 5 == 0):
            q_display[:3, 0] = test.b_des[0:3]
            q_display[3:7, 0] = test.b_des[3:7]
            q_display[7:, 0] = test.q_des[:]
            test.solo.display(q_display)

        test.q[0, 0] += params.dt_wbc * test.h_v[0, 0]
        test.q[1, 0] = 0.0
        test.q[5, 0] = 0.0

        

    from IPython import embed
    embed()

