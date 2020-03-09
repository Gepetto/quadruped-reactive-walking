# coding: utf8

import numpy as np
from time import clock
import scipy as scipy
import osqp as osqp
from matplotlib import pyplot as plt
import utils


class MPC:
    """Wrapper for the MPC to create constraint matrices, call the QP solver and
    retrieve the result.

    """

    def __init__(self, dt, sequencer):

        # Time step of the MPC solver
        self.dt = dt

        # Mass of the robot
        self.mass = 2.97784899

        # Inertia matrix of the robot in body frame (found in urdf)
        self.gI = np.diag([0.00578574, 0.01938108, 0.02476124])

        # Friction coefficient
        self.mu = 1

        # Number of time steps in the prediction horizon
        self.n_steps = sequencer.S.shape[0]

        # Reference trajectory matrix of size 12 by (1 + N)  with the current state of
        # the robot in column 0 and the N steps of the prediction horizon in the others
        self.xref = np.zeros((12, 1 + self.n_steps))

        # Result of the QP solver
        self.x = np.zeros((12 * self.n_steps * 2,))

        # Initial state vector of the robot (x, y, z, roll, pitch, yaw)
        self.q = np.array([[0.0, 0.0, 0.2027682, 0.0, 0.0, 0.0]]).transpose()

        # State vector of the trunk in the world frame
        self.q_w = self.q.copy()

        # Initial velocity vector of the robot in local frame
        self.v = np.zeros((6, 1))

        # Reference velocity vector of the robot in local frame
        self.v_ref = np.zeros((6, 1))

        # Reference height that the robot will try to maintain
        self.h_ref = self.q[2, 0]

        # Get number of feet in contact with the ground for each step of the gait sequence
        self.n_contacts = np.sum(sequencer.S, axis=1).astype(int)

        # Initial position of footholds in the "straight standing" default configuration
        self.footholds = np.array(
            [[0.19, 0.19, -0.19, -0.19],
             [0.15005, -0.15005, 0.15005, -0.15005],
             [0.0, 0.0, 0.0, 0.0]])

        R = np.array([[0.0, -1.0], [1.0, 0.0]])
        self.footholds[0:2, :] = R @ self.footholds[0:2, :]

    def update_v_ref(self, joystick):

        # Retrieving the reference velocity from the joystick
        self.v_ref = joystick.v_ref

        # Get the reference velocity in global frame
        c, s = np.cos(self.q_w[5, 0]), np.sin(self.q_w[5, 0])
        R = np.array([[c, -s, 0., 0., 0., 0.], [s, c, 0., 0., 0., 0], [0., 0., 1.0, 0., 0., 0.],
                      [0., 0., 0., c, -s, 0.], [0., 0., 0., s, c, 0.], [0., 0., 0., 0., 0., 1.0]])
        self.v_ref_world = np.dot(R, self.v_ref)

        return 0

    def getRefStates(self, k, sequencer, mpc_interface):
        """Returns the reference trajectory of the robot for each time step of the
        predition horizon. The ouput is a matrix of size 12 by N with N the number
        of time steps (around T_gait / dt) and 12 the position / orientation /
        linear velocity / angular velocity vertically stacked.

        Keyword arguments:
        qu -- current position/orientation of the robot (6 by 1)
        v_ref -- reference velocity vector of the flying base (6 by 1, linear and angular stacked)
        dt -- time step
        T_gait -- period of the current gait
        """

        # Update x and y velocities taking into account the rotation of the base over the prediction horizon
        yaw = np.linspace(0, sequencer.T_gait-self.dt, self.n_steps) * self.v_ref[5, 0]
        self.xref[6, 1:] = self.v_ref[0, 0] * np.cos(yaw) - self.v_ref[1, 0] * np.sin(yaw)
        self.xref[7, 1:] = self.v_ref[0, 0] * np.sin(yaw) + self.v_ref[1, 0] * np.cos(yaw)

        # Update x and y depending on x and y velocities (cumulative sum)
        self.xref[0, 1:] = self.dt * np.cumsum(self.xref[6, 1:])
        self.xref[1, 1:] = self.dt * np.cumsum(self.xref[7, 1:])

        # Start from position of the CoM in local frame
        self.xref[0, 1:] += mpc_interface.lC[0, 0]
        self.xref[1, 1:] += mpc_interface.lC[1, 0]

        # Desired height is supposed constant so we only need to set it once
        if k == 0:
            self.xref[2, 1:] = self.h_ref

        # No need to update Z velocity as the reference is always 0
        # No need to update roll and roll velocity as the reference is always 0 for those
        # No need to update pitch and pitch velocity as the reference is always 0 for those
        # Update yaw and yaw velocity
        dt_vector = np.linspace(self.dt, sequencer.T_gait, self.n_steps)
        self.xref[5, 1:] = self.v_ref[5, 0] * dt_vector
        self.xref[11, 1:] = self.v_ref[5, 0]

        # Update the current state
        self.xref[:6, 0:1] = self.q
        self.xref[6:, 0:1] = self.v

        # Current state vector of the robot
        self.x0 = self.xref[:, 0:1]

        return 0

    def retrieve_data(self, fstep_planner, ftraj_gen, mpc_interface):
        """Retrieve footsteps information from the FootstepPlanner
        and the FootTrajectoryGenerator

        Keyword arguments:
        fstep_planner -- FootstepPlanner object
        ftraj_gen -- FootTrajectoryGenerator object
        """

        self.footholds[0:2, :] = ftraj_gen.footsteps_lock.copy()
        self.footholds[0:2, :] = np.array(
            [[0.19, 0.19, -0.19, -0.19], [0.15005, -0.15005, 0.15005, -0.15005]])

        self.footholds[0:2, :] = mpc_interface.l_feet[0:2, :]

        # Information in world frame for visualisation purpose
        self.footholds_world = ftraj_gen.footsteps_lock_world.copy()

        return 0

    def create_matrices(self, sequencer):
        """
        Create the constraint matrices of the MPC (M.X = N and L.X <= K)
        Create the weight matrices P and Q of the MPC solver (cost 1/2 x^T * P * X + X^T * Q)
        """

        # Create the constraint matrices
        """self.create_M(sequencer)
        self.create_N()
        self.create_L()
        self.create_K()"""

        self.create_ML(sequencer)
        self.create_NK()
        """print("ML equal M stacked L: ", np.array_equal(np.vstack((self.M, self.L)), self.ML.toarray()))
        print("NK equal N stacked K: ", np.array_equal(np.vstack((self.N, np.array([self.K]).transpose())), self.NK))"""

        # Create the weight matrices
        self.create_weight_matrices()

        return 0

    def create_M(self, sequencer):
        """ Create the M matrix involved in the MPC constraint equations M.X = N and L.X <= K """

        # Create matrix M
        self.M = np.zeros((12*self.n_steps*2, 12*self.n_steps*2))

        # Put identity matrices in M
        self.M[np.arange(0, 12*self.n_steps, 1), np.arange(0, 12*self.n_steps, 1)] = - np.ones((12*self.n_steps))

        # Create matrix A
        self.A = np.eye(12)
        self.A[[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]] = np.ones((6,)) * self.dt

        # Put A matrices in M
        for k in range(self.n_steps-1):
            self.M[((k+1)*12):((k+2)*12), (k*12):((k+1)*12)] = self.A

        # Create matrix B
        self.B = np.zeros((12, 12))
        self.B[np.tile([6, 7, 8], 4), np.arange(0, 12, 1)] = (self.dt / self.mass) * np.ones((12,))

        # Put B matrices in M
        for k in range(self.n_steps):
            # Get inverse of the inertia matrix for time step k
            c, s = np.cos(self.xref[5, k]), np.sin(self.xref[5, k])
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
            I_inv = np.linalg.inv(np.dot(R, self.gI))

            # Get skew-symetric matrix for each foothold
            lever_arms = self.footholds - self.xref[0:3, k:(k+1)]
            for i in range(4):
                self.B[-3:, (i*3):((i+1)*3)] = self.dt * np.dot(I_inv, utils.getSkew(lever_arms[:, i]))

            self.M[(k*12):((k+1)*12), (12*(self.n_steps+k)):(12*(self.n_steps+k+1))] = self.B

        # Add lines to enable/disable forces
        # With = sequencer.S.reshape((-1,)) we directly initialize with the contact sequence but we have a dependency on the sequencer
        # With = np.ones((12*self.n_steps, )) we would not have this dependency but he would have to set the active forces later
        self.M[np.arange(12*self.n_steps, 12*self.n_steps*2, 1), np.arange(12*self.n_steps,
                                                                           12*self.n_steps*2, 1)] = 1 - np.repeat(sequencer.S.reshape((-1,)), 3)

        return 0

    def create_N(self):
        """ Create the N matrix involved in the MPC constraint equations M.X = N and L.X <= K """

        # Create N matrix
        self.N = np.zeros((12*self.n_steps, 1))

        # Create g matrix
        self.g = np.zeros((12, 1))
        self.g[8, 0] = -9.81 * self.dt

        # Fill N matrix with g matrices
        for k in range(self.n_steps):
            self.N[(12*k):(12*(k+1)), 0:1] = - self.g

        # Including - A*X0 in the first row of N
        self.N[0:12, 0:1] += np.dot(self.A, - self.x0)

        # Create matrix D (third term of N)
        self.D = np.zeros((12*self.n_steps, 12*self.n_steps))

        # Put identity matrices in D
        self.D[np.arange(0, 12*self.n_steps, 1), np.arange(0, 12*self.n_steps, 1)] = np.ones((12*self.n_steps))

        # Put A matrices in D
        for k in range(self.n_steps-1):
            self.D[((k+1)*12):((k+2)*12), (k*12):((k+1)*12)] = - self.A

        # Add third term to matrix N
        self.N += np.dot(self.D, self.xref[:, 1:].reshape((-1, 1), order='F'))

        # Add lines to enable/disable forces
        self.N = np.vstack((self.N, np.zeros((12*self.n_steps, 1))))

        return 0

    def create_L(self):
        """ Create the L matrix involved in the MPC constraint equations M.X = N and L.X <= K """

        # Create L matrix
        self.L = np.zeros((20*self.n_steps, 12*self.n_steps*2))

        # Create C matrix
        self.C = np.zeros((5, 3))
        self.C[[0, 1, 2, 3] * 2 + [4], [0, 0, 1, 1, 2, 2, 2, 2, 2]
               ] = np.array([1, -1, 1, -1, -self.mu, -self.mu, -self.mu, -self.mu, -1])

        # Create F matrix
        self.F = np.zeros((20, 12))
        for i in range(4):
            self.F[(5*i):(5*(i+1)), (3*i):(3*(i+1))] = self.C

        # Fill L matrix with F matrices
        for k in range(self.n_steps):
            self.L[(20*k):(20*(k+1)), (12*(self.n_steps+k)):(12*(self.n_steps+1+k))] = self.F

        return 0

    def create_K(self):
        """ Create the K matrix involved in the MPC constraint equations M.X = N and L.X <= K """

        # Create K matrix
        self.K = np.zeros((20*self.n_steps, ))

        return 0

    def create_ML(self, sequencer):
        """ Create the M and L matrices involved in the MPC constraint equations M.X = N and L.X <= K """

        # Create matrix ML
        self.ML = np.zeros((12*self.n_steps*2 + 20*self.n_steps, 12*self.n_steps*2))
        self.offset_L = 12*self.n_steps*2

        # Put identity matrices in M
        self.ML[np.arange(0, 12*self.n_steps, 1), np.arange(0, 12*self.n_steps, 1)] = - np.ones((12*self.n_steps))

        # Create matrix A
        self.A = np.eye(12)
        self.A[[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]] = np.ones((6,)) * self.dt

        # Put A matrices in M
        for k in range(self.n_steps-1):
            self.ML[((k+1)*12):((k+2)*12), (k*12):((k+1)*12)] = self.A

        # Create matrix B
        self.B = np.zeros((12, 12))
        self.B[np.tile([6, 7, 8], 4), np.arange(0, 12, 1)] = (self.dt / self.mass) * np.ones((12,))

        # Put B matrices in M
        for k in range(self.n_steps):
            # Force coefficients to non sparse value
            for i in range(4):
                self.B[-3:, (i*3):((i+1)*3)] = 8 * np.ones((3, 3))

            self.ML[(k*12):((k+1)*12), (12*(self.n_steps+k)):(12*(self.n_steps+k+1))] = self.B

        # Add lines to enable/disable forces
        # With = sequencer.S.reshape((-1,)) we directly initialize with the contact sequence but we have a dependency on the sequencer
        # With = np.ones((12*self.n_steps, )) we would not have this dependency but he would have to set the active forces later
        self.ML[np.arange(12*self.n_steps, 12*self.n_steps*2, 1), np.arange(12*self.n_steps,
                                                                            12*self.n_steps*2, 1)] = np.ones((12*self.n_steps,))

        # Create C matrix
        self.C = np.zeros((5, 3))
        self.C[[0, 1, 2, 3] * 2 + [4], [0, 0, 1, 1, 2, 2, 2, 2, 2]
               ] = np.array([1, -1, 1, -1, -self.mu, -self.mu, -self.mu, -self.mu, -1])

        # Create F matrix
        self.F = np.zeros((20, 12))
        for i in range(4):
            self.F[(5*i):(5*(i+1)), (3*i):(3*(i+1))] = self.C

        # Fill ML matrix with F matrices
        for k in range(self.n_steps):
            self.ML[(self.offset_L+20*k):(self.offset_L+20*(k+1)),
                    (12*(self.n_steps+k)):(12*(self.n_steps+1+k))] = self.F

        # Transformation into CSC matrix
        self.ML = scipy.sparse.csc.csc_matrix(self.ML, shape=self.ML.shape)

        # Create indices list that will be used to update ML
        self.i_x_B = [6, 9, 10, 11, 7, 9, 10, 11, 8, 9, 10, 11] * 4
        self.i_y_B = np.repeat(np.arange(0, 12, 1), 4)

        i_start = 30*self.n_steps-18
        i_data = np.tile(np.array([0, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 17]), 4)
        i_foot = np.repeat(np.array([0, 1, 2, 3]) * 24, 12)
        self.i_update_B = i_data + i_foot + i_start

        i_S = 4 * np.ones((12*self.n_steps), dtype='int64')
        i_off = np.tile(np.array([3, 3, 6]), 4*self.n_steps)
        i_off = np.roll(np.cumsum(i_S + i_off), 1)
        i_off[0] = 0
        self.i_update_S = i_S + i_off + i_start

        # Update state of B
        for k in range(self.n_steps):
            # Get inverse of the inertia matrix for time step k
            c, s = np.cos(self.xref[5, k]), np.sin(self.xref[5, k])
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
            I_inv = np.linalg.inv(np.dot(R, self.gI))

            # Get skew-symetric matrix for each foothold
            lever_arms = self.footholds - self.xref[0:3, k:(k+1)]
            for i in range(4):
                self.B[-3:, (i*3):((i+1)*3)] = self.dt * np.dot(I_inv, utils.getSkew(lever_arms[:, i]))

            i_iter = 24 * 4 * k
            self.ML.data[self.i_update_B + i_iter] = self.B[self.i_x_B, self.i_y_B]

        # Update state of legs
        self.ML.data[self.i_update_S] = (1 - np.repeat(sequencer.S.reshape((-1,)), 3)).ravel()

        return 0

    def create_NK(self):
        """ Create the N and K matrices involved in the MPC constraint equations M.X = N and L.X <= K """

        # Create N matrix
        self.NK = np.zeros((12*self.n_steps * 2 + 20*self.n_steps, 1))

        # Create g matrix
        self.g = np.zeros((12, 1))
        self.g[8, 0] = -9.81 * self.dt

        # Fill N matrix with g matrices
        for k in range(self.n_steps):
            self.NK[(12*k):(12*(k+1)), 0:1] = - self.g

        # Including - A*X0 in the first row of N
        self.NK[0:12, 0:1] += np.dot(self.A, - self.x0)

        # Create matrix D (third term of N)
        self.D = np.zeros((12*self.n_steps, 12*self.n_steps))

        # Put identity matrices in D
        self.D[np.arange(0, 12*self.n_steps, 1), np.arange(0, 12*self.n_steps, 1)] = np.ones((12*self.n_steps))

        # Put A matrices in D
        for k in range(self.n_steps-1):
            self.D[((k+1)*12):((k+2)*12), (k*12):((k+1)*12)] = - self.A

        # Add third term to matrix N
        self.NK[:12*self.n_steps, :] += np.dot(self.D, self.xref[:, 1:].reshape((-1, 1), order='F'))

        # Lines to enable/disable forces are already initialized (0 values)
        # Matrix K is already initialized (0 values)

        self.NK_inf = np.zeros((12*self.n_steps * 2 + 20*self.n_steps, ))
        self.inf_lower_bound = -np.inf * np.ones((20*self.n_steps,))
        self.inf_lower_bound[4::5] = - 25

        self.NK_inf[:12*self.n_steps * 2] = self.NK[:12*self.n_steps * 2, 0]
        self.NK_inf[12*self.n_steps * 2:] = self.inf_lower_bound

        return 0

    def create_weight_matrices(self):
        """Create the weight matrices in the cost x^T.P.x + x^T.q of the QP problem
        """

        # Number of states
        n_x = 12

        # Declaration of the P matrix in "x^T.P.x + x^T.q"
        # P_row, _col and _data satisfy the relationship P[P_row[k], P_col[k]] = P_data[k]
        P_row = np.array([], dtype=np.int64)
        P_col = np.array([], dtype=np.int64)
        P_data = np.array([], dtype=np.float64)

        # Define weights for the x-x_ref components of the optimization vector
        P_row = np.arange(0, n_x * self.n_steps, 1)
        P_col = np.arange(0, n_x * self.n_steps, 1)
        P_data = 0.0 * np.ones((n_x * self.n_steps,))

        # Hand-tuning of parameters if you want to give more weight to specific components
        P_data[0::12] = 10  # position along x
        P_data[1::12] = 10  # position along y
        P_data[2::12] = 2000  # position along z
        P_data[3::12] = 1  # roll
        P_data[4::12] = 5  # pitch
        P_data[5::12] = 1  # yaw
        P_data[6::12] = 200  # linear velocity along x
        P_data[7::12] = 200  # linear velocity along y
        P_data[8::12] = 100  # linear velocity along z
        P_data[9::12] = 1  # angular velocity along x
        P_data[10::12] = 1  # angular velocity along y
        P_data[11::12] = 1  # angular velocity along z

        # Define weights for the force components of the optimization vector
        P_row = np.hstack((P_row, np.arange(n_x * self.n_steps, n_x * self.n_steps * 2, 1)))
        P_col = np.hstack((P_col, np.arange(n_x * self.n_steps, n_x * self.n_steps * 2, 1)))
        P_data = np.hstack((P_data, 0.0*np.ones((n_x * self.n_steps * 2 - n_x * self.n_steps,))))

        P_data[(n_x * self.n_steps)::3] = 5e-3  # force along x
        P_data[(n_x * self.n_steps + 1)::3] = 5e-3  # force along y
        P_data[(n_x * self.n_steps + 2)::3] = 5e-3  # force along z

        # Convert P into a csc matrix for the solver
        self.P = scipy.sparse.csc.csc_matrix((P_data, (P_row, P_col)), shape=(
            n_x * self.n_steps * 2, n_x * self.n_steps * 2))

        # Declaration of the Q matrix in "x^T.P.x + x^T.Q"
        self.Q = np.hstack((np.zeros(n_x * self.n_steps,), 0.00 *
                            np.ones((n_x * self.n_steps * 2-n_x * self.n_steps, ))))

        # Weight for the z component of contact forces (fz > 0 so with a positive weight it tries to minimize fz)
        # q[(n_x * self.n_steps+2)::3] = 0.01

        return 0

    def update_matrices(self, sequencer, fstep_planner, mpc_interface):
        """Update the M, N, L and K constraint matrices depending on what happened
        """

        # M need to be updated between each iteration:
        # - lever_arms changes since the robot moves
        # - I_inv changes if the reference velocity vector is modified
        # - footholds need to be enabled/disabled depending on the contact sequence
        # self.update_M(sequencer, fstep_planner, mpc_interface)
        self.update_ML(sequencer, fstep_planner, mpc_interface)

        # N need to be updated between each iteration:
        # - X0 changes since the robot moves
        # - Xk* changes since X0 is not the same
        # self.update_N()
        self.update_NK()

        """print("ML equal M stacked L: ", np.array_equal(np.vstack((self.M, self.L)), self.ML.toarray()))
        print("NK equal N stacked K: ", np.array_equal(np.vstack((self.N, np.array([self.K]).transpose())), self.NK))"""

        # L matrix is constant
        # K matrix is constant

        return 0

    def update_M(self, sequencer, fstep_planner, mpc_interface):

        fth_w = np.zeros((4, self.n_steps, 3))

        # self.footholds contains the current position of feet in local frame
        future_fth = self.footholds.copy()
        S_tmp = sequencer.S.copy()

        # Put the future position of feet in swing phase in tmp
        fstep_planner.get_prediction(S_tmp, sequencer.t_stance,
                                     sequencer.T_gait, self.q, self.v, self.v_ref)
        for i in np.where(S_tmp[0, :] == False)[1]:
            future_fth[:, i] = fstep_planner.footsteps_prediction[:, i]

        # print("####")
        # print(future_fth[0, :])
        # The left part of M with A and identity matrices is constant

        # The right part of M need to be updated because B matrices are modified
        # Only the last rows of B need to be updated (those with lever arms of footholds)
        for k in range(self.n_steps):
            # Get inverse of the inertia matrix for time step k
            c, s = np.cos(self.xref[5, k]), np.sin(self.xref[5, k])
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
            I_inv = np.linalg.inv(np.dot(R, self.gI))

            if k > 0:
                S_tmp = np.roll(S_tmp, -1, axis=0)
                update = np.where((S_tmp[0, :] == False) & (S_tmp[-1, :] == True))[1]
                if np.any(update):
                    fstep_planner.get_prediction(S_tmp, sequencer.t_stance,
                                                 sequencer.T_gait, self.q, self.v, self.v_ref)
                    T = (self.xref[0:3, k] - self.xref[0:3, 0])
                    for i in update:
                        future_fth[0:2, i] = (np.dot(R, fstep_planner.footsteps_prediction[:, i]) + T)[0:2]

            # print(future_fth[0, :])

            for i in range(4):
                fth_w[i, k:(k+1), :] = (mpc_interface.oMl * future_fth[:, i]).transpose()

            # Get skew-symetric matrix for each foothold
            lever_arms = future_fth - self.xref[0:3, k:(k+1)]
            for i in range(4):
                self.B[-3:, (i*3):((i+1)*3)] = self.dt * np.dot(I_inv, utils.getSkew(lever_arms[:, i]))

            self.M[(k*12):((k+1)*12), (12*(self.n_steps+k)):(12*(self.n_steps+k+1))] = self.B

        # Update lines to enable/disable forces
        self.M[np.arange(12*self.n_steps, 12*self.n_steps*2, 1), np.arange(12*self.n_steps,
                                                                           12*self.n_steps*2, 1)] = 1 - np.repeat(sequencer.S.reshape((-1,)), 3)

        """plt.figure()
        for i in range(4):
            plt.subplot(4, 2, 2*i+1)
            plt.plot(fth_w[i, :, 0], linewidth=2)
            plt.plot([mpc_interface.o_feet[0, i]], marker="x")
            plt.xlabel("Time [s]")
            plt.ylabel("Position X [m]")
            plt.subplot(4, 2, 2*i+2)
            plt.plot(fth_w[i, :, 1], linewidth=2)
            plt.plot([mpc_interface.o_feet[1, i]], marker="x")
            plt.xlabel("Time [s]")
            plt.ylabel("Position Y [m]")
        plt.show(block=True)"""

        return 0

    def update_ML(self, sequencer, fstep_planner, mpc_interface):

        fth_w = np.zeros((4, self.n_steps, 3))

        # self.footholds contains the current position of feet in local frame
        future_fth = self.footholds.copy()
        print(self.footholds[0:2, :])
        S_tmp = sequencer.S.copy()

        # Put the future position of feet in swing phase in tmp
        fstep_planner.get_prediction(S_tmp, sequencer.t_stance,
                                     sequencer.T_gait, self.q, self.v, self.v_ref)
        for i in np.where(S_tmp[0, :] == False)[1]:
            future_fth[:, i] = fstep_planner.footsteps_prediction[:, i]

        # print("####")
        # print(future_fth[0, :])
        # The left part of M with A and identity matrices is constant

        # The right part of M need to be updated because B matrices are modified
        # Only the last rows of B need to be updated (those with lever arms of footholds)
        # Get inverse of the inertia matrix for time step k
        """c, s = np.cos(self.xref[5, 1]), np.sin(self.xref[5, 1])
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        I_inv = np.linalg.inv(np.dot(R, self.gI))"""
        for k in range(self.n_steps):
            # Get inverse of the inertia matrix for time step k
            c, s = np.cos(self.xref[5, k]), np.sin(self.xref[5, k])
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
            I_inv = np.linalg.inv(np.dot(R, self.gI))

            if k > 0:
                # S_tmp = np.roll(S_tmp, -1, axis=0)
                update = np.where((S_tmp[(k % self.n_steps), :] == False) & (S_tmp[k-1, :] == True))[1]

                if np.any(update):
                    fstep_planner.get_prediction(np.roll(S_tmp, -k, axis=0), sequencer.t_stance,
                                                 sequencer.T_gait, self.q, self.v, self.v_ref)
                    T = (self.xref[0:3, k] - self.xref[0:3, 0])
                    for i in update:
                        future_fth[0:2, i] = (np.dot(R, fstep_planner.footsteps_prediction[:, i]) + T)[0:2]

            # print(future_fth[0, :])

            """for i in range(4):
                fth_w[i, k:(k+1), :] = (mpc_interface.oMl * future_fth[:, i]).transpose()"""

            # Get skew-symetric matrix for each foothold
            lever_arms = future_fth - self.xref[0:3, k:(k+1)]
            for i in range(4):
                self.B[-3:, (i*3):((i+1)*3)] = self.dt * np.dot(I_inv, utils.getSkew(lever_arms[:, i]))

            # self.ML[(k*12):((k+1)*12), (12*(self.n_steps+k)):(12*(self.n_steps+k+1))] = self.B

            """self.ML.data[(30*self.n_steps-18+36*k):(30*self.n_steps-18+36*(k+1))
                         ] = self.B[i_x, i_y]"""

            i_iter = 24 * 4 * k
            self.ML.data[self.i_update_B + i_iter] = self.B[self.i_x_B, self.i_y_B]

        # Update lines to enable/disable forces
        tmp = 1 - np.repeat(sequencer.S.reshape((-1,)), 3)
        """self.ML[np.arange(12*self.n_steps, 12*self.n_steps*2, 1),
                np.arange(12*self.n_steps, 12*self.n_steps*2, 1)] = scipy.sparse.csc.csc_matrix(tmp, shape=tmp.shape)"""
        self.ML.data[self.i_update_S] = tmp.ravel()

        """self.ML.data[(66*self.n_steps-18):((66*self.n_steps-18)+12*self.n_steps)
                     ] = (1 - np.repeat(sequencer.S.reshape((-1,)), 3)).ravel()"""

        """plt.figure()
        for i in range(4):
            plt.subplot(4, 2, 2*i+1)
            plt.plot(fth_w[i, :, 0], linewidth=2)
            plt.plot([mpc_interface.o_feet[0, i]], marker="x")
            plt.xlabel("Time [s]")
            plt.ylabel("Position X [m]")
            plt.subplot(4, 2, 2*i+2)
            plt.plot(fth_w[i, :, 1], linewidth=2)
            plt.plot([mpc_interface.o_feet[1, i]], marker="x")
            plt.xlabel("Time [s]")
            plt.ylabel("Position Y [m]")
        plt.show(block=True)"""

        return 0

    def update_N(self):
        """ Create the N matrix involved in the MPC constraint equations M.X = N and L.X <= K """

        # Matrix g is already created and not changed
        # Fill N matrix with g matrices
        for k in range(self.n_steps):
            self.N[(12*k):(12*(k+1)), 0:1] = - self.g

        # Including - A*X0 in the first row of N
        self.N[0:12, 0:1] += np.dot(self.A, - self.x0)

        # Matrix D is already created and not changed
        # Add third term to matrix N
        self.N[0:12*self.n_steps, 0:1] += np.dot(self.D, self.xref[:, 1:].reshape((-1, 1), order='F'))

        return 0

    def update_NK(self):
        """ Update the N and K matrices involved in the MPC constraint equations M.X = N and L.X <= K """

        # Matrix g is already created and not changed
        # Fill N matrix with g matrices
        for k in range(self.n_steps):
            self.NK[(12*k):(12*(k+1)), 0:1] = - self.g

        # Including - A*X0 in the first row of N
        self.NK[0:12, 0:1] += np.dot(self.A, - self.x0)

        # Matrix D is already created and not changed
        # Add third term to matrix N
        self.NK[0:12*self.n_steps, 0:1] += np.dot(self.D, self.xref[:, 1:].reshape((-1, 1), order='F'))

        return 0

    def call_solver(self, sequencer):
        """Create an initial guess and call the solver to solve the QP problem
        """

        # Initial guess for forces (mass evenly supported by all legs in contact)
        f_temp = np.zeros((12*self.n_steps))
        # f_temp[2::3] = 2.2 * 9.81 / np.sum(sequencer.S[0,:])
        tmp = np.array(np.sum(sequencer.S, axis=1)).ravel().astype(int)

        # Initial guess of "mass/4" for time step with 4 feet in contact and "mass/2" for 2 feet in contact
        f_temp[2::3] = (np.repeat(tmp, 4)-4) / (2 - 4) * (self.mass * 9.81 * 0.5) + \
            (np.repeat(tmp, 4)-2) / (4 - 2) * (self.mass * 9.81 * 0.25)

        # Keep initial guess only for enabled feet
        # f_temp = np.array(np.multiply(np.repeat(sequencer.S.reshape((-1,)), 3), f_temp)).flatten()
        f_temp = self.x[self.xref.shape[0] * (self.xref.shape[1]-1):]

        # Initial guess (current state + guess for forces) to warm start the solver
        initx = np.hstack((np.zeros((12 * self.n_steps,)), np.roll(f_temp, -12)))

        # Create the QP solver object
        prob = osqp.OSQP()

        # Stack equality and inequality matrices
        """inf_lower_bound = -np.inf * np.ones((20*self.n_steps,))
        inf_lower_bound[4::5] = - 25

        print("###")
        t0 = clock()
        self.qp_A = scipy.sparse.vstack([self.L, self.M]).tocsc()
        self.qp_l = np.hstack([inf_lower_bound, self.N.ravel()])
        self.qp_u = np.hstack([self.K, self.N.ravel()])
        print(clock() - t0)"""

        self.qp_Abis = self.ML
        self.qp_ubis = self.NK.ravel()
        self.NK_inf[:12*self.n_steps * 2] = self.NK[:12*self.n_steps * 2, 0]

        # Setup the solver with the matrices and a warm start
        prob.setup(P=self.P, q=self.Q, A=self.qp_Abis, l=self.NK_inf, u=self.qp_ubis, verbose=False)
        prob.warm_start(x=initx)
        """
        else:  # Code to update the QP problem without creating it again
            qp_A = scipy.sparse.vstack([G, A]).tocsc()
            qp_l = np.hstack([l, b])
            qp_u = np.hstack([h, b])
            prob.update(A=qp_A, l=qp_l, u=qp_u)
        """

        # Run the solver to solve the QP problem
        # x = solve_qp(P, q, G, h, A, b, solver='osqp')
        self.x = prob.solve().x

        return 0

    def retrieve_result(self):
        """Extract relevant information from the output of the QP solver
        """

        # Retrieve the "robot state vector" part of the solution of the QP problem
        self.x_robot = (self.x[0:(self.xref.shape[0]*(self.xref.shape[1]-1))]
                        ).reshape((self.xref.shape[0], self.xref.shape[1]-1), order='F')

        # Retrieve the "contact forces" part of the solution of the QP problem
        self.f_applied = self.x[self.xref.shape[0]*(self.xref.shape[1]-1):(self.xref.shape[0] *
                                                                           (self.xref.shape[1]-1)
                                                                           + 12)]

        # As the QP problem is solved for (x_robot - x_ref), we need to add x_ref to the result to get x_robot
        self.x_robot += self.xref[:, 1:]

        # Predicted position and velocity of the robot during the next time step
        self.q_next = self.x_robot[0:6, 0:1]
        self.v_next = self.x_robot[6:12, 0:1]

        return 0

    def run(self, k, sequencer, fstep_planner, ftraj_gen, mpc_interface):

        # Get number of feet in contact with the ground for each step of the gait sequence
        if k > 0:
            self.n_contacts = np.roll(self.n_contacts, -1, axis=0)

        # Get the reference trajectory over the prediction horizon
        self.getRefStates(k, sequencer, mpc_interface)

        # Retrieve data from FootstepPlanner and FootTrajectoryGenerator
        self.retrieve_data(fstep_planner, ftraj_gen, mpc_interface)

        # Create the constraint and weight matrices used by the QP solver
        # Minimize x^T.P.x + x^T.Q with constraints M.X == N and L.X <= K
        if k == 0:
            self.create_matrices(sequencer)
        else:
            self.update_matrices(sequencer, fstep_planner, mpc_interface)

        # Create an initial guess and call the solver to solve the QP problem
        self.call_solver(sequencer)

        # Extract relevant information from the output of the QP solver
        self.retrieve_result()

        # Variation of position in world frame using the linear speed in local frame
        c_yaw, s_yaw = np.cos(self.q_w[5, 0]), np.sin(self.q_w[5, 0])
        R = np.array([[c_yaw, -s_yaw], [s_yaw, c_yaw]])
        self.q_w[0:2, 0:1] += np.dot(R, self.q_next[0:2, 0:1])
        self.q_w[2, 0] = self.q_next[2, 0]

        # Variation of orientation in world frame using the angular speed in local frame
        self.q_w[3:5, 0] = self.q_next[3:5, 0]
        self.q_w[5, 0] += self.q_next[5, 0]

        print("###")
        print("k: ", k)
        print(mpc_interface.lC)
        print(self.f_applied)
        print(self.q_next)

        return 0

    def plot_graphs(self, sequencer):

        # Display the predicted trajectory along X, Y and Z for the current iteration
        log_t = self.dt * np.arange(0, self.x_robot.shape[1], 1)

        plt.figure()
        plt.subplot(3, 2, 1)
        plt.plot(log_t, self.x_robot[0, :], "b", linewidth=2)
        plt.plot(log_t, self.xref[0, 1:], "r", linewidth=2)
        plt.xlabel("Time [s]")
        plt.ylabel("Position along X [m]")
        plt.legend(["Prediction", "Reference"])
        plt.subplot(3, 2, 3)
        plt.plot(log_t, self.x_robot[1, :], "b", linewidth=2)
        plt.plot(log_t, self.xref[1, 1:], "r", linewidth=2)
        plt.xlabel("Time [s]")
        plt.ylabel("Position along Y [m]")
        plt.legend(["Prediction", "Reference"])
        plt.subplot(3, 2, 5)
        plt.plot(log_t, self.x_robot[2, :], "b", linewidth=2)
        plt.plot(log_t, self.xref[2, 1:], "r", linewidth=2)
        plt.xlabel("Time [s]")
        plt.ylabel("Position along Z [m]")
        plt.legend(["Prediction", "Reference"])

        plt.subplot(3, 2, 2)
        plt.plot(log_t, self.x_robot[3, :], "b", linewidth=2)
        plt.plot(log_t, self.xref[3, 1:], "r", linewidth=2)
        plt.xlabel("Time [s]")
        plt.ylabel("Orientation along X [m]")
        plt.legend(["Prediction", "Reference"])
        plt.subplot(3, 2, 4)
        plt.plot(log_t, self.x_robot[4, :], "b", linewidth=2)
        plt.plot(log_t, self.xref[4, 1:], "r", linewidth=2)
        plt.xlabel("Time [s]")
        plt.ylabel("Orientation along Y [m]")
        plt.legend(["Prediction", "Reference"])
        plt.subplot(3, 2, 6)
        plt.plot(log_t, self.x_robot[5, :], "b", linewidth=2)
        plt.plot(log_t, self.xref[5, 1:], "r", linewidth=2)
        plt.xlabel("Time [s]")
        plt.ylabel("Orientation along Z [m]")
        plt.legend(["Prediction", "Reference"])
        plt.show(block=True)

        # Display the desired contact forces for each foot over the prediction horizon for the current iteration
        f_1 = np.zeros((3, (self.xref.shape[1]-1)))
        f_2 = np.zeros((3, (self.xref.shape[1]-1)))
        f_3 = np.zeros((3, (self.xref.shape[1]-1)))
        f_4 = np.zeros((3, (self.xref.shape[1]-1)))
        fs = [f_1, f_2, f_3, f_4]
        cpt_tot = 0
        for i_f in range((self.xref.shape[1]-1)):
            up = (sequencer.S[i_f, :] == 1)
            for i_up in range(4):
                if up[0, i_up] == True:
                    (fs[i_up])[:, i_f] = self.x[(self.xref.shape[0]*(self.xref.shape[1]-1) + 3 * cpt_tot):
                                                (self.xref.shape[0]*(self.xref.shape[1]-1) + 3 * cpt_tot + 3)]
                    cpt_tot += 1

        plt.close()
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.title("Front left")
        plt.plot(f_1[0, :], linewidth=2)
        plt.plot(f_1[1, :], linewidth=2)
        plt.plot(f_1[2, :], linewidth=2)
        plt.legend(["X component", "Y component", "Z component"])
        plt.subplot(2, 2, 2)
        plt.title("Front right")
        plt.plot(f_2[0, :], linewidth=2)
        plt.plot(f_2[1, :], linewidth=2)
        plt.plot(f_2[2, :], linewidth=2)
        plt.legend(["X component", "Y component", "Z component"])
        plt.subplot(2, 2, 3)
        plt.title("Hindleft")
        plt.plot(f_3[0, :], linewidth=2)
        plt.plot(f_3[1, :], linewidth=2)
        plt.plot(f_3[2, :], linewidth=2)
        plt.legend(["X component", "Y component", "Z component"])
        plt.subplot(2, 2, 4)
        plt.title("Hind right")
        plt.plot(f_4[0, :], linewidth=2)
        plt.plot(f_4[1, :], linewidth=2)
        plt.plot(f_4[2, :], linewidth=2)
        plt.legend(["X component", "Y component", "Z component"])
        plt.show(block=True)

        return 0
