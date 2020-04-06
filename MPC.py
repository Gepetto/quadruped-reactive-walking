# coding: utf8

import numpy as np
import scipy as scipy
import osqp as osqp
from matplotlib import pyplot as plt
import utils


class MPC:
    """Wrapper for the MPC to create constraint matrices, call the QP solver and
    retrieve the result.

    Args:
        dt (float): time step of the MPC
        n_steps (int): number of time step in one gait cycle
        n_contacts (int): cumulative number of feet touching the ground in one gait cycle, for instance if 4 feet
                          touch the ground during 10 time steps then 2 feet during 5 time steps then n_contacts = 50

    """

    def __init__(self, dt, n_steps):

        # Time step of the MPC solver
        self.dt = dt

        # Mass of the robot
        self.mass = 2.97784899

        # Inertia matrix of the robot in body frame (found in urdf)
        self.gI = np.diag([0.00578574, 0.01938108, 0.02476124])

        # Friction coefficient
        self.mu = 1

        # Number of time steps in the prediction horizon
        self.n_steps = n_steps

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

        # Initial position of footholds in the "straight standing" default configuration
        self.footholds = np.array(
            [[0.19, 0.19, -0.19, -0.19],
             [0.15005, -0.15005, 0.15005, -0.15005],
             [0.0, 0.0, 0.0, 0.0]])

        # Create the QP solver object
        self.prob = osqp.OSQP()

        # Inversed S matrix
        # self.inverse_S = np.zeros((self.n_steps, 4))

        # Lever arms of contact forces for update_ML function
        self.lever_arms = np.zeros((3, 4))

        self.S_gait = np.zeros((12*self.n_steps,))
        self.gait = np.zeros((6, 5))

    def update_v_ref(self, joystick):
        """Get reference velocity in local frame from a joystick-like object (gamepad for instance)

        Args:
            joystick (object): a joystick-like object with a v_ref attribute
        """

        # Retrieving the reference velocity from the joystick
        self.v_ref = joystick.v_ref

        return 0

    def create_matrices(self):
        """Create the constraint matrices of the MPC (M.X = N and L.X <= K)
        Create the weight matrices P and Q of the MPC solver (cost 1/2 x^T * P * X + X^T * Q)
        """

        # Create the constraint matrices
        self.create_ML()
        self.create_NK()

        # Create the weight matrices
        self.create_weight_matrices()

        return 0

    def create_ML(self):
        """Create the M and L matrices involved in the MPC constraint equations M.X = N and L.X <= K
        """

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
        self.ML[np.arange(12*self.n_steps, 12*self.n_steps*2, 1),
                np.arange(12*self.n_steps, 12*self.n_steps*2, 1)] = np.ones((12*self.n_steps,))

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
        # self.ML.data[self.i_update_S] = (1 - np.repeat(self.S.reshape((-1,)), 3)).ravel()

        # Update lines to enable/disable forces
        self.construct_S(self.gait)
        self.ML.data[self.i_update_S] = self.S_gait

        return 0

    def create_NK(self):
        """Create the N and K matrices involved in the MPC constraint equations M.X = N and L.X <= K
        """

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
        self.inf_lower_bound[4::5] = - 25.0  # - maximum normal force
        self.NK[12*self.n_steps * 2 + 4::5] = 0.0  # - 1.0  # - minimum normal force

        self.NK_inf[:12*self.n_steps * 2] = self.NK[:12*self.n_steps * 2, 0]
        self.NK_inf[12*self.n_steps * 2:] = self.inf_lower_bound

        return 0

    def create_weight_matrices(self):
        """Create the weight matrices P and q in the cost function x^T.P.x + x^T.q of the QP problem
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
        P_data[3::12] = 250  # roll
        P_data[4::12] = 250  # pitch
        P_data[5::12] = 10  # yaw
        P_data[6::12] = 200  # linear velocity along x
        P_data[7::12] = 200  # linear velocity along y
        P_data[8::12] = 100  # linear velocity along z
        P_data[9::12] = 10  # angular velocity along x
        P_data[10::12] = 10  # angular velocity along y
        P_data[11::12] = 10  # angular velocity along z

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

    def update_matrices(self, fsteps):
        """Update the M, N, L and K constraint matrices depending on what happened

        Args:
            fsteps (Xx13 array): contains the remaining number of steps of each phase of the gait (first column) and
                                 the [x, y, z]^T desired position of each foot for each phase of the gait (12 other
                                 columns). For feet currently touching the ground the desired position is where they
                                 currently are.
        """

        # M need to be updated between each iteration:
        # - lever_arms changes since the robot moves
        # - I_inv changes if the reference velocity vector is modified
        # - footholds need to be enabled/disabled depending on the contact sequence
        self.update_ML(fsteps)

        # N need to be updated between each iteration:
        # - X0 changes since the robot moves
        # - Xk* changes since X0 is not the same
        self.update_NK()

        # L matrix is constant
        # K matrix is constant

        return 0

    def update_ML(self, fsteps):
        """Update the M and L constaint matrices depending on the current state of the gait

        Args:
            fsteps (Xx13 array): contains the remaining number of steps of each phase of the gait (first column) and
                                 the [x, y, z]^T desired position of each foot for each phase of the gait (12 other
                                 columns). For feet currently touching the ground the desired position is where they
                                 currently are.
        """

        # Replace NaN values by zeroes
        fsteps[np.isnan(fsteps)] = 0.0

        # Compute cosinus and sinus of the yaw angle for the whole prediction horizon
        c, s = np.cos(self.xref[5, :]), np.sin(self.xref[5, :])

        j = 0
        k_cum = 0

        # Iterate over all phases of the gait
        while (self.gait[j, 0] != 0):
            for k in range(k_cum, k_cum+np.int(self.gait[j, 0])):
                # Get inverse of the inertia matrix for time step k
                R = np.array([[c[k], -s[k], 0], [s[k], c[k], 0], [0, 0, 1.0]])
                I_inv = np.linalg.inv(np.dot(R, self.gI))

                # Get skew-symetric matrix for each foothold
                self.lever_arms = np.reshape(fsteps[j, 1:], (3, 4), order='F') - self.xref[0:3, k:(k+1)]
                for i in range(4):
                    self.B[-3:, (i*3):((i+1)*3)] = self.dt * np.dot(I_inv, utils.getSkew(self.lever_arms[:, i]))

                # Replace the coefficient directly in ML.data
                i_iter = 24 * 4 * k
                self.ML.data[self.i_update_B + i_iter] = self.B[self.i_x_B, self.i_y_B]

            k_cum += np.int(self.gait[j, 0])
            j += 1

        # Construct the activation/desactivation matrix based on the current gait
        self.construct_S(self.gait)

        # Update lines to enable/disable forces
        self.ML.data[self.i_update_S] = self.S_gait

        return 0

    def update_NK(self):
        """ Update the N and K matrices involved in the MPC constraint equations M.X = N and L.X <= K
        """

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

    def call_solver(self, k):
        """Create an initial guess and call the solver to solve the QP problem

        Args:
            k (int): number of MPC iterations since the start of the simulation
        """

        # Initial guess for forces (mass evenly supported by all legs in contact)
        """
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
        initx = np.hstack((np.zeros((12 * self.n_steps,)), np.roll(f_temp, -12)))"""

        # Copy the "equality" part of NK on the other side of the constaint
        # since NK_inf <= A X <= NK
        self.NK_inf[:12*self.n_steps * 2] = self.NK[:12*self.n_steps * 2, 0]

        # Setup the solver (first iteration) then just update it
        if k == 0:  # Setup the solver with the matrices
            self.prob.setup(P=self.P, q=self.Q, A=self.ML, l=self.NK_inf, u=self.NK.ravel(), verbose=False)
            # self.prob.warm_start(x=initx)
        else:  # Code to update the QP problem without creating it again
            self.prob.update(Ax=self.ML.data, l=self.NK_inf, u=self.NK.ravel())

        # Run the solver to solve the QP problem
        self.x = self.prob.solve().x

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

    def run(self, k, T_gait, t_stance, lC, abg, lV, lW, l_feet, xref, x0, v_ref, fsteps):
        """Run one iteration of the whole MPC by calling all the necessary functions (data retrieval,
           update of constraint matrices, update of the solver, running the solver, retrieving result)

        Args:
            k (int): the number of MPC iterations since the start of the simulation
            T_gait (float): duration of one period of gait
            t_stance (float): duration of one stance phase
            lC (3x0 array): position of the center of mass in local frame
            abg (3x0 array): orientation of the trunk in local frame
            lV (3x0 array): linear velocity of the CoM in local frame
            lW (3x0 array): angular velocity of the trunk in local frame
            l_feet (3x4 array): current position of feet in local frame
            xref (12x(N+1) array): current state vector of the robot (first column) and future desired state vectors
                                   (other columns). N is the number of time step in the prediction horizon
            x0 (12x1 array): current state vector of the robot (position/orientation/linear vel/angular vel)
            v_ref (6x1 array): desired velocity vector of the flying base in local frame (linear and angular stacked)
            fsteps (Xx13 array): contains the remaining number of steps of each phase of the gait (first column) and
                                 the [x, y, z]^T desired position of each foot for each phase of the gait (12 other
                                 columns). For feet currently touching the ground the desired position is where they
                                 currently are.
        """

        # Recontruct the gait based on the computed footsteps
        self.construct_gait(fsteps)

        # Retrieving the reference velocity from the joystick
        self.v_ref = v_ref

        # Update MPC's state vectors by retrieving information from the mpc_interface
        if k > 0:
            self.q[0:3, 0:1] = lC
            self.q[3:6, 0:1] = abg
            self.v[0:3, 0:1] = lV
            self.v[3:6, 0:1] = lW

        # Retrieve data required for the MPC
        self.T_gait = T_gait
        self.t_stance = t_stance
        self.lC = lC
        self.footholds[0:2, :] = l_feet[0:2, :]
        self.xref = xref
        self.x0 = x0

        # Create the constraint and weight matrices used by the QP solver
        # Minimize x^T.P.x + x^T.Q with constraints M.X == N and L.X <= K
        if k == 0:
            self.create_matrices()
        else:
            self.update_matrices(fsteps)

        # Create an initial guess and call the solver to solve the QP problem
        self.call_solver(k)

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

        return 0

    def plot_graphs(self, sequencer):
        """Plot graphs

        Args:
            sequencer (object): ContactSequencer object
        """

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

    def construct_S(self, gait):
        """Construct an array of size 12*N that contains information about the contact state of feet.
           This matrix is used to enable/disable contact forces in the QP problem.
           N is the number of time step in the prediction horizon.

        Args:
            gait (Xx5 array): contains information about the remaining number of steps for each phase of the gait (1st
                              column) and information about the contact state of feet during each phase (4 other
                              columns). In gait[:, 1:], the coefficient (i, 1) is equal to 1.0 if the j-th feet is
                              touching the ground during the i-th time step of the prediction horizon, 0.0 otherwise.
        """

        i = 0
        k = 0

        while (gait[i, 0] != 0):

            self.S_gait[(k*12):((k+np.int(gait[i, 0]))*12)] = np.tile(np.repeat(1.0 - gait[i, 1:], 3),
                                                                      (np.int(gait[i, 0]),))
            k += np.int(gait[i, 0])
            i += 1

        return 0

    def construct_gait(self, fsteps):
        """Reconstruct the gait matrix based on the fsteps matrix since only the last one is received by the MPC

        Args:
            fsteps (Xx13 array): contains the remaining number of steps of each phase of the gait (first column) and
                                 the [x, y, z]^T desired position of each foot for each phase of the gait (12 other
                                 columns). For feet currently touching the ground the desired position is where they
                                 currently are.
        """

        # Index of the first empty line
        index = next((idx for idx, val in np.ndenumerate(fsteps[:, 0]) if val==0.0), 0.0)[0]

        self.gait[:, 0] = fsteps[:, 0]

        self.gait[:index, 1:] = 1.0 - (np.isnan(fsteps[:index, 1::3]) | (fsteps[:index, 1::3] == 0.0))

        return 0
