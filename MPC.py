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
        self.mass = 3.0

        # Inertia matrix of the robot in body frame (found in urdf)
        self.gI = np.diag([0.00578574, 0.01938108, 0.02476124])

        # Friction coefficient
        self.mu = 2

        # Number of time steps in the prediction horizon
        self.n_steps = sequencer.S.shape[0]

        # Reference trajectory matrix of size 12 by (1 + N)  with the current state of
        # the robot in column 0 and the N steps of the prediction horizon in the others
        self.xref = np.zeros((12, 1 + self.n_steps))

        # Initial state vector of the robot (x, y, z, roll, pitch, yaw)
        self.q = np.array([[0.0, 0.0, 0.235 - 0.01205385, 0.0, 0.0, 0.0]]).transpose()

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

    def update_v_ref(self, joystick):

        # Retrieving the reference velocity from the joystick
        self.v_ref = joystick.v_ref

        # Get the reference velocity in global frame
        c, s = np.cos(self.q_w[5, 0]), np.sin(self.q_w[5, 0])
        R = np.array([[c, -s, 0., 0., 0., 0.], [s, c, 0., 0., 0., 0], [0., 0., 1.0, 0., 0., 0.],
                      [0., 0., 0., c, -s, 0.], [0., 0., 0., s, c, 0.], [0., 0., 0., 0., 0., 1.0]])
        self.v_ref_world = np.dot(R, self.v_ref)

        return 0

    def getRefStatesDuringTrajectory(self, sequencer):
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

        # TODO: Put stuff directly in x_ref instead of allocating qu_ref temporarily each time the function is called

        n_steps = int(np.round(sequencer.T_gait/self.dt))
        qu_ref = np.zeros((6, n_steps))

        dt_vector = np.linspace(self.dt, sequencer.T_gait, n_steps)
        qu_ref = self.v_ref_world * dt_vector

        # Take into account the rotation of the base over the prediction horizon
        yaw = np.linspace(0, sequencer.T_gait-self.dt, n_steps) * self.v_ref_world[5, 0]
        qu_ref[0, :] = self.dt * np.cumsum(self.v_ref_world[0, 0] * np.cos(yaw) -
                                           self.v_ref_world[1, 0] * np.sin(yaw))
        qu_ref[1, :] = self.dt * np.cumsum(self.v_ref_world[0, 0] * np.sin(yaw) +
                                           self.v_ref_world[1, 0] * np.cos(yaw))

        # Stack the reference velocity to the reference position to get the reference state vector
        self.x_ref = np.vstack((qu_ref, np.tile(self.v_ref_world, (1, n_steps))))

        # Desired height is supposed constant
        self.x_ref[2, :] = self.h_ref

        # Stack the reference trajectory (future states) with the current state
        self.xref[6:, 0:1] = self.v_ref
        self.xref[:, 1:] = self.x_ref
        self.xref[2, 0] = self.h_ref

        # Current state vector of the robot
        self.x0 = np.vstack((self.q, self.v))

        return 0

    def create_M(self, sequencer):

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
        self.M[np.arange(12*self.n_steps, 12*self.n_steps*2, 1), np.arange(12*self.n_steps, 12*self.n_steps*2, 1)] = np.repeat(sequencer.S.reshape((-1,)),3)

        return 0

    def create_N(self):

        # Create N matrix
        self.N = np.zeros((12*self.n_steps, 1))

        # Create g matrix
        g = np.zeros((12, 1))
        g[8, 0] = -9.81 * self.dt

        # Fill N matrix with g matrices
        for k in range(self.n_steps):
            self.N[(12*k):(12*(k+1)), 0:1] = - g

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

        # Reshape N into one dimensional array
        self.N = self.N.reshape((-1,))

        return 0

    def create_L(self):

        # Create L matrix
        self.L = np.zeros((20*self.n_steps, 12*self.n_steps*2))

        # Create C matrix
        self.C = np.zeros((5, 3))
        self.C[[0, 1, 2, 3] * 2 + [4], [0, 0, 1, 1, 2, 2, 2, 2, 2]] = np.array([1, -1, 1, -1, -self.mu, -self.mu, -self.mu, -self.mu, -1])

        # Create F matrix
        self.F = np.zeros((20, 12))
        for i in range(4):
            self.F[(5*i):(5*(i+1)), (3*i):(3*(i+1))] = self.C

        # Fill L matrix with F matrices
        for k in range(self.n_steps):
            self.L[(20*k):(20*(k+1)), (12*(self.n_steps+k)):(12*(self.n_steps+1+k))] = self.F

        return 0

    def create_K(self):

        # Create K matrix
        self.K = np.zeros((20*self.n_steps, ))

        return 0

