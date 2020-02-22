# coding: utf8

import numpy as np
from time import clock
import scipy as scipy
import osqp as osqp
from matplotlib import pyplot as plt


class MPC:
    """Wrapper for the MPC to create constraint matrices, call the QP solver and
    retrieve the result.

    """

    def __init__(self, dt, sequencer):

        # Time step of the MPC solver
        self.dt = dt

        # Mass of the robot
        self.mass = 3.0

        # Number of time steps in the prediction horizon
        self.n_steps = sequencer.S.shape[0]

    def create_A(self):

        self.A = np.eye(12)
        self.A[[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]] = np.ones((6,)) * self.dt

        return 0

    def create_B(self):

        self.B = np.zeros((12, 12))
        self.B[np.tile([6, 7, 8], 4), np.arange(0, 12, 1)] = (self.dt / self.mass) * np.ones((12,))
        

