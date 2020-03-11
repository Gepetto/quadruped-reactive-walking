# coding: utf8

import numpy as np
from time import clock


class Joystick:
    """Joystick-like controller that outputs the reference velocity in local frame
    """

    def __init__(self):

        # Starting time if we want to ouput reference velocities based on elapsed time
        self.t_start = clock()

        # Reference velocity in local frame
        self.v_ref = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T

    def update_v_ref(self, k_loop):

        # Change reference velocity during the simulation (in trunk frame)
        # Moving forwards
        if k_loop == 200:
            self.v_ref = np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
        """
        # Turning
        if k_loop == 1500:
            self.v_ref = np.array([[0.1, 0.0, 0.0, 0.0, 0.0, -0.2]]).T

        # Moving forwards
        if k_loop == 2500:
            self.v_ref = np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]]).T

        # Stoping
        if k_loop == 3500:
            self.v_ref = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T

        # Sideways
        if k_loop == 4500:
            self.v_ref = np.array([[0.0, 0.1, 0.0, 0.0, 0.0, 0.0]]).T

        # Sideways
        if k_loop == 5500:
            self.v_ref = np.array([[0.0, 0.1, 0.0, 0.0, 0.0, 0.2]]).T"""

        return 0
