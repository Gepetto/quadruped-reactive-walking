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
        self.v_ref = np.array([[0.05, 0.0, 0.0, 0.0, 0.0, 0.78]]).T

    def update_v_ref(self, k_loop):

        # Change reference velocity during the simulation (in trunk frame)

        # Moving forwards
        """if k_loop == 75:
            self.v_ref = np.array([[0.1, 0, 0.0, 0, 0, 0.0]]).T"""

        # Turning
        """if k_loop == 151:
            self.v_ref = np.array([[0.1, 0, 0.0, 0, 0, 0.4]]).T"""

        return 0
