# coding: utf8

import numpy as np
from time import clock
import inputs


class Joystick:
    """Joystick-like controller that outputs the reference velocity in local frame
    """

    def __init__(self):

        # Starting time if we want to ouput reference velocities based on elapsed time
        self.t_start = clock()

        # Reference velocity in local frame
        self.v_ref = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T

        # Joystick variables
        self.vX = 0.
        self.vY = 0.
        self.vYaw = 0.
        self.VxScale = 0.1/32768
        self.VyScale = 0.1/32768
        self.vYawScale = 0.4/32768

    def update_v_ref(self, k_loop):
        """events = inputs.get_gamepad()
        for event in events:
            # print(event.ev_type, event.code, event.state)
            if (event.ev_type == 'Absolute'):
                if event.code == 'ABS_X':
                    self.vX = event.state * self.VxScale
                if event.code == 'ABS_Y':
                    self.vY = event.state * self.VyScale
                if event.code == 'ABS_RX':
                    self.vYaw = event.state * self.vYawScale
                print(- self.vY, - self.vX, - self.vYaw)

        self.v_ref = np.array([[- self.vY, - self.vX, 0.0, 0.0, 0.0, - self.vYaw]]).T"""

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
