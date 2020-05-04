# coding: utf8

import numpy as np
import gamepadClient as gC


class Joystick:
    """Joystick-like controller that outputs the reference velocity in local frame
    """

    def __init__(self):

        # Reference velocity in local frame
        self.v_ref = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T

        self.reduced = False

        # Joystick variables (linear and angular velocity and their scaling for the joystick)
        self.vX = 0.
        self.vY = 0.
        self.vYaw = 0.
        self.VxScale = 0.2
        self.VyScale = 0.4
        self.vYawScale = 0.4

    def update_v_ref(self, k_loop):
        """Update the reference velocity of the robot along X, Y and Yaw in local frame by
        listening to a gamepad handled by an independent thread

        Args:
            k_loop (int): number of MPC iterations since the start of the simulation
        """

        if k_loop == 0:
            self.gp = gC.GamepadClient()

        self.vX = self.gp.leftJoystickX.value * self.VxScale
        self.vY = self.gp.leftJoystickY.value * self.VyScale
        self.vYaw = self.gp.rightJoystickX.value * self.vYawScale

        self.v_ref = np.array([[- self.vY, - self.vX, 0.0, 0.0, 0.0, - self.vYaw]]).T

        if self.gp.startButton.value == True:
            self.reduced = not self.reduced

        return 0

    def update_v_ref_predefined(self, k_loop):
        """Update the reference velocity of the robot along X, Y and Yaw in local frame
        according to a predefined sequence

        Args:
            k_loop (int): number of MPC iterations since the start of the simulation
        """

        # Moving forwards
        if k_loop == 20*16:
            self.v_ref = np.array([[0.3, 0.0, 0.0, 0.0, 0.0, 0.0]]).T

        # Turning
        if k_loop == 2000:
            self.v_ref = np.array([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0]]).T

        # Turning
        if k_loop == 4000:
            self.v_ref = np.array([[0.3, 0.0, 0.0, 0.0, 0.0, 0.0]]).T

        # Turning
        if k_loop == 6000:
            self.v_ref = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T

        """# Moving forwards
        if k_loop == 200:
            self.v_ref = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.4]]).T

        # Turning
        if k_loop == 4200:
            self.v_ref = np.array([[0.3, 0.0, 0.0, 0.0, 0.0, 0.0]]).T"""

        # Moving forwards
        """if k_loop == 16000:
            self.v_ref = np.array([[0.3, 0.0, 0.0, 0.0, 0.0, 0.0]]).T

        # Stoping
        if k_loop == 35000:
            self.v_ref = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T

        # Sideways
        if k_loop == 45000:
            self.v_ref = np.array([[0.0, 0.1, 0.0, 0.0, 0.0, 0.0]]).T

        # Sideways + Turning
        if k_loop == 55000:
            self.v_ref = np.array([[0.0, 0.1, 0.0, 0.0, 0.0, 0.2]]).T"""

        return 0
