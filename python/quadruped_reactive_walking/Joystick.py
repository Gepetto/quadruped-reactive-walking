# coding: utf8

import numpy as np
import gamepadClient as gC


class Joystick:
    """Joystick-like controller that outputs the reference velocity in local frame

    Args:
        predefined (bool): use either a predefined velocity profile (True) or a gamepad (False)
    """

    def __init__(self, predefined, multi_simu=False):

        # Reference velocity in local frame
        self.v_ref = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T

        self.reduced = False
        self.stop = False

        dT = 0.0020  # velocity reference is updated every ms
        fc = 100  #  cutoff frequency
        y = 1 - np.cos(2*np.pi*fc*dT)
        self.alpha = -y+np.sqrt(y*y+2*y)

        tc = 0.02  #  cutoff frequency at 50 Hz
        dT = 0.0020  # velocity reference is updated every ms
        self.alpha = dT / tc

        # Bool to modify the update of v_ref
        # Used to launch multiple simulations
        self.multi_simu = multi_simu

        # If we are using a predefined reference velocity (True) or a joystick (False)
        self.predefined = predefined

        # Joystick variables (linear and angular velocity and their scaling for the joystick)
        self.vX = 0.
        self.vY = 0.
        self.vYaw = 0.
        self.VxScale = 0.6
        self.VyScale = 1.2
        self.vYawScale = 1.6

        self.Vx_ref = 0.3
        self.Vy_ref = 0.0
        self.Vw_ref = 0.0

        # Y, B, A and X buttons (in that order)
        self.northButton = False
        self.eastButton = False
        self.southButton = False
        self.westButton = False

    def update_v_ref(self, k_loop, velID):
        """Update the reference velocity of the robot along X, Y and Yaw in local frame by
        listening to a gamepad handled by an independent thread

        Args:
            k_loop (int): numero of the current iteration
            velID (int): Identifier of the current velocity profile to be able to handle different scenarios
        """

        if self.predefined:
            if self.multi_simu:
                self.update_v_ref_multi_simu(k_loop)
            else:
                self.update_v_ref_predefined(k_loop, velID)
        else:
            self.update_v_ref_gamepad(k_loop)

        return 0

    def update_v_ref_gamepad(self, k_loop):
        """Update the reference velocity of the robot along X, Y and Yaw in local frame by
        listening to a gamepad handled by an independent thread

        Args:
            k_loop (int): numero of the current iteration
        """

        # Create the gamepad client
        if k_loop == 0:
            self.gp = gC.GamepadClient()

        # Get the velocity command based on the position of joysticks
        self.vX = self.gp.leftJoystickX.value * self.VxScale
        self.vY = self.gp.leftJoystickY.value * self.VyScale
        self.vYaw = self.gp.rightJoystickX.value * self.vYawScale

        if self.gp.L1Button.value:  # If L1 is pressed the orientation of the base is controlled
            self.v_gp = np.array(
                [[0.0, 0.0, - self.vYaw * 0.25, - self.vX * 5, - self.vY * 2, 0.0]]).T
        else:  # Otherwise the Vx, Vy, Vyaw is controlled
            self.v_gp = np.array(
                [[- self.vY, - self.vX, 0.0, 0.0, 0.0, - self.vYaw]]).T

        # Reduce the size of the support polygon by pressing Start
        if self.gp.startButton.value:
            self.reduced = not self.reduced

        # Switch to safety controller if the Back key is pressed
        if self.gp.backButton.value:
            self.stop = True

        # Switch gaits
        if self.gp.northButton.value:
            self.northButton = True
            self.eastButton = False
            self.southButton = False
            self.westButton = False
        elif self.gp.eastButton.value:
            self.northButton = False
            self.eastButton = True
            self.southButton = False
            self.westButton = False
        elif self.gp.southButton.value:
            self.northButton = False
            self.eastButton = False
            self.southButton = True
            self.westButton = False
        elif self.gp.westButton.value:
            self.northButton = False
            self.eastButton = False
            self.southButton = False
            self.westButton = True

        # Low pass filter to slow down the changes of velocity when moving the joysticks
        self.v_ref = self.alpha * self.v_gp + (1-self.alpha) * self.v_ref
        self.v_ref[(self.v_ref < 0.005) & (self.v_ref > -0.005)] = 0.0

        return 0

    def handle_v_switch(self, k):
        """Handle the change of reference velocity according to the chosen predefined velocity profile

        Args:
            k (int): numero of the current iteration
        """

        i = 1
        while (i < self.k_switch.shape[0]) and (self.k_switch[i] <= k):
            i += 1
        if (i != self.k_switch.shape[0]):
            self.apply_velocity_change(k, i)

    def apply_velocity_change(self, k, i):
        """Change the velocity reference sent to the robot
        4-th order polynomial: zero force and force velocity at start and end
        (bell-like force trajectory)

        Args:
            k (int): numero of the current iteration
            i (int): numero of the active phase of the reference velocity profile
        """

        ev = k - self.k_switch[i-1]
        t1 = self.k_switch[i] - self.k_switch[i-1]
        A3 = 2 * (self.v_switch[:, (i-1):i] -
                  self.v_switch[:, i:(i+1)]) / t1**3
        A2 = (-3/2) * t1 * A3
        self.v_ref = self.v_switch[:, (i-1):i] + A2*ev**2 + A3*ev**3

        return 0

    def update_v_ref_predefined(self, k_loop, velID):
        """Update the reference velocity of the robot along X, Y and Yaw in local frame
        according to a predefined sequence

        Args:
            k_loop (int): numero of the current iteration
            velID (int): identifier of the current velocity profile to be able to handle different scenarios
        """

        if velID == 0:
            if (k_loop == 0):
                self.k_switch = np.array(
                    [0, 500, 2000, 3000, 4000, 13000, 20000, 30000])
                self.v_switch = np.array([[0.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0,  0.0, 0.0,
                                              0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0,  0.0, 0.0,
                                              0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0,  0.0, 0.0,
                                              0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0,  0.0, 0.0,
                                              0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        elif velID == 1:
            if (k_loop == 0):
                V_max = 0.5
                self.k_switch = np.array([0, 1000, 3000, 8000, 12000, 16000, 20000, 22000,
                                          23000, 26000, 30000, 33000, 34000, 40000, 41000, 43000,
                                          44000, 45000])
                self.v_switch = np.zeros((6, self.k_switch.shape[0]))
                self.v_switch[0, :] = np.array([0.0, 0.0, V_max, V_max, 0.0, 0.0, 0.0,
                                                0.0, -V_max, -V_max, 0.0, 0.0, 0.0, V_max, V_max, V_max,
                                                V_max, V_max])
                self.v_switch[1, :] = np.array([0.0, 0.0,  0.0, 0.0, -V_max, -V_max, 0.0,
                                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                0.0, 0.0])
                self.v_switch[5, :] = np.array([0.0, 0.0,  0.3, 0.3, 0.3, 0.0, 0.0, 0.0,
                                                0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.0, 0.0,
                                                -0.3, 0.0])
        elif velID == 2:
            self.k_switch = np.array([0, 10000, 20000, 30000])
            self.v_switch = np.array([[0.0, 0.5, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0]])
        elif velID == 3:
            if (k_loop == 0):
                self.k_switch = np.array([0, 1000, 2000, 7000, 26000, 30000])
                self.v_switch = np.array([[0.0, 0.0,  0.0, 0.3, 0.3, 0.0],
                                          [0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0,  0.3, 0.0, 0.0, 0.0]])
        elif velID == 4:
            if (k_loop == 0):
                self.k_switch = np.array([0, 1000, 3000, 7000, 9000, 30000])
                self.v_switch = np.array([[0.0, 0.0,  1.5, 1.5, 1.5, 1.5],
                                          [0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0,  0.0, 0.0, 0.4, 0.4]])
        elif velID == 5:
            if (k_loop == 0):
                """self.k_switch = np.array([0, 500, 1500, 2600, 5000, 6500, 8000])
                self.v_switch = np.array([[0.0, 0.0,  0.7, 0.6, 0.3, 0.3, 0.0],
                                          [0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0,  0.0, 0.4, 0.6, 0.0, 0.0]])"""
                self.k_switch = np.array([0, 500, 1500, 2600, 5000, 6500, 7000, 8000, 9000])
                self.v_switch = np.array([[0.0, 0.0,  0.5, 0.6, 0.3, 0.6, -0.5, 0.7, 0.0],
                                          [0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0,  0.2, 0.7, 0.7, 0.0, -0.4, -0.6, 0.0]])

        elif velID == 6:
            if (k_loop == 0):
                self.k_switch = np.array(
                    [0, 1000, 2500, 5000, 7500, 8000, 10000])
                self.v_switch = np.array([[0.0, 0.0,  0.8, 0.4, 0.8, 0.8, 0.0],
                                          [0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0,  0.0, 0.55, 0.3, 0.0, 0.0]])

        self.handle_v_switch(k_loop)
        return 0

    def update_v_ref_multi_simu(self, k_loop):
        """Update the reference velocity of the robot along X, Y and Yaw in local frame
        according to a predefined sequence

        Args:
            k_loop (int): number of MPC iterations since the start of the simulation
            velID (int): Identifier of the current velocity profile to be able to handle different scenarios
        """

        # Moving forwards
        """if k_loop == self.k_mpc*16*3:
            self.v_ref = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T"""

        beta_x = int(max(abs(self.Vx_ref)*10000, 100.0))
        alpha_x = np.max([np.min([(k_loop-self.k_mpc*16*3)/beta_x, 1.0]), 0.0])

        beta_y = int(max(abs(self.Vy_ref)*10000, 100.0))
        alpha_y = np.max([np.min([(k_loop-self.k_mpc*16*3)/beta_y, 1.0]), 0.0])

        beta_w = int(max(abs(self.Vw_ref)*2500, 100.0))
        alpha_w = np.max([np.min([(k_loop-self.k_mpc*16*3)/beta_w, 1.0]), 0.0])

        # self.v_ref = np.array([[0.3*alpha, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
        self.v_ref = np.array(
            [[self.Vx_ref*alpha_x, self.Vy_ref*alpha_y, 0.0, 0.0, 0.0, self.Vw_ref*alpha_w]]).T

        return 0
