# coding: utf8

import numpy as np
import gamepadClient as gC


class Joystick:
    """Joystick-like controller that outputs the reference velocity in local frame
    """

    def __init__(self, k_mpc, predefined, multi_simu=False):

        # Number of TSID steps for 1 step of the MPC
        self.k_mpc = k_mpc

        # Reference velocity in local frame
        self.v_ref = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T

        self.reduced = False

        # Bool to modify the update of v_ref
        # Used to launch multiple simulations
        self.multi_simu = multi_simu

        # If we are using a predefined reference velocity (True) or a joystick (False)
        self.predefined = predefined

        # Joystick variables (linear and angular velocity and their scaling for the joystick)
        self.vX = 0.
        self.vY = 0.
        self.vYaw = 0.
        self.VxScale = 0.3
        self.VyScale = 0.3
        self.vYawScale = 0.8

        self.Vx_ref = 0.3
        self.Vy_ref = 0.0
        self.Vw_ref = 0.0

    def update_v_ref(self, k_loop, velID):
        """Update the reference velocity of the robot along X, Y and Yaw in local frame by
        listening to a gamepad handled by an independent thread

        Args:
            k_loop (int): number of MPC iterations since the start of the simulation
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
            k_loop (int): number of MPC iterations since the start of the simulation
        """

        if k_loop == 0:
            self.gp = gC.GamepadClient()

        self.vX = self.gp.leftJoystickX.value * self.VxScale
        self.vY = self.gp.leftJoystickY.value * self.VyScale
        self.vYaw = self.gp.rightJoystickX.value * self.vYawScale

        if self.gp.L1Button.value:
            self.v_gp = np.array([[0.0, 0.0, - self.vYaw * 0.25, - self.vX * 5, - self.vY * 2, 0.0]]).T
        else:
            self.v_gp = np.array([[- self.vY, - self.vX, 0.0, 0.0, 0.0, - self.vYaw]]).T

        if self.gp.startButton.value == True:
            self.reduced = not self.reduced

        tc = 0.04  #  cutoff frequency at 50 Hz
        dT = 0.002  # velocity reference is updated every ms
        alpha = dT / tc
        self.v_ref = alpha * self.v_gp + (1-alpha) * self.v_ref

        return 0

    def handle_v_switch(self, k):

        i = 1
        while (i < self.k_switch.shape[0]) and (self.k_switch[i] <= k):
            i += 1
        if (i != self.k_switch.shape[0]):
            self.apply_velocity_change(k, i)

    def apply_velocity_change(self, k, i):
        """Change the velocity reference sent to the robot
        4-th order polynomial: zero force and force velocity at start and end
        (bell-like force trajectory)
        """

        ev = k - self.k_switch[i-1]
        t1 = self.k_switch[i] - self.k_switch[i-1]
        A3 = 2 * (self.v_switch[:, (i-1):i] - self.v_switch[:, i:(i+1)]) / t1**3
        A2 = (-3/2) * t1 * A3
        self.v_ref = self.v_switch[:, (i-1):i] + A2*ev**2 + A3*ev**3

        return 0

    def update_v_ref_predefined(self, k_loop, velID):
        """Update the reference velocity of the robot along X, Y and Yaw in local frame
        according to a predefined sequence

        Args:
            k_loop (int): number of MPC iterations since the start of the simulation
            velID (int): Identifier of the current velocity profile to be able to handle different scenarios
        """

        # Moving forwards
        """if k_loop == self.k_mpc*16*3:
            self.v_ref = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T"""

        if velID == 0:
            if (k_loop == 0):
                self.k_switch = np.array([0, 1000, 4000, 7000, 10000, 13000])
                self.v_switch = np.array([[0.0, 0.0, 0.25, 0.45, 0.65, 0.7],
                                          [0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0,  0.0, 0.0, 0.0, 0.0]])
            self.handle_v_switch(k_loop)

        # Video Demo 16/06/2020
        """V_max = 0.3
        Rot_max = 0.2
        if k_loop < 4000:
            alpha = np.max([np.min([(k_loop-self.k_mpc*16*2)/3000, 1.0]), 0.0])
            self.v_ref = np.array([[V_max*alpha, 0.0, 0.0, 0.0, 0.0, 0.0]]).T

        elif k_loop < 5000:
            alpha = np.max([np.min([(k_loop-4000)/500, 1.0]), 0.0])
            self.v_ref = np.array([[V_max, 0.0, 0.0, 0.0, 0.0, -Rot_max*alpha]]).T

        elif k_loop < 6000:
            alpha = np.max([np.min([(k_loop-5000)/500, 1.0]), 0.0])
            self.v_ref = np.array([[V_max, 0.0, 0.0, 0.0, 0.0, -Rot_max*(1.0-alpha)]]).T

        elif k_loop < 8000:
            alpha = np.max([np.min([(k_loop-6000)/2000, 1.0]), 0.0])
            self.v_ref = np.array([[V_max*(1-alpha), V_max*alpha, 0.0, 0.0, 0.0, 0.0]]).T

        else:
            alpha = np.max([np.min([(k_loop-8000)/1000, 1.0]), 0.0])
            self.v_ref = np.array([[0.0, V_max*(1.0-alpha), 0.0, 0.0, 0.0, 0.0]]).T"""
        # End Video Demo 16/06/2020

        if velID == 1:
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
                self.v_switch[5, :] = np.array([0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.0, 0.0,
                                                -0.3, 0.0])
            self.handle_v_switch(k_loop)

        # Video Demo 24/06/2020
        if velID == 2:
            V_max = 0.3
            Rot_max = 0.2
            if k_loop < 8000:
                alpha = np.max([np.min([(k_loop-self.k_mpc*16*2)/2000, 1.0]), 0.0])
                self.v_ref = np.array([[V_max*alpha, 0.0, 0.0, 0.0, 0.0, 0.0]]).T

            elif k_loop < 12000:
                alpha = np.max([np.min([(k_loop-8000)/4000, 1.0]), 0.0])
                self.v_ref = np.array([[V_max*(1-alpha), -V_max*alpha, 0.0, 0.0, 0.0, 0.0]]).T

            elif k_loop < 20000:
                alpha = np.max([np.min([(k_loop-12000)/7000, 1.0]), 0.0])
                self.v_ref = np.array([[0.0, -V_max*(1-alpha), 0.0, 0.0, 0.0, 0]]).T

            elif k_loop < 22000:
                self.v_ref = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T

            elif k_loop < 26000:
                alpha = np.max([np.min([(k_loop-22000)/1000, 1.0]), 0.0])
                self.v_ref = np.array([[-V_max*alpha, 0.0, 0.0, 0.0, 0.0, 0.0]]).T

            elif k_loop < 33000:
                alpha = np.max([np.min([(k_loop-26000)/4000, 1.0]), 0.0])
                self.v_ref = np.array([[-V_max*(1-alpha), 0.0, 0.0, 0.0, 0.0, 0.0]]).T

            elif k_loop < 36000:
                alpha = np.max([np.min([(k_loop-33000)/1000, 1.0]), 0.0])
                self.v_ref = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, +0.3*alpha]]).T

            elif k_loop < 40000:
                alpha = np.max([np.min([(k_loop-36000)/1000, 1.0]), 0.0])
                self.v_ref = np.array([[V_max*alpha, 0.0, 0.0, 0.0, 0.0, +0.3]]).T

            elif k_loop < 41000:
                alpha = np.max([np.min([(k_loop-40000)/1000, 1.0]), 0.0])
                self.v_ref = np.array([[V_max, 0.0, 0.0, 0.0, 0.0, +0.3*(1-alpha)]]).T

            elif k_loop < 43000:
                self.v_ref = np.array([[V_max, 0.0, 0.0, 0.0, 0.0, 0.0]]).T

            elif k_loop < 44000:
                alpha = np.max([np.min([(k_loop-43000)/500, 1.0]), 0.0])
                self.v_ref = np.array([[V_max, 0.0, 0.0, 0.0, 0.0, -0.3*alpha]]).T

            else:
                alpha = np.max([np.min([(k_loop-44000)/1450, 1.0]), 0.0])
                self.v_ref = np.array([[V_max, 0.0, 0.0, 0.0, 0.0, -0.3*(1-alpha)]]).T
        # End Video Demo 24/06/2020

        """if k_loop < 8000:
            alpha = np.max([np.min([(k_loop-self.k_mpc*16*2)/2500, 1.0]), 0.0])
            self.v_ref = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, +0.2*alpha]]).T
        elif k_loop < 13000:
            alpha = np.max([np.min([(k_loop-8000)/5000, 1.0]), 0.0])
            self.v_ref = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, +0.2*(1-alpha)]]).T
        else:
            alpha = np.max([np.min([(k_loop-13000)/2000, 1.0]), 0.0])
            self.v_ref = np.array([[V_max*alpha, 0.0, 0.0, 0.0, 0.0, 0.0]]).T"""

        """if k_loop == self.k_mpc*16*6:
            self.v_ref = np.array([[0.3, 0.0, 0.0, 0.0, 0.0, 0.0]]).T

        if k_loop == self.k_mpc*16*8:
            self.v_ref = np.array([[0.6, 0.0, 0.0, 0.0, 0.0, 0.0]]).T

        if k_loop == self.k_mpc*16*11+20*8:
            self.v_ref = np.array([[0.9, 0.0, 0.0, 0.0, 0.0, 0.0]]).T

        if k_loop == self.k_mpc*16*12:
            self.v_ref = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T

        if k_loop == self.k_mpc*16*14:
            self.v_ref = np.array([[1.2, 0.0, 0.0, 0.0, 0.0, 0.0]]).T

        if k_loop == self.k_mpc*16*15:
            self.v_ref = np.array([[1.3, 0.0, 0.0, 0.0, 0.0, 0.0]]).T

        if k_loop == self.k_mpc*16*16:
            self.v_ref = np.array([[1.4, 0.0, 0.0, 0.0, 0.0, 0.0]]).T

        if k_loop == self.k_mpc*16*17:
            self.v_ref = np.array([[1.5, 0.0, 0.0, 0.0, 0.0, 0.0]]).T

        if k_loop == self.k_mpc*16*18:
            self.v_ref = np.array([[1.7, 0.0, 0.0, 0.0, 0.0, 0.0]]).T

        if k_loop == self.k_mpc*16*19:
            self.v_ref = np.array([[1.9, 0.0, 0.0, 0.0, 0.0, 0.0]]).T"""

        """# Turning
        if k_loop == 4000:
            self.v_ref = np.array([[0.3, 0.0, 0.0, 0.0, 0.0, 0.0]]).T

        # Turning
        if k_loop == 6000:
            self.v_ref = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T"""

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
        self.v_ref = np.array([[self.Vx_ref*alpha_x, self.Vy_ref*alpha_y, 0.0, 0.0, 0.0, self.Vw_ref*alpha_w]]).T

        return 0
