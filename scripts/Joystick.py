# coding: utf8

import numpy as np
import gamepadClient as gC
import libquadruped_reactive_walking as lqrw

class Joystick:
    """Joystick-like controller that outputs the reference velocity in local frame

    Args:
        predefined (bool): use either a predefined velocity profile (True) or a gamepad (False)
    """

    def __init__(self, params, multi_simu=False):

        # Controller parameters
        self.dt_wbc = params.dt_wbc
        self.dt_mpc = params.dt_mpc

        # Reference velocity in local frame
        self.v_ref = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
        self.v_gp = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T

        self.reduced = False
        self.stop = False

        self.alpha = 0.0005  # Coefficient to low pass the joystick velocity

        # Bool to modify the update of v_ref
        # Used to launch multiple simulations
        self.multi_simu = multi_simu

        # If we are using a predefined reference velocity (True) or a joystick (False)
        self.predefined = params.predefined_vel

        # If we are performing an analysis from outside
        self.analysis = False

        # Joystick variables (linear and angular velocity and their scaling for the joystick)
        self.vX = 0.
        self.vY = 0.
        self.vYaw = 0.
        self.VxScale = 0.5
        self.VyScale = 0.8
        self.vYawScale = 0.8

        self.Vx_ref = 0.3
        self.Vy_ref = 0.0
        self.Vw_ref = 0.0

        # Y, B, A and X buttons (in that order)
        self.northButton = False
        self.eastButton = False
        self.southButton = False
        self.westButton = False
        self.joystick_code = 0  # Code to carry information about pressed buttons

        self.joyCpp = lqrw.Joystick()

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
            elif self.analysis:
                self.handle_v_switch(k_loop)
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
        self.v_gp[(self.v_gp < 0.004) & (self.v_gp > -0.004)] = 0.0
        self.v_ref = self.alpha * self.v_gp + (1-self.alpha) * self.v_ref

        # Update joystick code depending on which buttons are pressed
        self.computeCode()

        return 0

    def computeCode(self):
        # Check joystick buttons to trigger a change of gait type
        self.joystick_code = 0
        if self.northButton:
            self.joystick_code = 1
            self.northButton = False
        elif self.eastButton:
            self.joystick_code = 2
            self.eastButton = False
        elif self.southButton:
            self.joystick_code = 3
            self.southButton = False
        elif self.westButton:
            self.joystick_code = 4
            self.westButton = False

    def handle_v_switch(self, k):
        """Handle the change of reference velocity according to the chosen predefined velocity profile

        Args:
            k (int): numero of the current iteration
        """

        self.v_ref[:, 0] = self.joyCpp.handle_v_switch(k, self.k_switch.reshape((-1, 1)), self.v_switch)

    def update_v_ref_predefined(self, k_loop, velID):
        """Update the reference velocity of the robot along X, Y and Yaw in local frame
        according to a predefined sequence

        Args:
            k_loop (int): numero of the current iteration
            velID (int): identifier of the current velocity profile to be able to handle different scenarios
        """

        if (k_loop == 0):
            if velID == 0:
                self.t_switch = np.array([0, 1, 4, 6, 8, 26, 40, 60])
                self.v_switch = np.array([[0.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0],
                                         [0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                         [0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                         [0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                         [0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                         [0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
            elif velID == 1:
                V_max = 1.0
                R_max = 0.3
                self.t_switch = np.array([0, 2, 6, 16, 24, 32, 40, 44,
                                          46, 52, 60, 66, 68, 80, 82, 86,
                                          88, 90])
                self.v_switch = np.zeros((6, self.t_switch.shape[0]))
                self.v_switch[0, :] = np.array([0.0, 0.0, V_max, V_max, 0.0, 0.0, 0.0,
                                                0.0, -V_max, -V_max, 0.0, 0.0, 0.0, V_max, V_max, V_max,
                                                V_max, V_max])
                self.v_switch[1, :] = np.array([0.0, 0.0,  0.0, 0.0, -V_max*0.5, -V_max*0.5, 0.0,
                                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                0.0, 0.0])
                self.v_switch[5, :] = np.array([0.0, 0.0,  R_max, R_max, R_max, 0.0, 0.0, 0.0,
                                                0.0, 0.0, 0.0, 0.0, R_max, R_max, 0.0, 0.0,
                                                -R_max, 0.0])
            elif velID == 2:
                self.t_switch = np.array([0, 5, 8, 10, 12])
                self.v_switch = np.array([[0.0, 0.8, 0.8, 0.8, 0.8  ],
                                         [0.0, 0.0, 0.0, 0.0, 0.0],
                                         [0.0, 0.0, 0.0, 0.0, 0.0],
                                         [0.0, 0.0, 0.0, 0.0, 0.0],
                                         [0.0, 0.0, 0.0, 0.0, 0.0],
                                         [0.0, 0., 1., 0., -1. ]])
            elif velID == 3:
                self.t_switch = np.array([0, 2, 6, 8, 12, 60])
                self.v_switch = np.array([[0.0, 0.0,  0.4, 0.4, 0.0, 0.0],
                                         [0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
                                         [0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
                                         [0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
                                         [0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
                                         [0.0, 0.0,  0.0, 0.0, 0.0, 0.0]])
            elif velID == 4:
                self.t_switch = np.array([0, 2, 6, 14, 18, 60])
                self.v_switch = np.array([[0.0, 0.0,  1.5, 1.5, 1.5, 1.5],
                                         [0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
                                         [0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
                                         [0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
                                         [0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
                                         [0.0, 0.0,  0.0, 0.0, 0.4, 0.4]])
            elif velID == 5:
                self.t_switch = np.array([0, 1, 3, 5.2, 10, 13, 14, 16, 18])
                self.v_switch = np.array([[0.0, 0.0,  0.5, 0.6, 0.3, 0.6, -0.5, 0.7, 0.0],
                                         [0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                         [0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                         [0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                         [0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                         [0.0, 0.0,  0.2, 0.7, 0.7, 0.0, -0.4, -0.6, 0.0]])
            elif velID == 6:
                self.t_switch = np.array([0, 2, 5, 10, 15, 16, 20])
                self.v_switch = np.array([[0.0, 0.0,  0.8, 0.4, 0.8, 0.8, 0.0],
                                         [0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0],
                                         [0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0],
                                         [0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0],
                                         [0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0],
                                         [0.0, 0.0,  0.0, 0.55, 0.3, 0.0, 0.0]])

        self.k_switch = (self.t_switch / self.dt_wbc).astype(int)
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

    def update_for_analysis(self, des_vel_analysis, N_analysis, N_steady):

        self.analysis = True

        self.k_switch = np.array([0, int(1/self.dt_wbc), N_analysis, N_analysis + N_steady])
        self.v_switch = np.zeros((6, 4))
        self.v_switch[:, 2] = des_vel_analysis
        self.v_switch[:, 3] = des_vel_analysis

        return 0

if __name__ == "__main__":

    from matplotlib import pyplot as plt
    import libquadruped_reactive_walking as lqrw
    from time import clock
    params = lqrw.Params()  # Object that holds all controller parameters
    params.predefined_vel = False
    joystick = Joystick(params)
    k = 0
    vx = [0.0] * 1000
    fig = plt.figure()
    ax = plt.gca()
    ax.set_ylim([-0.5, 0.5])
    h, = plt.plot(np.linspace(0.001, 1.0, 1000), vx, "b", linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Forward reference velocity [m/s]")
    plt.show(block=False)
    
    print("Start")
    while True:
        # Update the reference velocity coming from the gamepad
        joystick.update_v_ref(k, 0)
        vx.pop(0)
        vx.append(joystick.v_ref[0, 0])

        if k % 50 == 0:
            h.set_ydata(vx)
            print("Joystick raw:      ", joystick.v_gp[0, 0])
            print("Joystick filtered: ", joystick.v_ref[0, 0])
            plt.pause(0.0001)

        k += 1