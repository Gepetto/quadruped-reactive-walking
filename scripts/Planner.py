import time
import numpy as np
import libquadruped_reactive_walking as la


class PyPlanner:
    def __init__(self, dt, dt_tsid, T_gait, T_mpc, k_mpc, on_solo8, h_ref, fsteps_init):
        # Reference height for the trunk
        self.h_ref = h_ref

        # Number of time steps in the prediction horizon
        self.n_steps = np.int(T_gait/dt)

        # Reference trajectory matrix of size 12 by (1 + N)  with the current state of
        # the robot in column 0 and the N steps of the prediction horizon in the others
        self.xref = np.zeros((12, 1 + self.n_steps))

        # Gait matrix
        self.gait = np.zeros((20, 5))
        self.is_static = False  # Flag for static gait
        self.q_static = np.zeros(19)
        self.RPY_static = np.zeros((3, 1))

        self.goals = fsteps_init.copy()  # Store 3D target position for feet
        self.vgoals = np.zeros((3, 4))  # Store 3D target velocity for feet
        self.agoals = np.zeros((3, 4))  # Store 3D target acceleration for feet
        self.target_position = np.zeros((3, 4))  # Store 3D target acceleration for feet

        # C++ class
        shoulders = np.zeros((3, 4))
        shoulders[0, :] = [0.1946, 0.1946, -0.1946, -0.1946]
        shoulders[1, :] = [0.14695, -0.14695, 0.14695, -0.14695]
        self.planner = la.Planner(dt, dt_tsid, T_gait, T_mpc, k_mpc, h_ref, fsteps_init, shoulders)

    def run_planner(self, k, k_mpc, q, v, b_vref, h_estim, z_average, joystick=None):
        joystick_code = 0
        if joystick is not None:
            if joystick.northButton:
                joystick_code = 1
                self.is_static = False
                joystick.northButton = False
            elif joystick.eastButton:
                joystick_code = 2
                self.is_static = False
                joystick.eastButton = False
            elif joystick.southButton:
                joystick_code = 3
                self.is_static = False
                joystick.southButton = False
            elif joystick.westButton:
                joystick_code = 4
                self.is_static = True
                self.q_static[0:7, 0:1] = q.copy()
                joystick.westButton = False

        self.planner.run_planner(k, q, v, b_vref, np.double(z_average), joystick_code)

        self.xref = self.planner.get_xref()
        self.fsteps = self.planner.get_fsteps()
        self.gait = self.planner.get_gait()
        self.goals = self.planner.get_goals()
        self.vgoals = self.planner.get_vgoals()
        self.agoals = self.planner.get_agoals()
        
        return 0
