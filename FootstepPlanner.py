# coding: utf8

import numpy as np


class FootstepPlanner:
    """A footstep planner that handles the choice of future
    footsteps location depending on the current and reference
    velocities of the quadruped.

    :param k_feedback: A float, the gain for the feedback term of the planner
    :param shoulders: A 2 by 4 numpy array, the position of shoulders in local frame
    :param dt: A float, time step of the contact sequence
    """

    def __init__(self, k_feedback, shoulders, dt):

        # Feedback gain for the feedback term of the planner
        self.k_feedback = k_feedback

        # Position of shoulders in local frame
        self.shoulders = shoulders

        # Time step of the contact sequence
        self.dt = dt

        # Value of the gravity acceleartion
        self.g = 9.81

        # The desired (x,y) position of footsteps
        # If a foot is in swing phase it is where it should land
        # If a foot is in stance phase is is where it should land at the end of its next swing phase
        self.footsteps = self.shoulders.copy()

        # Previous variable but in world frame for visualisation purpose
        self.footsteps_world = self.footsteps.copy()

    def update_footsteps(self, vel_ref, vel_cur, t_stance, t_remaining, T, h):
        """Returns a 2 by 4 matrix containing the [x, y]^T position of the next desired footholds for the four feet
        For feet in a swing phase it is where they should land and for feet currently touching the ground it is
        where they should land at the end of their next swing phase

        Keyword arguments:
        vel_ref -- reference velocity vector of the flying base (6 by 1, linear and angular stacked)
        vel_cur -- current velocity vector of the flying base (6 by 1, linear and angular stacked)
        t_stance -- duration of the stance phase
        t_remaining -- time remaining before the end of the currrent swing phase
        T -- period of the current gait
        """

        # Order of feet: FL, FR, HL, HR

        p = np.zeros((2, 1))

        # Shift initial position of contact outwards for more stability
        # p[1, :] += np.array([0.025, -0.025, 0.025, -0.025])

        # Add symmetry term
        p += t_stance * 0.5 * vel_cur[0:2, 0:1]

        # Add feedback term
        p += self.k_feedback * (vel_cur[0:2, 0:1] - vel_ref[0:2, 0:1])

        # Add centrifugal term
        cross = np.cross(vel_cur[0:3, 0:1], vel_ref[3:6, 0:1], 0, 0).T
        p += 0.5 * np.sqrt(h/self.g) * cross[0:2, 0:1]

        # Add velocity forecast
        #  p += np.tile(v[0:2, 0:1], (1, 4)) * t_remaining
        """for i in range(4):
            yaw = np.linspace(0, t_remaining[0, i]-self.dt, int(np.floor(t_remaining[0, i]/self.dt))) * vel_cur[5, 0]
            p[0, i] += (self.dt * np.cumsum(vel_cur[0, 0] * np.cos(yaw) - vel_cur[1, 0] * np.sin(yaw)))[-1]
            p[1, i] += (self.dt * np.cumsum(vel_cur[0, 0] * np.sin(yaw) + vel_cur[1, 0] * np.cos(yaw)))[-1]"""
        """for i in range(4):
            p[0, i] += t_remaining[0, i] * vel_cur[0, 0]
            p[1, i] += t_remaining[0, i] * vel_cur[1, 0]"""

        # Update target_footholds_no_lock
        self.footsteps = np.tile(p, (1, 4))

        return 0

    def update_footsteps_mpc(self, vel_ref, vel_cur, t_stance, S, T, h):
        """Returns a 2 by 4 matrix containing the [x, y]^T position of the next desired footholds for the four feet
        For feet in a swing phase it is where they should land and for feet currently touching the ground it is
        where they should land at the end of their next swing phase

        Keyword arguments:
        vel_ref -- reference velocity vector of the flying base (6 by 1, linear and angular stacked)
        vel_cur -- current velocity vector of the flying base (6 by 1, linear and angular stacked)
        t_stance -- duration of the stance phase
        S -- contact sequence that defines the current gait
        T -- period of the current gait
        """

        # Order of feet: FL, FR, HL, HR

        # Start with shoulder term
        p = np.tile(np.array([[0], [0]]), (1, 4)) + self.shoulders  # + np.dot(R, shoulders)

        # Shift initial position of contact outwards for more stability
        p[1, :] += np.array([0.025, -0.025, 0.025, -0.025])

        # Add symmetry term
        p += t_stance * 0.5 * vel_cur[0:2, 0:1]

        # Add feedback term
        p += self.k_feedback * (vel_cur[0:2, 0:1] - vel_ref[0:2, 0:1])

        # Add centrifugal term
        cross = np.cross(vel_cur[0:3, 0:1], vel_ref[3:6, 0:1], 0, 0).T
        p += 0.5 * np.sqrt(h/self.g) * cross[0:2, 0:1]

        # Time remaining before the end of the currrent swing phase
        t_remaining = np.zeros((1, 4))
        for i in range(4):
            indexes_stance = (np.where(S[:, i] == True))[0]
            indexes_swing = (np.where(S[:, i] == False))[0]
            # index = (np.where(S[:, i] == True))[0][0]
            if (S[0, i] == True) and (S[-1, i] == False):
                t_remaining[0, i] = T
            else:
                index = (indexes_stance[indexes_stance > indexes_swing[0]])[0]
                t_remaining[0, i] = index * self.dt

        # Add velocity forecast
        #  p += np.tile(v[0:2, 0:1], (1, 4)) * t_remaining
        for i in range(4):
            yaw = np.linspace(0, t_remaining[0, i]-self.dt, np.floor(t_remaining[0, i]/self.dt)) * vel_cur[5, 0]
            p[0, i] += (self.dt * np.cumsum(vel_cur[0, 0] * np.cos(yaw) - vel_cur[1, 0] * np.sin(yaw)))[-1]
            p[1, i] += (self.dt * np.cumsum(vel_cur[0, 0] * np.sin(yaw) + vel_cur[1, 0] * np.cos(yaw)))[-1]

        # Update target_footholds_no_lock
        self.footsteps = p

        return 0

    def update_world_frame(self, q_w):
        """Update quantities expressed in the world frame

        Keyword arguments:
        :param q_w: Position vector of the quadruped in the world frame (6 by 1)
        """

        c, s = np.cos(q_w[5, 0]), np.sin(q_w[5, 0])
        self.footsteps_world[0, :] = q_w[0, 0] \
            + c * self.footsteps[0, :] - s * self.footsteps[1, :]
        self.footsteps_world[1, :] = q_w[1, 0] \
            + s * self.footsteps[0, :] + c * self.footsteps[1, :]

        return 0

    def update_viewer(self, viewer, initialisation):
        """Update display for visualization purpose

        Keyword arguments:
        :param viewer: A gepetto viewer object
        :param initialisation: A bool, is it the first iteration of the main loop
        """

        # Display non-locked target footholds with green spheres (gepetto gui)
        rgbt = [0.0, 1.0, 0.0, 0.5]
        for i in range(4):
            if initialisation:
                viewer.gui.addSphere("world/sphere"+str(i)+"_nolock", .02, rgbt)  # .1 is the radius
            viewer.gui.applyConfiguration(
                "world/sphere"+str(i)+"_nolock", (self.footsteps_world[0, i],
                                                  self.footsteps_world[1, i], 0.0, 1., 0., 0., 0.))

        return 0
