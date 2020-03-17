# coding: utf8

import numpy as np


class FootstepPlanner:
    """A footstep planner that handles the choice of future
    footsteps location depending on the current and reference
    velocities of the quadruped.

    :param dt: A float, time step of the contact sequence
    """

    def __init__(self, dt):

        # Feedback gain for the feedback term of the planner
        self.k_feedback = 0.03

        # Position of shoulders in local frame
        self.shoulders = np.array(
            [[0.19, 0.19, -0.19, -0.19], [0.15005, -0.15005, 0.15005, -0.15005]])

        # Time step of the contact sequence
        self.dt = dt

        # Value of the gravity acceleartion
        self.g = 9.81

        # Value of the maximum allowed deviation due to leg length
        self.L = 0.06

        # The desired (x,y) position of footsteps
        # If a foot is in swing phase it is where it should land
        # If a foot is in stance phase is is where it should land at the end of its next swing phase
        R = np.array([[0.0, -1.0], [1.0, 0.0]])
        self.footsteps = R @ self.shoulders.copy()

        # Previous variable but in world frame for visualisation purpose
        self.footsteps_world = self.footsteps.copy()

        # To store the result of the get_prediction function
        self.footsteps_prediction = np.zeros((3, 4))

        # To store the result of the update_footsteps_tsid function
        self.footsteps_tsid = np.zeros((3, 4))
        self.t_remaining_tsid = np.zeros((1, 4))

    def update_footsteps_tsid(self, sequencer, vel_ref, v_xy, t_stance, T, h):
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

        # self.footsteps_tsid = np.zeros((3, 4))

        # Shift initial position of contact outwards for more stability
        # p[1, :] += np.array([0.025, -0.025, 0.025, -0.025])

        # Add symmetry term
        self.footsteps_tsid[0:2, :] = t_stance * 0.5 * np.tile(v_xy, 4)

        # Add feedback term
        self.footsteps_tsid[0:2, :] += self.k_feedback * (v_xy - vel_ref[0:2, 0:1])

        # Add centrifugal term
        # cross = np.cross(vel_cur[0:3, 0:1], vel_ref[3:6, 0:1], 0, 0).T
        self.footsteps_tsid[0:2, :] += 0.5 * np.sqrt(h/self.g) * np.array([[v_xy[1, 0] * vel_ref[5, 0]],
                                                                           [- v_xy[0, 0] * vel_ref[5, 0]]])

        # Time remaining before the end of the currrent swing phase
        self.t_remaining_tsid = np.zeros((1, 4))
        for i in range(4):
            # indexes_stance = (np.where(sequencer.S[:, i] == True))[0]
            # indexes_swing = (np.where(sequencer.S[:, i] == False))[0]
            # index = (np.where(S[:, i] == True))[0][0]
            if (sequencer.S[0, i] == 1.0) and (sequencer.S[-1, i] == 0.0):
                self.t_remaining_tsid[0, i] = sequencer.T_gait
            else:
                index = next((idx for idx, val in np.ndenumerate(sequencer.S[:, i]) if val==1.0), 0.0)[0]
                self.t_remaining_tsid[0, i] = index * self.dt

        # Add velocity forecast
        if vel_ref[5, 0] != 0:
            self.footsteps_tsid[0, :] += (v_xy[0, 0] * np.sin(vel_ref[5, 0] * self.t_remaining_tsid[0, :]) +
                                          v_xy[1, 0] * (np.cos(vel_ref[5, 0] * self.t_remaining_tsid[0, :]) - 1)) / vel_ref[5, 0]
            self.footsteps_tsid[1, :] += (v_xy[1, 0] * np.sin(vel_ref[5, 0] * self.t_remaining_tsid[0, :]) -
                                          v_xy[0, 0] * (np.cos(vel_ref[5, 0] * self.t_remaining_tsid[0, :]) - 1)) / vel_ref[5, 0]
        else:
            self.footsteps_tsid[0, :] += v_xy[0, 0] * self.t_remaining_tsid[0, :]
            self.footsteps_tsid[1, :] += v_xy[1, 0] * self.t_remaining_tsid[0, :]

        # Legs have a limited length so the deviation has to be limited
        self.footsteps_tsid[0:2, :] = np.clip(self.footsteps_tsid[0:2, :], -self.L, self.L)

        # Update target_footholds_no_lock
        # self.footsteps_tsid = p  # np.tile(p, (1, 4))

        return 0

    def update_footsteps_mpc(self, sequencer, mpc, mpc_interface):
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

        # Initial deviation
        p = np.zeros((2, 4))

        # Shift initial position of contact outwards for more stability
        # p[1, :] += np.array([0.025, -0.025, 0.025, -0.025])

        # Add symmetry term
        p += sequencer.t_stance * 0.5 * mpc.v[0:2, 0:1]

        # Add feedback term
        p += self.k_feedback * (mpc.v[0:2, 0:1] - mpc.v_ref[0:2, 0:1])

        # Add centrifugal term
        cross = np.cross(mpc.v[0:3, 0:1], mpc.v_ref[3:6, 0:1], 0, 0).T
        p += 0.5 * np.sqrt(mpc.q[2, 0]/self.g) * cross[0:2, 0:1]

        # Time remaining before the end of the currrent swing phase
        t_remaining = np.zeros((1, 4))
        for i in range(4):
            indexes_stance = (np.where(sequencer.S[:, i] == True))[0]
            indexes_swing = (np.where(sequencer.S[:, i] == False))[0]
            # index = (np.where(S[:, i] == True))[0][0]
            if (sequencer.S[0, i] == True) and (sequencer.S[-1, i] == False):
                t_remaining[0, i] = sequencer.T_gait
            else:
                index = (indexes_stance[indexes_stance > indexes_swing[0]])[0]
                t_remaining[0, i] = index * self.dt

        # Add velocity forecast
        if mpc.v_ref[5, 0] != 0:
            p[0, :] += (mpc.v[0, 0] * np.sin(mpc.v_ref[5, 0] * t_remaining[0, :]) +
                        mpc.v[1, 0] * (np.cos(mpc.v_ref[5, 0] * t_remaining[0, :]) - 1)) / mpc.v_ref[5, 0]
            p[1, :] += (mpc.v[1, 0] * np.sin(mpc.v_ref[5, 0] * t_remaining[0, :]) -
                        mpc.v[0, 0] * (np.cos(mpc.v_ref[5, 0] * t_remaining[0, :]) - 1)) / mpc.v_ref[5, 0]       
        else:
            p[0, :] += mpc.v[0, 0] * t_remaining[0, :]
            p[1, :] += mpc.v[1, 0] * t_remaining[0, :]

        # Legs have a limited length so the deviation has to be limited
        p[0:2, :] = np.clip(p[0:2, :], -self.L, self.L)

        # Add shoulders
        p[0:2, :] += self.shoulders

        # Update target_footholds_no_lock
        self.footsteps = mpc_interface.l_feet[0:2, :].copy()
        for i in np.where(sequencer.S[0, :] == False)[0]:
            self.footsteps[:, i] = p[:, i]

        # Updating quantities expressed in world frame
        self.update_world_frame(mpc.q_w)

        return 0

    def get_prediction(self, S, t_stance, T_gait, q, v, v_ref):

        # Order of feet: FL, FR, HL, HR

        p = np.zeros((3, 4))

        # Add symmetry term
        p[0:2, :] += t_stance * 0.5 * v[0:2, 0:1]

        # Add feedback term
        p[0:2, :] += self.k_feedback * (v[0:2, 0:1] - v_ref[0:2, 0:1])

        # Add centrifugal term
        cross = np.cross(v[0:3, 0:1], v_ref[3:6, 0:1], 0, 0).T
        p[0:2, :] += 0.5 * np.sqrt(q[2, 0]/self.g) * cross[0:2, 0:1]

        # Time remaining before the end of the currrent swing phase
        t_remaining = np.zeros((1, 4))
        for i in range(4):
            indexes_stance = (np.where(S[:, i] == True))[0]
            indexes_swing = (np.where(S[:, i] == False))[0]
            if (S[0, i] == True) and (S[-1, i] == False):
                t_remaining[0, i] = T_gait
            else:
                index = (indexes_stance[indexes_stance > indexes_swing[0]])[0]
                t_remaining[0, i] = index * self.dt

        # Add velocity forecast
        if v_ref[5, 0] != 0:
            p[0, :] += (v[0, 0] * np.sin(v_ref[5, 0] * t_remaining[0, :]) +
                        v[1, 0] * (np.cos(v_ref[5, 0] * t_remaining[0, :]) - 1)) / v_ref[5, 0]
            p[1, :] += (v[1, 0] * np.sin(v_ref[5, 0] * t_remaining[0, :]) -
                        v[0, 0] * (np.cos(v_ref[5, 0] * t_remaining[0, :]) - 1)) / v_ref[5, 0]       
        else:
            p[0, :] += v[0, 0] * t_remaining[0, :]
            p[1, :] += v[1, 0] * t_remaining[0, :]

        # Legs have a limited length so the deviation has to be limited
        p[0:2, :] = np.clip(p[0:2, :], -self.L, self.L)

        # Add shoulders
        p[0:2, :] += self.shoulders

        self.footsteps_prediction = p

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
