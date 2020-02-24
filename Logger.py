# coding: utf8

import numpy as np
from matplotlib import pyplot as plt


class Logger:
    """Joystick-like controller that outputs the reference velocity in local frame
    """

    def __init__(self, k_max_loop):

        # Max number of iterations of the main loop
        self.k_max_loop = k_max_loop

        # Log state vector and reference state vector
        self.log_state = np.zeros((12, k_max_loop))
        self.log_state_ref = np.zeros((12, k_max_loop))

        # Log footholds position in world frame
        self.log_footholds = np.zeros((4, k_max_loop, 2))
        self.log_footholds_w = np.zeros((4, k_max_loop, 2))

    def log_state_vectors(self, mpc, k_loop):
        """ Log current and reference state vectors (position + velocity)
        """

        self.log_state[:, k_loop:(k_loop+1)] = np.vstack((mpc.q_w, mpc.v))
        self.log_state_ref[:, k_loop:(k_loop+1)] = mpc.xref[:, 1:2]

        return 0

    def log_various_footholds(self, mpc, k_loop):
        """ Log current and reference state vectors (position + velocity)
        """

        self.log_footholds[:, k_loop, :] = mpc.footholds[0:2, :].transpose()
        self.log_footholds_w[:, k_loop, :] = mpc.footholds_world[0:2, :].transpose()

        return 0

    def call_log_functions(self, fstep_planner, ftraj_gen, mpc, k_loop):
        """ Call logging functions of the Logger class
        """

        # Logging reference and current state vectors
        self.log_state_vectors(mpc, k_loop)

        # Log footholds
        self.log_various_footholds(mpc, k_loop)

        return 0

    def plot_graphs(self, dt, k_max_loop):

        log_t = [k*dt for k in range(self.k_max_loop)]

        plt.figure()
        ylabels = ["Position X", "Position Y", "Position Z",
                   "Orientation Roll", "Orientation Pitch", "Orientation Yaw"]
        for i, j in enumerate([1, 3, 5, 2, 4, 6]):
            plt.subplot(3, 2, j)
            plt.plot(log_t, self.log_state[i, :], "b", linewidth=2)
            if i not in [0, 1, 6]:
                plt.plot(log_t, self.log_state_ref[i, :], "r", linewidth=2)
            plt.legend(["Robot", "Reference"])
            plt.xlabel("Time [s]")
            plt.ylabel(ylabels[i])

        plt.figure()
        ylabels = ["Linear vel X", "Linear vel Y", "Linear vel Z",
                   "Angular vel Roll", "Angular vel Pitch", "Angular vel Yaw"]
        for i, j in enumerate([1, 3, 5, 2, 4, 6]):
            plt.subplot(3, 2, j)
            plt.plot(log_t, self.log_state[i+6, :], "b", linewidth=2)
            plt.plot(log_t, self.log_state_ref[i+6, :], "r", linewidth=2)
            plt.legend(["Robot", "Reference"])
            plt.xlabel("Time [s]")
            plt.ylabel(ylabels[i])

        plt.figure()
        plt.plot(self.log_state[0, :], self.log_state[1, :], "b", linewidth=2)
        plt.plot(self.log_state_ref[0, :], self.log_state_ref[1, :], "r", linewidth=2)
        for i_foot in range(4):
            plt.plot(self.log_footholds_w[i_foot, :, 0],
                     self.log_footholds_w[i_foot, :, 1], linestyle=None, marker="*")
        plt.legend(["Robot", "Reference"])
        plt.xlabel("Position X [m]")
        plt.ylabel("Position Y [m]")

        plt.figure()
        for i in range(4):
            plt.subplot(4, 2, 2*i+1)
            plt.plot(log_t, self.log_footholds[i_foot, :, 0], linewidth=2)
            plt.xlabel("Time [s]")
            plt.ylabel("Position X [m]")
            plt.subplot(4, 2, 2*i+2)
            plt.plot(log_t, self.log_footholds[i_foot, :, 1], linewidth=2)
            plt.xlabel("Time [s]")
            plt.ylabel("Position Y [m]")

        plt.show(block=True)
        return 0
