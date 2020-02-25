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

        # Log a few predicted trajectories along time to see how it changes
        self.log_predicted_traj = np.zeros((12, 59, 120))

        # Log contact forces
        self.log_contact_forces = np.zeros((4, k_max_loop, 3))

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

    def log_predicted_trajectory(self, mpc, k_loop):
        """ Log the trajectory predicted by the MPC to see how it changes over time
        """

        k_start = 5
        k_gap = 20
        if k_loop >= k_start:
            if ((k_loop - k_start) % k_gap) == 0:
                x_log = mpc.x_robot.copy()
                c, s = np.cos(mpc.q_w[5, 0]), np.sin(mpc.q_w[5, 0])
                R = np.array([[c, -s], [s, c]])
                x_log[0:2, :] = np.dot(R, x_log[0:2, :]) + mpc.q_w[0:2, 0:1]
                x_log[5, :] += mpc.q_w[5, 0]
                i = int((k_loop - k_start)/k_gap)
                if i < self.log_predicted_traj.shape[1]:
                    self.log_predicted_traj[:, i, :] = x_log

        return 0

    def log_desired_contact_forces(self, mpc, sequencer, k_loop):
        """ Log the output contact forces of the MPC
        """

        cpt = 0
        update = np.array(sequencer.S[0]).ravel()
        for i in range(4):
            if update[i]:
                self.log_contact_forces[i, k_loop, :] = mpc.f_applied[(cpt*3):((cpt+1)*3)]
                cpt += 1

        return 0

    def call_log_functions(self, sequencer, fstep_planner, ftraj_gen, mpc, k_loop):
        """ Call logging functions of the Logger class
        """

        # Logging reference and current state vectors
        self.log_state_vectors(mpc, k_loop)

        # Log footholds
        self.log_various_footholds(mpc, k_loop)

        # Logging predicted trajectory of the robot to see how it changes over time
        self.log_predicted_trajectory(mpc, k_loop)

        # Log desired contact forces
        self.log_desired_contact_forces(mpc, sequencer, k_loop)

        return 0

    def plot_graphs(self, dt, k_max_loop, mycontroller=None):

        log_t = [k*dt for k in range(self.k_max_loop)]

        # Evolution of the position and orientation of the robot over time
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

        # Evolution of the linear and angular velocities of the robot over time
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

        # Evolution of the position of the robot in world frame (X, Y) graph
        plt.figure()
        plt.plot(self.log_state[0, :], self.log_state[1, :], "b", linewidth=2)
        plt.plot(self.log_state_ref[0, :], self.log_state_ref[1, :], "r", linewidth=2)
        for i_foot in range(4):
            plt.plot(self.log_footholds_w[i_foot, :, 0],
                     self.log_footholds_w[i_foot, :, 1], linestyle=None, marker="*")
        plt.legend(["Robot", "Reference"])
        plt.xlabel("Position X [m]")
        plt.ylabel("Position Y [m]")

        # Evolution of the position of footholds over time
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

        # Evolution of predicted trajectory over time
        log_t_pred = np.array([k*dt for k in range(self.log_predicted_traj.shape[2])])

        plt.figure()
        plt.subplot(3, 2, 1)
        c = [[i/(self.log_predicted_traj.shape[1]+5), 0.0, i/(self.log_predicted_traj.shape[1]+5)]
             for i in range(self.log_predicted_traj.shape[1])]
        for i in range(self.log_predicted_traj.shape[1]):
            plt.plot(log_t_pred+i*dt, self.log_predicted_traj[0, i, :], "b", linewidth=2, color=c[i])
        plt.plot(np.array([k*dt for k in range(self.log_predicted_traj.shape[1])]),
                 self.log_predicted_traj[0, :, 0], linestyle=None, marker='x', color="r", linewidth=1)
        plt.xlabel("Time [s]")
        plt.title("Predicted trajectory along X [m]")
        plt.legend(["Robot"])
        plt.subplot(3, 2, 3)
        for i in range(self.log_predicted_traj.shape[1]):
            plt.plot(log_t_pred+i*dt, self.log_predicted_traj[1, i, :], "b", linewidth=2, color=c[i])
        plt.plot(np.array([k*dt for k in range(self.log_predicted_traj.shape[1])]),
                 self.log_predicted_traj[1, :, 0], linestyle=None, marker='x', color="r", linewidth=1)
        plt.xlabel("Time [s]")
        plt.title("Predicted trajectory along Y [m]")
        plt.legend(["Robot"])
        plt.subplot(3, 2, 5)
        for i in range(self.log_predicted_traj.shape[1]):
            plt.plot(log_t_pred+i*dt, self.log_predicted_traj[2, i, :], "b", linewidth=2, color=c[i])
        plt.plot(np.array([k*dt for k in range(self.log_predicted_traj.shape[1])]),
                 self.log_predicted_traj[2, :, 0], linestyle=None, marker='x', color="r", linewidth=1)
        plt.xlabel("Time [s]")
        plt.title("Predicted trajectory along Z [m]")
        plt.legend(["Robot"])
        plt.subplot(3, 2, 2)
        for i in range(self.log_predicted_traj.shape[1]):
            plt.plot(log_t_pred+i*dt, self.log_predicted_traj[3, i, :], "b", linewidth=2, color=c[i])
        plt.plot(np.array([k*dt for k in range(self.log_predicted_traj.shape[1])]),
                 self.log_predicted_traj[3, :, 0], linestyle=None, marker='x', color="r", linewidth=1)
        plt.xlabel("Time [s]")
        plt.title("Predicted trajectory in Roll [rad/s]")
        plt.legend(["Robot"])
        plt.subplot(3, 2, 4)
        for i in range(self.log_predicted_traj.shape[1]):
            plt.plot(log_t_pred+i*dt, self.log_predicted_traj[4, i, :], "b", linewidth=2, color=c[i])
        plt.plot(np.array([k*dt for k in range(self.log_predicted_traj.shape[1])]),
                 self.log_predicted_traj[4, :, 0], linestyle=None, marker='x', color="r", linewidth=1)
        plt.xlabel("Time [s]")
        plt.title("Predicted trajectory in Pitch [rad/s]")
        plt.legend(["Robot"])
        plt.subplot(3, 2, 6)
        for i in range(self.log_predicted_traj.shape[1]):
            plt.plot(log_t_pred+i*dt, self.log_predicted_traj[5, i, :], "b", linewidth=2, color=c[i])
        plt.plot(np.array([k*dt for k in range(self.log_predicted_traj.shape[1])]),
                 self.log_predicted_traj[5, :, 0], linestyle=None, marker='x', color="r", linewidth=1)
        plt.xlabel("Time [s]")
        plt.title("Predicted trajectory in Yaw [rad/s]")
        plt.legend(["Robot"])

        # Plot desired contact forces
        plt.figure()
        c = ["r", "g", "b", "rebeccapurple"]
        legends = ["FL", "FR", "HL", "HR"]
        for i in range(4):
            plt.subplot(3, 4, i+1)
            plt.plot(log_t, self.log_contact_forces[i, :, 0], color="r", linewidth=3)
            if mycontroller is not None:
                plt.plot(log_t, mycontroller.c_forces[i, :, 0], color="b", linewidth=2)
            tmp = np.zeros((k_max_loop,))
            for k in range(np.floor(k_max_loop/(0.3/0.005)+1).astype(int)):
                plt.plot([k*0.3, k*0.3], [np.min(self.log_contact_forces[i, :, 0]),
                                          np.max(self.log_contact_forces[i, :, 0])], "grey", linewidth=1, linestyle="--")
            plt.xlabel("Time [s]")
            plt.ylabel("Contact force along X [N]")
            plt.legend([legends[i] + "_MPC", legends[i] + "_TSID"])

        for i in range(4):
            plt.subplot(3, 4, i+5)
            plt.plot(log_t, self.log_contact_forces[i, :, 1], color="r", linewidth=3)
            if mycontroller is not None:
                plt.plot(log_t, mycontroller.c_forces[i, :, 1], color="b", linewidth=2)
            tmp = np.zeros((k_max_loop,))
            for k in range(np.floor(k_max_loop/(0.3/0.005)+1).astype(int)):
                plt.plot([k*0.3, k*0.3], [np.min(self.log_contact_forces[i, :, 1]),
                                          np.max(self.log_contact_forces[i, :, 1])], "grey", linewidth=1, linestyle="--")
            plt.xlabel("Time [s]")
            plt.ylabel("Contact force along Y [N]")
            plt.legend([legends[i] + "_MPC", legends[i] + "_TSID"])

        for i in range(4):
            plt.subplot(3, 4, i+9)
            plt.plot(log_t, self.log_contact_forces[i, :, 2], color="r", linewidth=3)
            if mycontroller is not None:
                plt.plot(log_t, mycontroller.c_forces[i, :, 2], color="b", linewidth=2)
            tmp = np.zeros((k_max_loop,))
            for k in range(np.floor(k_max_loop/(0.3/0.005)+1).astype(int)):
                plt.plot([k*0.3, k*0.3], [np.min(self.log_contact_forces[i, :, 2]),
                                          np.max(self.log_contact_forces[i, :, 2])], "grey", linewidth=1, linestyle="--")
            plt.xlabel("Time [s]")
            plt.ylabel("Contact force along Z [N]")
            plt.legend([legends[i] + "_MPC", legends[i] + "_TSID"])

        plt.show(block=False)

        return 0
