# coding: utf8

import numpy as np
from matplotlib import pyplot as plt
import pybullet as pyb

class Logger:
    """Joystick-like controller that outputs the reference velocity in local frame
    """

    def __init__(self, k_max_loop, dt):

        # Max number of iterations of the main loop
        self.k_max_loop = k_max_loop

        # Time stamp
        self.dt = dt

        """# Log state vector and reference state vector
        self.log_state = np.zeros((12, k_max_loop))
        self.log_state_ref = np.zeros((12, k_max_loop))

        # Log footholds position in world frame
        self.log_footholds = np.zeros((4, k_max_loop, 2))
        self.log_footholds_w = np.zeros((4, k_max_loop, 2))

        # Log a few predicted trajectories along time to see how it changes
        self.log_predicted_traj = np.zeros((12, int((k_max_loop/20)), 60))
        self.log_predicted_fc = np.zeros((12, int((k_max_loop/20)), 60))

        # Log contact forces
        self.log_contact_forces = np.zeros((4, k_max_loop, 3))"""

        ###
        ###

        # Store time range
        self.t_range = np.array([k*dt for k in range(self.k_max_loop)])

        # Store current and desired position, velocity and acceleration of feet over time
        # Used in log_footsteps function
        self.feet_pos = np.zeros((3, 4, k_max_loop))
        self.feet_vel = np.zeros((3, 4, k_max_loop))
        self.feet_acc = np.zeros((3, 4, k_max_loop))
        self.feet_pos_target = np.zeros((3, 4, k_max_loop))
        self.feet_vel_target = np.zeros((3, 4, k_max_loop))
        self.feet_acc_target = np.zeros((3, 4, k_max_loop))

        # Store information about the state of the robot
        self.RPY = np.zeros((3, k_max_loop))  # roll, pitch, yaw of the base in world frame
        self.oC = np.zeros((3, k_max_loop))  #  position of the CoM in world frame
        self.oV = np.zeros((3, k_max_loop))  #  linear velocity of the CoM in world frame
        self.oW = np.zeros((3, k_max_loop))  # angular velocity of the CoM in world frame
        self.lC = np.zeros((3, k_max_loop))  #  position of the CoM in local frame
        self.lV = np.zeros((3, k_max_loop))  #  linear velocity of the CoM in local frame
        self.lW = np.zeros((3, k_max_loop))  #  angular velocity of the CoM in local frame
        self.state_ref = np.zeros((12, k_max_loop))  #  reference state vector

        # Store information about contact forces
        self.forces_order = [0, 1, 2, 3]
        self.forces_status = [1, 1, 1, 1]
        self.forces_mpc = np.zeros((12, k_max_loop))  # output of MPC
        self.forces_tsid = np.zeros((12, k_max_loop))  # output of TSID
        self.forces_pyb = np.zeros((12, k_max_loop))  # forces applied in PyBullet

        # Store information about torques
        self.torques_ff = np.zeros((12, k_max_loop))
        self.torques_sent = np.zeros((12, k_max_loop))

        # Store information about the cost function of the MPC
        self.cost_components = np.zeros((13, k_max_loop))

        # Store information about the predicted evolution of the optimization vector components
        T = 0.32
        dt_mpc = 0.02
        self.pred_trajectories = np.zeros((12, int(T/dt_mpc), int(k_max_loop/20)))

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

        """cpt = 0
        update = np.array(sequencer.S[0]).ravel()
        for i in range(4):
            if update[i]:
                self.log_contact_forces[i, k_loop, :] = mpc.f_applied[(cpt*3):((cpt+1)*3)]
                cpt += 1"""

        self.log_contact_forces[:, k_loop, :] = mpc.f_applied.reshape((4, 3))

        return 0

    def log_footsteps(self, k, mpc_interface, tsid_controller):
        """ Store current and desired position, velocity and acceleration of feet over time
        """

        self.feet_pos[:, :, k] = mpc_interface.o_feet
        self.feet_vel[:, :, k] = mpc_interface.ov_feet
        self.feet_acc[:, :, k] = mpc_interface.oa_feet
        self.feet_pos_target[0:2, :, k] = tsid_controller.goals.copy()
        self.feet_vel_target[0:2, :, k] = tsid_controller.vgoals.copy()
        self.feet_acc_target[0:2, :, k] = tsid_controller.agoals.copy()

        return 0

    def plot_footsteps(self):
        """ Plot current and desired position, velocity and acceleration of feet over time
        """

        index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
        index = [1, 3, 5, 2, 4, 6]

        lgd = ["Pos X FL", "Pos Y FL", "Pos Z FL", "Pos X FR", "Pos Y FR", "Pos Z FR"]
        plt.figure()
        for i in range(6):
            plt.subplot(3, 2, index[i])
            plt.plot(self.t_range, self.feet_pos[i % 3, np.int(i/3), :], linewidth=2, marker='x')
            plt.plot(self.t_range, self.feet_pos_target[i % 3, np.int(i/3), :], linewidth=2, marker='x')
            plt.legend([lgd[i], lgd[i]+" Ref"])
        plt.suptitle("Current and desired position of feet (world frame)")

        lgd = ["Vel X FL", "Vel Y FL", "Vel Z FL", "Vel X FR", "Vel Y FR", "Vel Z FR"]
        plt.figure()
        for i in range(6):
            plt.subplot(3, 2, index[i])
            plt.plot(self.t_range, self.feet_vel[i % 3, np.int(i/3), :], linewidth=2, marker='x')
            plt.plot(self.t_range, self.feet_vel_target[i % 3, np.int(i/3), :], linewidth=2, marker='x')
            plt.legend([lgd[i], lgd[i]+" Ref"])
        plt.suptitle("Current and desired velocity of feet (world frame)")

        lgd = ["Acc X FL", "Acc Y FL", "Acc Z FL", "Acc X FR", "Acc Y FR", "Acc Z FR"]
        plt.figure()
        for i in range(6):
            plt.subplot(3, 2, index[i])
            plt.plot(self.t_range, self.feet_acc[i % 3, np.int(i/3), :], linewidth=2, marker='x')
            plt.plot(self.t_range, self.feet_acc_target[i % 3, np.int(i/3), :], linewidth=2, marker='x')
            plt.legend([lgd[i], lgd[i]+" Ref"])
        plt.suptitle("Current and desired acceleration of feet (world frame)")

        return 0

    def log_state(self, k, joystick, mpc_interface, mpc_wrapper):
        """ Store information about the state of the robot
        """

        self.RPY[:, k:(k+1)] = mpc_interface.RPY[:, 0]  # roll, pitch, yaw of the base in world frame
        self.oC[:, k:(k+1)] = mpc_interface.oC[:, 0]  #  position of the CoM in world frame
        self.oV[:, k:(k+1)] = mpc_interface.oV[:, 0]  #  linear velocity of the CoM in world frame
        self.oW[:, k] = mpc_interface.oW[:, 0]  # angular velocity of the CoM in world frame
        self.lC[:, k:(k+1)] = mpc_interface.lC[:, 0]  #  position of the CoM in local frame
        self.lV[:, k:(k+1)] = mpc_interface.lV[:, 0]  #  linear velocity of the CoM in local frame
        self.lW[:, k:(k+1)] = mpc_interface.lW[:, 0]  #  angular velocity of the CoM in local frame

        # Reference state vector in local frame
        # Velocity control for x, y and yaw components (user input)
        # Position control for z, roll and pitch components (hardcoded default values of h_ref, 0.0 and 0.0)
        self.state_ref[0:6, k] = np.array([0.0, 0.0, mpc_wrapper.mpc.h_ref, 0.0, 0.0, 0.0])
        self.state_ref[6:12, k] = joystick.v_ref[:, 0]

        return 0

    def plot_state(self):
        """ Plot information about the state of the robot
        """

        # Evolution of the position of CoM and orientation of the robot over time (world frame)
        plt.figure()
        ylabels = ["Position X", "Position Y", "Position Z",
                   "Orientation Roll", "Orientation Pitch", "Orientation Yaw"]
        for i, j in enumerate([1, 3, 5, 2, 4, 6]):
            plt.subplot(3, 2, j)
            if i < 3:
                plt.plot(self.t_range, self.oC[i, :], "b", linewidth=2)
            else:
                plt.plot(self.t_range, self.RPY[i-3, :], "b", linewidth=2)
            plt.xlabel("Time [s]")
            plt.ylabel(ylabels[i])
        plt.suptitle("Position of CoM and orientation of the robot (world frame)")

        # Evolution of the linear and angular velocities of the robot over time (world frame)
        plt.figure()
        ylabels = ["Linear vel X", "Linear vel Y", "Linear vel Z",
                   "Angular vel Roll", "Angular vel Pitch", "Angular vel Yaw"]
        for i, j in enumerate([1, 3, 5, 2, 4, 6]):
            plt.subplot(3, 2, j)
            if i < 3:
                plt.plot(self.t_range, self.oV[i, :], "b", linewidth=2)
            else:
                plt.plot(self.t_range, self.oW[i-3, :], "b", linewidth=2)
            plt.xlabel("Time [s]")
            plt.ylabel(ylabels[i])
        plt.suptitle("Linear and angular velocities of the robot (world frame)")

        # Evolution of the linear and angular velocities of the robot over time (local frame)
        plt.figure()
        ylabels = ["Linear vel X", "Linear vel Y", "Linear vel Z",
                   "Angular vel Roll", "Angular vel Pitch", "Angular vel Yaw"]
        for i, j in enumerate([1, 3, 5, 2, 4, 6]):
            plt.subplot(3, 2, j)
            if i < 3:
                plt.plot(self.t_range, self.lV[i, :], "b", linewidth=2)
            else:
                plt.plot(self.t_range, self.lW[i-3, :], "b", linewidth=2)
            plt.xlabel("Time [s]")
            plt.ylabel(ylabels[i])
        plt.suptitle("Linear and angular velocities of the robot (local frame)")

        # Evolution of the position of the center of mass over time (local frame)
        plt.figure()
        ylabels = ["Position X", "Position Y", "Position Z"]
        for i, j in enumerate([1, 2, 3]):
            plt.subplot(3, 1, j)
            plt.plot(self.t_range, self.lC[i, :], "b", linewidth=2)
            plt.xlabel("Time [s]")
            plt.ylabel(ylabels[i])
        plt.suptitle("Position of the center of mass over time (local frame)")

        # Evolution of the position of the robot CoM in world frame (X, Y) graph
        plt.figure()
        plt.plot(self.oC[0, :], self.oC[1, :], "b", linewidth=2)
        plt.xlabel("Position X [m]")
        plt.ylabel("Position Y [m]")
        plt.title("Position of the robot CoM in world frame (X, Y) graph")

        # Evolution of the linear and angular velocities of the robot over time (world frame)
        plt.figure()
        ylabels = ["Position Z", "Position Roll", "Position Pitch", "Linear vel X", "Linear vel Y", "Linear vel Z",
                   "Angular vel Roll", "Angular vel Pitch", "Angular vel Yaw"]
        for i, j in enumerate([1, 4, 7, 2, 5, 8, 3, 6, 9]):
            plt.subplot(3, 3, j)
            if i == 0:
                plt.plot(self.t_range, self.lC[2, :], "b", linewidth=2)
                plt.plot(self.t_range, self.state_ref[2, :], "r", linewidth=2)
            elif i <= 2:
                plt.plot(self.t_range, self.RPY[i-1, :], "b", linewidth=2)
                plt.plot(self.t_range, self.state_ref[2+i, :], "r", linewidth=2)
            elif i <= 5:
                plt.plot(self.t_range, self.lV[i-3, :], "b", linewidth=2)
                plt.plot(self.t_range, self.state_ref[6+i-3, :], "r", linewidth=2)
            else:
                plt.plot(self.t_range, self.lW[i-6, :], "b", linewidth=2)
                plt.plot(self.t_range, self.state_ref[6+i-3, :], "r", linewidth=2)
            plt.xlabel("Time [s]")
            plt.ylabel(ylabels[i])
            plt.legend(["Performed", "Desired"])

        plt.suptitle("Performed trajectory VS Desired trajectory (local frame)")

        return 0

    def getContactPoint(self, contactPoints):
        """ Sort contacts points as there should be only one contact per foot
            and sometimes PyBullet detect several of them. There is one contact
            with a non zero force and the others have a zero contact force
        """

        indexes = []
        for i in range(0, len(contactPoints)):
            # There may be several contact points for each foot but only one of them as a non zero normal force
            if (contactPoints[i][9] != 0):
                indexes.append(i)

        if len(indexes) > 0:
            # Return contact points
            return [contactPoints[i] for i in indexes]
        else:
            # If it returns 0 then it means there is no contact point with a non zero normal force (should not happen)
            return [0]

    def log_forces(self, k, mpc_interface, tsid_controller, robotId, planeId):
        """ Store information about contact forces
        """

        """if (k > 0) and (k % 20 == 0):

            # Index of the first empty line
            index = next((idx for idx, val in np.ndenumerate(fstep_planner.gait[:, 0]) if val == 0.0), 0.0)[0]

            # Change of state (+1 if start of stance phase, -1 if start of swing phase, 0 if no state change)
            if not np.array_equal(fstep_planner.gait[0, 1:], fstep_planner.gait[index-1, 1:]):
                self.forces_status = fstep_planner.gait[0, 1:] - fstep_planner.gait[index-1, 1:]

            # Update contact forces order
            for i in range(4):
                if self.forces_status[i] == 1:
                    self.forces_order.append(i)
                elif self.forces_status[i] == -1:
                    self.forces_order.remove(i)

            print(self.forces_status)
            deb = 1 """

        # Contact forces desired by MPC (transformed into world frame)
        for f in range(4):
            self.forces_mpc[3*f:(3*(f+1)), k:(k+1)] = (mpc_interface.oMl.rotation @ tsid_controller.f_applied[3*f:3*(f+1)]).T

        # Contact forces desired by TSID (world frame)
        for i, j in enumerate(tsid_controller.contacts_order):
            self.forces_tsid[(3*j):(3*(j+1)), k:(k+1)] = tsid_controller.fc[(3*i):(3*(i+1))]

        # Contact forces applied in PyBullet
        contactPoints_FL = pyb.getContactPoints(robotId, planeId, linkIndexA=3)  # Front left  foot
        contactPoints_FR = pyb.getContactPoints(robotId, planeId, linkIndexA=7)  # Front right foot
        contactPoints_HL = pyb.getContactPoints(robotId, planeId, linkIndexA=11)  # Hind left  foot
        contactPoints_HR = pyb.getContactPoints(robotId, planeId, linkIndexA=15)  # Hind right foot

        # Sort contacts points to get only one contact per foot
        contactPoints = []
        contactPoints += self.getContactPoint(contactPoints_FL)
        contactPoints += self.getContactPoint(contactPoints_FR)
        contactPoints += self.getContactPoint(contactPoints_HL)
        contactPoints += self.getContactPoint(contactPoints_HR)

        # Display debug lines for contact forces visualization
        f_tmps = np.zeros((3, 4))
        f_tmp = [0.0] * 3
        for contact in contactPoints:
            if not isinstance(contact, int):  # type(contact) != type(0):
                start = [contact[6][0], contact[6][1], contact[6][2]+0.04]
                end = [contact[6][0], contact[6][1], contact[6][2]+0.04]
                K = 0.02
                for i_direction in range(0, 3):
                    f_tmp[i_direction] = (contact[9] * contact[7][i_direction] + contact[10] *
                                          contact[11][i_direction] + contact[12] * contact[13][i_direction])
                    end[i_direction] += K * f_tmp[i_direction]

                """if contact[3] < 10:
                    print("Link  ", contact[3], "| Contact force: (", f_tmp[0], ", ", f_tmp[1], ", ", f_tmp[2], ")")
                else:
                    print("Link ", contact[3], "| Contact force: (", f_tmp[0], ", ", f_tmp[1], ", ", f_tmp[2], ")")"""

                f_tmps[:, int(contact[3]/4)] += np.array(f_tmp)

        for i in range(4):
            self.forces_pyb[(3*i):(3*(i+1)), k] = f_tmps[:, i]

        return 0

    def plot_forces(self):
        """ Plot information about contact forces
        """

        # Evolution of the linear and angular velocities of the robot over time (world frame)
        plt.figure()
        ylabels = ["Contact forces X", "Contact forces Y", "Contact forces Z"]
        for i, j in enumerate([1, 2, 3]):
            plt.subplot(3, 1, j)
            for f in range(4):
                h1, = plt.plot(self.t_range, self.forces_mpc[3*f+i, :], "b", linewidth=2)
                h2, = plt.plot(self.t_range, self.forces_tsid[3*f+i, :], "r", linewidth=2)
                h3, = plt.plot(self.t_range, self.forces_pyb[3*f+i, :], "darkgreen", linewidth=2)
            if i == 2:
                tmp = self.forces_pyb[2, :]
                for f in range(1, 4):
                    tmp += self.forces_pyb[3*f+2, :]
                h4, = plt.plot(self.t_range, tmp, "rebeccapurple", linewidth=2, linestyle="--")
                plt.legend([h1, h2, h3, h4], ["MPC", "TSID", "PyB", "Sum 4 feet"])
            else:
                plt.legend([h1, h2, h3], ["MPC", "TSID", "PyB"])
            plt.xlabel("Time [s]")
            plt.ylabel(ylabels[i])

        plt.suptitle("MPC, TSID and PyBullet contact forces (world frame)")     

        return 0

    def log_torques(self, k, tsid_controller):
        """ Store information about torques
        """

        self.torques_ff[:, k:(k+1)] = tsid_controller.tau_ff
        self.torques_sent[:, k:(k+1)] = tsid_controller.tau.transpose()

        return 0

    def plot_torques(self):
        """ Plot information about torques
        """

        index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]

        lgd = ["DoF 1 FL", "DoF 2 FL", "DoF 3 FL", "DoF 1 FR", "DoF 2 FR", "DoF 3 FR",
               "DoF 1 HL", "DoF 2 HL", "DoF 3 HL", "DoF 1 HR", "DoF 2 HR", "DoF 3 HR"]
        plt.figure()
        for i in range(12):
            plt.subplot(3, 4, index[i])
            h1, = plt.plot(self.t_range, self.torques_ff[i, :], linewidth=2)
            h2, = plt.plot(self.t_range, self.torques_sent[i, :], linewidth=2)
            plt.legend([h1, h2], [lgd[i] + " FF", lgd[i]+" Sent"])
            plt.xlabel("Time [s]")
            plt.ylabel("Torque [Nm]")
        plt.suptitle("Feedforward torques and sent torques (output of PD + saturation)")

        return 0

    def log_cost_function(self, k, mpc_wrapper):
        """ Store information about the cost function of the mpc
        """

        # Cost of each component of the cost function over the prediction horizon (state vector and contact forces)
        cost = (np.diag(mpc_wrapper.mpc.x) @ np.diag(mpc_wrapper.mpc.P.data)) @ np.array([mpc_wrapper.mpc.x]).transpose()

        # Sum components of the state vector
        for i in range(12):
            self.cost_components[i, k:(k+1)] = np.sum(cost[i:(12*mpc_wrapper.mpc.n_steps):12])

        # Sum components of the contact forces
        self.cost_components[12, k:(k+1)] = np.sum(cost[(12*mpc_wrapper.mpc.n_steps):])

        return 0

    def plot_cost_function(self):
        """ Plot information about the cost function of the mpc
        """

        lgd = ["$X$", "$Y$", "$Z$", "$\phi$", "$\\theta$", "$\psi$", "$\dot X$", "$\dot Y$", "$\dot Z$",
               "$\dot \phi$", "$\dot \\theta$", "$\dot \psi$", "$f$", "Total"]
        plt.figure()
        hs = []
        for i in range(14):
            if i < 10:
                h, = plt.plot(self.t_range, self.cost_components[i, :], linewidth=2)
            elif i<=12:
                h, = plt.plot(self.t_range, self.cost_components[i, :], linewidth=2, linestyle="--")
            else:
                h, = plt.plot(self.t_range, np.sum(self.cost_components, axis=0), linewidth=2, linestyle="--")
            hs.append(h)
            plt.xlabel("Time [s]")
            plt.ylabel("Cost")
        plt.legend(hs, lgd)
        plt.title("Contribution of each component in the cost function")

        return 0

    def log_predicted_trajectories(self, k, mpc_wrapper):
        """ Store information about the predicted evolution of the optimization vector components
        """

        self.pred_trajectories[:, :, int(k/20)] = mpc_wrapper.mpc.x_robot

        return 0

    def plot_predicted_trajectories(self):
        """ Plot information about the predicted evolution of the optimization vector components
        """

        dt_mpc = 0.02
        t_pred = np.array([(k+1)*dt_mpc for k in range(16)])

        #index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
        index = [1, 3, 5, 2, 4, 6]

        lgd = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
        plt.figure()
        for i in range(6):
            plt.subplot(3, 2, index[i])
            for j in range(self.pred_trajectories.shape[2]):
                if (j % 6) == 0:
                    h, = plt.plot(t_pred + j*dt_mpc*20, self.pred_trajectories[i, :, j], linewidth=2, marker='x')
            plt.ylabel(lgd[i])
        plt.suptitle("Predicted trajectories (local frame)")

        return 0

    def call_log_functions(self, k, sequencer, joystick, fstep_planner, mpc_interface, mpc_wrapper, tsid_controller, enable_multiprocessing, robotId, planeId):
        """ Call logging functions of the Logger class
        """

        """# Logging reference and current state vectors
        self.log_state_vectors(mpc, k_loop)

        # Log footholds
        self.log_various_footholds(mpc, k_loop)

        # Logging predicted trajectory of the robot to see how it changes over time
        self.log_predicted_trajectory(mpc, k_loop)

        # Log desired contact forces
        self.log_desired_contact_forces(mpc, sequencer, k_loop)"""

        # Store current and desired position, velocity and acceleration of feet over time
        self.log_footsteps(k, mpc_interface, tsid_controller)

        # Store information about the state of the robot
        if not enable_multiprocessing:
            self.log_state(k, joystick, mpc_interface, mpc_wrapper)

        # Store information about contact forces
        self.log_forces(k, mpc_interface, tsid_controller, robotId, planeId)

        # Store information about torques
        self.log_torques(k, tsid_controller)

        # Store information about the cost function
        if not enable_multiprocessing:
            self.log_cost_function(k, mpc_wrapper)

        # Store information about the predicted evolution of the optimization vector components
        if not enable_multiprocessing and ((k % 20) == 0):
            self.log_predicted_trajectories(k, mpc_wrapper)

        return 0

    def plot_graphs(self, enable_multiprocessing):

        # Plot current and desired position, velocity and acceleration of feet over time
        self.plot_footsteps()

        # Plot information about the state of the robot
        if not enable_multiprocessing:
            self.plot_state()

        # Plot information about the contact forces
        self.plot_forces()

        # Plot information about the torques
        self.plot_torques()

        # Plot information about the state of the robot
        if not enable_multiprocessing:
            self.plot_cost_function()

        # Plot information about the predicted evolution of the optimization vector components
        if not enable_multiprocessing:
            self.plot_predicted_trajectories()

        # Display graphs
        plt.show(block=True)

        """# Evolution of the position and orientation of the robot over time
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
            plt.plot(log_t, self.log_footholds[i, :, 0], linewidth=2)
            plt.xlabel("Time [s]")
            plt.ylabel("Position X [m]")
            plt.subplot(4, 2, 2*i+2)
            plt.plot(log_t, self.log_footholds[i, :, 1], linewidth=2)
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

        plt.show(block=False)"""

        return 0
