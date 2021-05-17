# coding: utf8

import numpy as np
#from matplotlib import pyplot as plt
import pybullet as pyb
import pinocchio as pin
# import scipy.stats as scipystats
from matplotlib import cm


class Logger:
    """Logger object to store information about plenty of different quantities in the simulation

    Args:
        k_max_loop (int): maximum number of iterations of the simulation
        dt (float): time step of TSID
        dt_mpc (float): time step of the MPC
        k_mpc (int): number of tsid iterations for one iteration of the mpc
        T_mpc (float): duration of mpc prediction horizon
        type_MPC (bool): which MPC you want to use (PA's or Thomas')
    """

    def __init__(self, k_max_loop, dt, dt_mpc, k_mpc, T_mpc, type_MPC):

        # Max number of iterations of the main loop
        self.k_max_loop = k_max_loop

        # Time step of TSID
        self.dt = dt

        # Time step of MPC
        self.dt_mpc = dt_mpc

        # Number of TSID steps for 1 step of the MPC
        self.k_mpc = k_mpc

        # Which MPC solver you want to use
        # True to have PA's MPC, to False to have Thomas's MPC
        self.type_MPC = type_MPC

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

        # Store logging timestamps
        self.tstamps = np.zeros(k_max_loop)

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
        self.mot = np.zeros((12, k_max_loop))  #  angular position of actuators
        self.Vmot = np.zeros((12, k_max_loop))  #  angular velocity of actuators

        # Position and velocity data in PyBullet simulation
        self.lC_pyb = np.zeros((3, k_max_loop))
        self.RPY_pyb = np.zeros((3, k_max_loop))
        self.mot_pyb = np.zeros((12, k_max_loop))
        self.lV_pyb = np.zeros((3, k_max_loop))
        self.lW_pyb = np.zeros((3, k_max_loop))
        self.Vmot_pyb = np.zeros((12, k_max_loop))

        # Store information about contact forces
        self.forces_order = [0, 1, 2, 3]
        self.forces_status = [1, 1, 1, 1]
        self.forces_mpc = np.zeros((12, k_max_loop))  # output of MPC
        self.forces_tsid = np.zeros((12, k_max_loop))  # output of TSID
        self.forces_pyb = np.zeros((12, k_max_loop))  # forces applied in PyBullet

        # Store information about torques
        self.torques_ff = np.zeros((12, k_max_loop))
        self.torques_pd = np.zeros((12, k_max_loop))
        self.torques_sent = np.zeros((12, k_max_loop))

        # Store information about the cost function of the MPC
        self.cost_components = np.zeros((13, k_max_loop))

        # Store information about the predicted evolution of the optimization vector components
        # Usefull to compare osqp & ddp solver
        self.T = T_mpc
        self.pred_trajectories = np.zeros((12, int(T_mpc/dt_mpc), int(k_max_loop/k_mpc)))
        self.pred_forces = np.zeros((12, int(T_mpc/dt_mpc), int(k_max_loop/k_mpc)))
        self.fsteps = np.zeros((20, 13, int(k_max_loop/k_mpc)))
        self.xref = np.zeros((12, int(T_mpc/dt_mpc) + 1, int(k_max_loop/k_mpc)))

        # Store information about one of the tracking task
        self.pos = np.zeros((12, k_max_loop))
        self.pos_ref = np.zeros((12, k_max_loop))
        self.pos_err = np.zeros((3, k_max_loop))
        self.vel = np.zeros((6, k_max_loop))
        self.vel_ref = np.zeros((6, k_max_loop))
        self.vel_err = np.zeros((3, k_max_loop))

        # Store information about shoulder position
        self.o_shoulders = np.zeros((3, 4,k_max_loop))

    def log_footsteps(self, k, interface, tsid_controller):
        """ Store current and desired position, velocity and acceleration of feet over time
        """

        self.feet_pos[:, :, k] = interface.o_feet
        self.feet_vel[:, :, k] = interface.ov_feet
        self.feet_acc[:, :, k] = interface.oa_feet
        self.feet_pos_target[:, :, k] = tsid_controller.goals.copy()
        self.feet_vel_target[:, :, k] = tsid_controller.vgoals.copy()
        self.feet_acc_target[:, :, k] = tsid_controller.agoals.copy()

        #  Shoulder position in world frame
        self.o_shoulders[:, :, k] = interface.o_shoulders

        return 0

    def plot_footsteps(self):
        """ Plot current and desired position, velocity and acceleration of feet over time
        """

        # Target feet positions are only updated during swing phase so during the initial stance phase these
        # positions are 0.0 for x, y, z even if it is not the initial feet positions
        # For display purpose we set feet target positions to feet positions from the beginning of the simulation to
        # the start of the first swing phase
        # Instead of having 0.0 0.0 0.0 0.19 0.19 0.19 (step when the first swing phase start and the target position
        # is updated) we have 0.19 0.19 0.19 0.19 0.19 0.19 (avoid the step on the graph)
        for i in range(4):
            index = next((idx for idx, val in np.ndenumerate(
                self.feet_pos_target[0, i, :]) if ((not (val == 0.0)))), [-1])[0]
            if index > 0:
                for j in range(2):
                    self.feet_pos_target[j, i, :(index+1)] = self.feet_pos_target[j, i, index+1]

        index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
        lgd_X = ["FL", "FR", "HL", "HR"]
        lgd_Y = ["Pos X", "Pos Y", "Pos Z"]
        plt.figure()
        for i in range(12):
            plt.subplot(3, 4, index[i])
            plt.plot(self.t_range, self.feet_pos[i % 3, np.int(i/3), :], color='b', linewidth=3, marker='')
            plt.plot(self.t_range, self.feet_pos_target[i % 3, np.int(i/3), :], color='r', linewidth=3, marker='')
            plt.legend([lgd_Y[i % 3] + " " + lgd_X[np.int(i/3)], lgd_Y[i % 3] + " " + lgd_X[np.int(i/3)]+" Ref"])
        plt.suptitle("Current and reference positions of feet (world frame)")

        index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
        lgd_X = ["FL", "FR", "HL", "HR"]
        lgd_Y = ["Vel X", "Vel Y", "Vel Z"]
        plt.figure()
        for i in range(12):
            plt.subplot(3, 4, index[i])
            plt.plot(self.t_range, self.feet_vel[i % 3, np.int(i/3), :], color='b', linewidth=3, marker='')
            plt.plot(self.t_range, self.feet_vel_target[i % 3, np.int(i/3), :], color='r', linewidth=3, marker='')
            plt.legend([lgd_Y[i % 3] + " " + lgd_X[np.int(i/3)], lgd_Y[i % 3] + " " + lgd_X[np.int(i/3)]+" Ref"])
        plt.suptitle("Current and reference velocities of feet (world frame)")

        index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
        lgd_X = ["FL", "FR", "HL", "HR"]
        lgd_Y = ["Acc X", "Acc Y", "Acc Z"]
        plt.figure()
        for i in range(12):
            plt.subplot(3, 4, index[i])
            plt.plot(self.t_range, self.feet_acc[i % 3, np.int(i/3), :], color='b', linewidth=3, marker='')
            plt.plot(self.t_range, self.feet_acc_target[i % 3, np.int(i/3), :], color='r', linewidth=3, marker='')
            plt.legend([lgd_Y[i % 3] + " " + lgd_X[np.int(i/3)], lgd_Y[i % 3] + " " + lgd_X[np.int(i/3)]+" Ref"])
        plt.suptitle("Current and reference accelerations of feet (world frame)")

        return 0

    def log_state(self, k, pyb_sim, joystick, interface, mpc_wrapper, solo):
        """ Store information about the state of the robot
        """

        self.RPY[:, k:(k+1)] = np.reshape(interface.RPY[:], (3, 1))  # roll, pitch, yaw of the base in world frame
        self.oC[:, k:(k+1)] = np.reshape(interface.oC[:, 0], (3, 1))  #  position of the CoM in world frame
        self.oV[:, k:(k+1)] = np.reshape(interface.oV[:, 0], (3, 1))  #  linear velocity of the CoM in world frame
        self.oW[:, k] = interface.oW[:, 0]  # angular velocity of the CoM in world frame
        self.lC[:, k:(k+1)] = np.reshape(interface.lC[:, 0], (3, 1))  #  position of the CoM in local frame
        self.lV[:, k:(k+1)] = np.reshape(interface.lV[:, 0], (3, 1))  #  linear velocity of the CoM in local frame
        self.lW[:, k:(k+1)] = np.reshape(interface.lW[:, 0], (3, 1))  #  angular velocity of the CoM in local frame
        self.mot[:, k:(k+1)] = interface.mot[:, 0:1]
        self.Vmot[:, k:(k+1)] = interface.vmes12_base[6:, 0:1]

        # Get PyBullet velocity in base frame for Pinocchio
        oRb = pin.Quaternion(pyb_sim.qmes12[3:7]).matrix()
        vmes12_base = pyb_sim.vmes12.copy()
        vmes12_base[0:3, 0:1] = oRb.transpose() @ vmes12_base[0:3, 0:1]
        vmes12_base[3:6, 0:1] = oRb.transpose() @ vmes12_base[3:6, 0:1]

        # Get CoM position in PyBullet simulation
        pin.centerOfMass(solo.model, solo.data, pyb_sim.qmes12, vmes12_base)

        self.RPY_pyb[:, k:(k+1)] = np.reshape(pin.rpy.matrixToRpy((pin.SE3(pin.Quaternion(pyb_sim.qmes12[3:7]),
                                                                           np.array([0.0, 0.0, 0.0]))).rotation),
                                              (3, 1))
        oMl = pin.SE3(pin.utils.rotate('z', self.RPY_pyb[2, k]),
                      np.array([pyb_sim.qmes12[0, 0], pyb_sim.qmes12[1, 0], interface.mean_feet_z]))

        # Store data about PyBullet simulation
        self.lC_pyb[:, k:(k+1)] = np.reshape(oMl.inverse() * solo.data.com[0], (3, 1))
        # self.RPY_pyb[2, k:(k+1)] = 0.0
        self.mot_pyb[:, k:(k+1)] = pyb_sim.qmes12[7:, 0:1]
        self.lV_pyb[:, k:(k+1)] = np.reshape(oMl.rotation.transpose() @ solo.data.vcom[0], (3, 1))
        self.lW_pyb[:, k:(k+1)] = np.reshape(oMl.rotation.transpose() @ pyb_sim.vmes12[3:6, 0:1], (3, 1))
        self.Vmot_pyb[:, k:(k+1)] = pyb_sim.vmes12[6:, 0:1]

        # Reference state vector in local frame
        # Velocity control for x, y and yaw components (user input)
        # Position control for z, roll and pitch components (hardcoded default values of h_ref, 0.0 and 0.0)
        # if self.type_MPC:
        #    self.state_ref[0:6, k] = np.array([0.0, 0.0, mpc_wrapper.mpc.h_ref, 0.0, 0.0, 0.0])
        # else:
        self.state_ref[0:6, k] = np.array([0.0, 0.0, 0.2027682, 0.0, 0.0, 0.0])
        self.state_ref[6:12, k] = joystick.v_ref[:, 0]

        return 0

    def plot_state(self):
        """ Plot information about the state of the robot
        """

        index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
        lgd = ["Pos CoM X [m]", "Pos CoM Y [m]", "Pos CoM Z [m]", "Roll [deg]", "Pitch [deg]", "Yaw [deg]",
               "Lin Vel CoM X [m/s]", "Lin Vel CoM Y [m/s]", "Lin Vel CoM Z [m/s]", "Ang Vel Roll [deg/s]",
               "Ang Vel Pitch [deg/s]", "Ang Vel Yaw [deg/s]"]
        plt.figure()
        for i in range(12):
            plt.subplot(3, 4, index[i])
            if i < 3:
                plt.plot(self.t_range, self.lC[i, :], "b", linewidth=3)
                plt.plot(self.t_range, self.lC_pyb[i, :], "g", linewidth=3)
            elif i < 6:
                plt.plot(self.t_range, (180/3.1415)*self.RPY[i-3, :], "b", linewidth=3)
                plt.plot(self.t_range, (180/3.1415)*self.RPY_pyb[i-3, :], "g", linewidth=3)
            elif i < 9:
                plt.plot(self.t_range, self.lV[i-6, :], "b", linewidth=3)
                plt.plot(self.t_range, self.lV_pyb[i-6, :], "g", linewidth=3)
            else:
                plt.plot(self.t_range, (180/3.1415)*self.lW[i-9, :], "b", linewidth=3)
                plt.plot(self.t_range, (180/3.1415)*self.lW_pyb[i-9, :], "g", linewidth=3)

            if i in [2, 6, 7, 8]:
                plt.plot(self.t_range, self.state_ref[i, :], "r", linewidth=3)
            elif i in [3, 4, 9, 10, 11]:
                plt.plot(self.t_range, (180/3.1415)*self.state_ref[i, :], "r", linewidth=3)

            plt.xlabel("Time [s]")
            plt.ylabel(lgd[i])

            plt.legend(["TSID", "Pyb", "Ref"])

            if i < 2:
                plt.ylim([-0.07, 0.07])
            elif i == 2:
                plt.ylim([0.16, 0.24])
            elif i < 6:
                plt.ylim([-10, 10])
            elif i == 6:
                plt.ylim([-0.05, 0.7])
            elif i == 7:
                plt.ylim([-0.1, 0.1])
            elif i == 8:
                plt.ylim([-0.2, 0.2])
            else:
                plt.ylim([-80.0, 80.0])

        plt.suptitle("State of the robot in TSID Vs PyBullet Vs Reference (local frame)")

        index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]

        lgd = ["DoF 1 FL", "DoF 2 FL", "DoF 3 FL", "DoF 1 FR", "DoF 2 FR", "DoF 3 FR",
               "DoF 1 HL", "DoF 2 HL", "DoF 3 HL", "DoF 1 HR", "DoF 2 HR", "DoF 3 HR"]
        plt.figure()
        for i in range(12):
            plt.subplot(3, 4, index[i])
            h1, = plt.plot(self.t_range, self.mot[i, :], color="b", linewidth=3)
            h2, = plt.plot(self.t_range, self.mot_pyb[i, :], color="g", linewidth=3)
            plt.legend([h1, h2], [lgd[i] + " TSID", lgd[i] + " Pyb"])
            plt.xlabel("Time [s]")
            plt.ylabel("Angular position [rad]")
        plt.suptitle("Angular positions of actuators in TSID and PyBullet")

        plt.figure()
        for i in range(12):
            plt.subplot(3, 4, index[i])
            h1, = plt.plot(self.t_range, self.Vmot[i, :], color="b", linewidth=3)
            h2, = plt.plot(self.t_range, self.Vmot_pyb[i, :], color="g", linewidth=3)
            plt.legend([h1, h2], [lgd[i] + " TSID", lgd[i] + " Pyb"])
            plt.xlabel("Time [s]")
            plt.ylabel("Angular velocity [rad/s]")
        plt.suptitle("Angular velocities of actuators in TSID and PyBullet")

        print_correlation = False
        if not print_correlation:
            return 0

        R = np.zeros((12, 12))

        plt.figure()
        lgd = ["Pos CoM X [m]", "Pos CoM Y [m]", "Pos CoM Z [m]",
               "Roll [rad]", "Pitch [rad]", "Yaw [rad]",
               "Vel CoM X [m]", "Vel CoM Y [m]", "Vel CoM Z [m]",
               "Ang Vel Roll", "Ang Vel Pitch", "Ang Vel Yaw"]
        for i in range(12):
            for j in range(12):
                plt.subplot(12, 12, (11-i) * 12 + j + 1)
                if i < 3:
                    x1 = self.lC[i, :]
                    x2 = self.lC_pyb[i, :]
                    if i == 2:
                        x1[0] = x1[1]
                elif i < 6:
                    x1 = self.RPY[i-3, :]
                    x2 = self.RPY_pyb[i-3, :]
                elif i < 9:
                    x1 = self.lV[i-6, :]
                    x2 = self.lV_pyb[i-6, :]
                else:
                    x1 = self.lW[i-9, :]
                    x2 = self.lW_pyb[i-9, :]
                if j < 3:
                    y1 = self.lC[j, :]
                    y2 = self.lC_pyb[j, :]
                    if j == 2:
                        y1[0] = y1[1]
                elif j < 6:
                    y1 = self.RPY[j-3, :]
                    y2 = self.RPY_pyb[j-3, :]
                elif j < 9:
                    y1 = self.lV[j-6, :]
                    y2 = self.lV_pyb[j-6, :]
                else:
                    y1 = self.lW[j-9, :]
                    y2 = self.lW_pyb[j-9, :]

                plt.plot(y1, x1, color="b", linestyle='', marker='*', markersize=6)
                plt.tick_params(
                    axis='both',        # changes apply to the x-axis
                    which='both',       # both major and minor ticks are affected
                    bottom=False,       # ticks along the bottom edge are off
                    left=False,         # ticks along the top edge are off
                    labelbottom=False,  # labels along the bottom edge are off
                    labelleft=False)
                if i == 0:
                    plt.xlabel(lgd[j])
                if j == 0:
                    plt.ylabel(lgd[i])

                # slope, intercept, r_value, p_value, std_err = scipystats.linregress(x1, y1)

                R[i, j] = r_value
        plt.suptitle("Correlation of state variables")

        cmap = cm.get_cmap('gist_heat', 256)
        fig = plt.figure()
        ax = fig.gca()
        psm = ax.pcolormesh(R, cmap=cmap, rasterized=True, vmin=0, vmax=1)
        fig.colorbar(psm, ax=ax)
        plt.suptitle("R coefficient of a first order regression")

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

    def log_forces(self, k, interface, tsid_controller, robotId, planeId):
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
            self.forces_mpc[3*f:(3*(f+1)), k:(k+1)] = np.reshape((interface.oMl.rotation @
                                                                  tsid_controller.f_applied[3*f:3*(f+1)]).T, (3, 1))

        # Contact forces desired by TSID (world frame)
        for i, j in enumerate(tsid_controller.contacts_order):
            self.forces_tsid[(3*j):(3*(j+1)), k:(k+1)] = np.reshape(tsid_controller.fc[(3*i):(3*(i+1))], (3, 1))

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
            self.forces_pyb[(3*i):(3*(i+1)), k] = - f_tmps[:, i]

        return 0

    def plot_forces(self):
        """ Plot information about contact forces
        """

        index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
        lgd1 = ["Ctct force X", "Ctct force Y", "Ctct force Z"]
        lgd2 = ["FL", "FR", "HL", "HR"]
        plt.figure()
        for i in range(12):
            plt.subplot(3, 4, index[i])

            h1, = plt.plot(self.t_range, self.forces_mpc[i, :], "r", linewidth=5)
            h2, = plt.plot(self.t_range, self.forces_tsid[i, :], "b", linewidth=3)
            h3, = plt.plot(self.t_range, self.forces_pyb[i, :], "g", linewidth=3, linestyle="--")

            plt.xlabel("Time [s]")
            plt.ylabel(lgd1[i % 3]+" "+lgd2[int(i/3)])

            plt.legend([h1, h2, h3], [lgd1[i % 3]+" "+lgd2[int(i/3)], lgd1[i % 3]+" "+lgd2[int(i/3)],
                                      lgd1[i % 3]+" "+lgd2[int(i/3)]])

            if (i % 3) == 2:
                plt.ylim([-1.0, 20.0])
            else:
                plt.ylim([-8.0, 8.0])

        plt.suptitle("MPC, TSID and PyBullet contact forces (world frame)")

        plt.figure()
        plt.plot(self.t_range, np.sum(self.forces_mpc[2::3, :], axis=0), "r", linewidth=5)
        plt.plot(self.t_range, np.sum(self.forces_tsid[2::3, :], axis=0), "b", linewidth=3)
        plt.plot(self.t_range, np.sum(self.forces_pyb[2::3, :], axis=0), "g", linewidth=3, linestyle="--")
        plt.plot(self.t_range, 2.5*9.81*np.ones((len(self.t_range))), "k", linewidth=5)
        plt.suptitle("Total vertical contact force considering all contacts")

        index = [1, 2, 3, 4]
        lgd = ["FL", "FR", "HL", "HR"]
        plt.figure()
        for i in range(4):
            plt.subplot(2, 2, index[i])

            ctct_status = np.sum(np.abs(self.forces_mpc[(3*i):(3*(i+1)), :]), axis=0) > 0.05
            ctct_mismatch = np.logical_xor((np.sum(np.abs(self.forces_tsid[(3*i):(3*(i+1)), :]), axis=0) != 0),
                                           (np.sum(np.abs(self.forces_pyb[(3*i):(3*(i+1)), :]), axis=0) != 0))

            plt.plot(self.t_range, ctct_status, "r", linewidth=3)
            plt.plot(self.t_range, ctct_mismatch, "k", linewidth=3)

            plt.xlabel("Time [s]")
            plt.ylabel("Contact status mismatch (True/False)")
            plt.legend(["MPC Contact Status " + lgd[i], "Mismatch " + lgd[i]])
        plt.suptitle("Contact status mismatch between MPC/TSID and PyBullet")

        return 0

    def log_torques(self, k, tsid_controller):
        """ Store information about torques
        """

        self.torques_ff[:, k:(k+1)] = np.reshape(tsid_controller.tau_ff, (12, 1))
        self.torques_pd[:, k:(k+1)] = np.reshape(tsid_controller.tau_pd, (12, 1))
        self.torques_sent[:, k:(k+1)] = np.reshape(tsid_controller.torques12, (12, 1))

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
            h1, = plt.plot(self.t_range, self.torques_ff[i, :], color="b", linewidth=3)
            h2, = plt.plot(self.t_range, self.torques_pd[i, :], color="r", linewidth=3)
            h3, = plt.plot(self.t_range, self.torques_sent[i, :], color="g", linewidth=3)
            plt.legend([h1, h2, h3], [lgd[i] + " FF", lgd[i] + " PD", lgd[i]+" Sent"])
            plt.xlabel("Time [s]")
            plt.ylabel("Torque [Nm]")
        plt.suptitle("Feedforward, PD and sent torques (output of PD+ with saturation)")

        return 0

    def log_cost_function(self, k, mpc_wrapper):
        """ Store information about the cost function of the mpc
        """

        # Cost of each component of the cost function over the prediction horizon (state vector and contact forces)
        cost = (np.diag(mpc_wrapper.mpc.x) @ np.diag(mpc_wrapper.mpc.P.data)
                ) @ np.array([mpc_wrapper.mpc.x]).transpose()

        # Sum components of the state vector
        for i in range(12):
            self.cost_components[i, k:(k+1)] = np.sum(cost[i:(12*mpc_wrapper.mpc.n_steps):12])

        # Sum components of the contact forces
        self.cost_components[12, k:(k+1)] = np.sum(cost[(12*mpc_wrapper.mpc.n_steps):])

        """if k % 50 == 0:
            print(np.sum(np.power(mpc_wrapper.solver.mpc.x[6:(12*mpc_wrapper.solver.mpc.n_steps):12], 2)))

        absc = np.array([i for i in range(16)])
        if k == 0:
            plt.figure()
        if k % 100 == 0:
            plt.plot(absc+k, np.power(mpc_wrapper.solver.mpc.x[6:(12*mpc_wrapper.solver.mpc.n_steps):12], 2))
        if k == 5999 == 0:
            plt.show(block=True)"""
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
            elif i <= 12:
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
        if self.type_MPC:
            self.pred_trajectories[:, :, int(k/self.k_mpc)] = mpc_wrapper.mpc.x_robot
            self.pred_forces[:, :, int(k/self.k_mpc)] = mpc_wrapper.mpc.x[mpc_wrapper.mpc.xref.shape[0]*(mpc_wrapper.mpc.xref.shape[1]-1):].reshape((mpc_wrapper.mpc.xref.shape[0],
                                                                                                                                                     mpc_wrapper.mpc.xref.shape[1]-1),
                                                                                                                                                    order='F')
        else:

            self.pred_trajectories[:, :, int(k/self.k_mpc)] = mpc_wrapper.mpc.get_xrobot()
            self.pred_forces[:, :, int(k/self.k_mpc)] = mpc_wrapper.mpc.get_fpredicted()

        return 0

    def log_fstep_planner(self, k, fstep_planner):

        self.fsteps[:, :, int(k/self.k_mpc)] = fstep_planner.fsteps
        self.xref[:, :, int(k/self.k_mpc)] = fstep_planner.xref

        return 0

    def plot_fstep_planner(self):
        """ Plot information about the footstep planner
        """

        self.fsteps[np.isnan(self.fsteps)] = 0.0
        index = [1, 3, 2, 4]
        lgd = ["X FL", "Y FR", "X FR", "Y FR"]
        plt.figure()
        for i in range(4):
            plt.subplot(2, 2, index[i])
            if i < 2:
                plt.plot(self.t_range[::int(self.dt_mpc/self.dt)],
                         self.fsteps[0, i+1, :], color='k', linewidth=2, marker='x')
            else:
                plt.plot(self.t_range[::int(self.dt_mpc/self.dt)],
                         self.fsteps[0, i-2+3+1, :], color='k', linewidth=2, marker='x')

            plt.xlabel("Time [s]")
            plt.ylabel(lgd[i])
        plt.suptitle("Foostep planner output for FL and FR feet")

    def plot_predicted_trajectories(self):
        """ Plot information about the predicted evolution of the optimization vector components
        """

        t_pred = np.array([(k)*self.dt_mpc for k in range(np.int(self.T/self.dt_mpc))])

        #index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
        index = [1, 4, 7, 2, 5, 8, 3, 6, 9]

        lgd = ["Position Z", "Position Roll", "Position Pitch", "Linear vel X", "Linear vel Y", "Linear vel Z",
               "Angular vel Roll", "Angular vel Pitch", "Angular vel Yaw"]
        plt.figure()
        for i, o in enumerate([2, 3, 4, 6, 7, 8, 9, 10, 11]):
            plt.subplot(3, 3, index[i])
            for j in range(self.pred_trajectories.shape[2]):
                if (j*self.k_mpc > self.k_max_loop):
                    break
                # if (j % 1) == 0:
                #h, = plt.plot(t_pred[0:15] + j*self.dt_mpc, self.pred_trajectories[o, 0:15, j], linewidth=2, marker='x')
                h, = plt.plot(t_pred[0:16] + j*self.dt_mpc, np.hstack(([self.lW[1, j*self.k_mpc]],
                                                                       self.pred_trajectories[o, 0:15, j])), linewidth=2, marker='x')
            #h, = plt.plot(self.t_range[::20], self.state_ref[i, ::20], "r", linewidth=3, marker='*')
            if i == 0:
                plt.plot(self.t_range[::20], self.lC[2, ::20], "r", linewidth=2)  # , marker='o', linestyle="--")
            elif i <= 2:
                plt.plot(self.t_range[::20], self.RPY[i-1, ::20], "r", linewidth=2)  # , marker='o', linestyle="--")
            elif i <= 5:
                plt.plot(self.t_range[::20], self.lV[i-3, ::20], "r", linewidth=2)  # , marker='o', linestyle="--")
            else:
                plt.plot(self.t_range[::20], self.lW[i-6, ::20], "r", linewidth=2)  # , marker='o', linestyle="--")
            plt.ylabel(lgd[i])
        plt.suptitle("Predicted trajectories (local frame)")

        return 0

    def log_tracking_foot(self, k, tsid_controller, solo):
        """ Store information about one of the foot tracking task
        """

        self.pos[:, k:(k+1)] = np.reshape(tsid_controller.feetTask[1].position, (12, 1))
        self.pos_ref[:, k:(k+1)] = np.reshape(tsid_controller.feetTask[1].position_ref, (12, 1))
        self.pos_err[:, k:(k+1)] = np.reshape(tsid_controller.feetTask[1].position_error, (3, 1))
        self.vel[0:3, k:(k+1)] = np.reshape((solo.data.oMi[solo.model.frames[18].parent].act(solo.model.frames[18].placement)
                                             ).rotation @ tsid_controller.feetTask[1].velocity[0:3], (3, 1))
        self.vel[3:6, k:(k+1)] = np.reshape((solo.data.oMi[solo.model.frames[18].parent].act(solo.model.frames[18].placement)
                                             ).rotation @ tsid_controller.feetTask[1].velocity[3:6], (3, 1))
        self.vel_ref[:, k:(k+1)] = np.reshape(tsid_controller.feetTask[1].velocity_ref, (6, 1))
        self.vel_err[:, k:(k+1)] = np.reshape(tsid_controller.feetTask[1].velocity_error, (3, 1))

        return 0

    def plot_tracking_foot(self):
        """ Plot information about one of the foot tracking task
        """

        index = [1, 3, 5, 2, 4, 6]
        lgd = ["X", "Y", "Z", "$\dot X$", "$\dot Y$", "$\dot Z$"]
        plt.figure()
        for i in range(6):
            plt.subplot(3, 2, index[i])
            if i < 3:
                plt.plot(self.pos[i, :], color='r', linewidth=2, marker='x')
                plt.plot(self.pos_ref[i, :], color='g', linewidth=2, marker='x')
                plt.plot(self.feet_pos_target[i, 1, :], color='b', linewidth=2, marker='x')
                #plt.plot(self.pos_err[i, :], color='b', linewidth=2, marker='x')
                plt.legend(["Pos", "Pos ref", "Pos target"])
            else:
                plt.plot(self.vel[i-3, :], color='r', linewidth=2, marker='x')
                plt.plot(self.vel_ref[i-3, :], color='g', linewidth=2, marker='x')
                #plt.plot(self.vel_err[i-3, :], color='b', linewidth=2, marker='x')
                plt.plot(self.feet_vel_target[i-3, 1, :], color='b', linewidth=2, marker='x')
                plt.legend(["Vel", "Vel ref", "Vel target"])

            plt.ylabel(lgd[i])
        plt.suptitle("Tracking FR foot")

        return 0

    def call_log_functions(self, k, pyb_sim, joystick, fstep_planner, interface, mpc_wrapper, tsid_controller, enable_multiprocessing, robotId, planeId, solo):
        """ Call logging functions of the Logger class
        """

        # Store current and desired position, velocity and acceleration of feet over time
        self.log_footsteps(k, interface, tsid_controller)

        # Store information about the state of the robot
        if not enable_multiprocessing:
            self.log_state(k, pyb_sim, joystick, interface, mpc_wrapper, solo)

        # Store information about contact forces
        self.log_forces(k, interface, tsid_controller, robotId, planeId)

        # Store information about torques
        self.log_torques(k, tsid_controller)

        # Store information about the cost function
        """if self.type_MPC and not enable_multiprocessing:
            self.log_cost_function(k, mpc_wrapper)"""

        # Store information about the predicted evolution of the optimization vector components
        if not enable_multiprocessing and ((k % self.k_mpc) == 0):
            # self.log_predicted_trajectories(k, mpc_wrapper)
            self.log_fstep_planner(k, fstep_planner)

        # Store information about one of the foot tracking task
        self.log_tracking_foot(k, tsid_controller, solo)

        return 0

    def plot_graphs(self, enable_multiprocessing, show_block=True):

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
        # Cost not comparable between the two solvers
        if self.type_MPC and not enable_multiprocessing:
            self.plot_cost_function()

        # Plot information about the predicted evolution of the optimization vector components
        # if not enable_multiprocessing:
        #    self.plot_predicted_trajectories()

        # Plot information about one of the foot tracking task
        # self.plot_tracking_foot()

        # Display graphs
        plt.show(block=show_block)

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
