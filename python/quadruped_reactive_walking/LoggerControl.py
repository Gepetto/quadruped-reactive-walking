'''This class will log 1d array in Nd matrix from device and qualisys object'''
import numpy as np
from datetime import datetime as datetime
from time import time
from utils_mpc import quaternionToRPY


class LoggerControl():
    def __init__(self, dt, joystick=None, estimator=None, loop=None, planner=None, logSize=60e3, ringBuffer=False):
        self.ringBuffer = ringBuffer
        logSize = np.int(logSize)
        self.logSize = logSize
        self.i = 0

        self.dt = dt

        # Allocate the data:
        # Joystick
        self.joy_v_ref = np.zeros([logSize, 6])  # reference velocity of the joystick

        # Estimator
        self.esti_feet_status = np.zeros([logSize, 4])  # input feet status (contact or not)
        self.esti_feet_goals = np.zeros([logSize, 3, 4])  # input feet goals (desired on the ground)
        self.esti_q_filt = np.zeros([logSize, 19])  # output position
        self.esti_v_filt = np.zeros([logSize, 18])  # output velocity
        self.esti_v_secu = np.zeros([logSize, 12])  # filtered output velocity for security check

        self.esti_FK_lin_vel = np.zeros([logSize, 3])  # estimated velocity of the base with FK
        self.esti_FK_xyz = np.zeros([logSize, 3])  # estimated position of the base with FK
        self.esti_xyz_mean_feet = np.zeros([logSize, 3])  # average of feet goals

        self.esti_HP_x = np.zeros([logSize, 3])  # x input of the velocity complementary filter
        self.esti_HP_dx = np.zeros([logSize, 3])  # dx input of the velocity complementary filter
        self.esti_HP_alpha = np.zeros([logSize, 3])  # alpha parameter of the velocity complementary filter
        self.esti_HP_filt_x = np.zeros([logSize, 3])  # filtered output of the velocity complementary filter

        self.esti_LP_x = np.zeros([logSize, 3])  # x input of the position complementary filter
        self.esti_LP_dx = np.zeros([logSize, 3])  # dx input of the position complementary filter
        self.esti_LP_alpha = np.zeros([logSize, 3])  # alpha parameter of the position complementary filter
        self.esti_LP_filt_x = np.zeros([logSize, 3])  # filtered output of the position complementary filter

        self.esti_kf_X = np.zeros([logSize, 18])  # state of the Kalman filter
        self.esti_kf_Z = np.zeros([logSize, 16])  # measurement for the Kalman filter

        # Loop
        self.loop_o_q_int = np.zeros([logSize, 19])  # position in world frame (esti_q_filt + dt * loop_o_v)
        self.loop_o_v = np.zeros([logSize, 18])  # estimated velocity in world frame

        # Planner
        self.planner_q_static = np.zeros([logSize, 19])  # position in static mode (4 stance phase)
        self.planner_RPY_static = np.zeros([logSize, 3])  # RPY orientation in static mode (4 stance phase)
        self.planner_xref = np.zeros([logSize, 12, 1+planner.n_steps])  # Reference trajectory
        self.planner_fsteps = np.zeros([logSize, planner.gait.shape[0], 13])  # Reference footsteps position
        self.planner_gait = np.zeros([logSize, 20, 5])  # Gait sequence
        self.planner_goals = np.zeros([logSize, 3, 4])  # 3D target feet positions
        self.planner_vgoals = np.zeros([logSize, 3, 4])  # 3D target feet velocities
        self.planner_agoals = np.zeros([logSize, 3, 4])  # 3D target feet accelerations
        self.planner_is_static = np.zeros([logSize])  # if the planner is in static mode or not
        self.planner_h_ref = np.zeros([logSize])  # reference height of the planner

        # Model Predictive Control
        # output vector of the MPC (next state + reference contact force)
        self.mpc_x_f = np.zeros([logSize, 24, planner.n_steps])

        # Whole body control
        self.wbc_x_f = np.zeros([logSize, 24])  # input vector of the WBC (next state + reference contact force)
        self.wbc_P = np.zeros([logSize, 12])  # proportionnal gains of the PD+
        self.wbc_D = np.zeros([logSize, 12])  # derivative gains of the PD+
        self.wbc_q_des = np.zeros([logSize, 12])  # desired position of actuators
        self.wbc_v_des = np.zeros([logSize, 12])  # desired velocity of actuators
        self.wbc_tau_ff = np.zeros([logSize, 12])  # feedforward torques computed by the WBC
        self.wbc_f_ctc = np.zeros([logSize, 12])  # contact forces computed by the WBC
        self.wbc_feet_pos = np.zeros([logSize, 3, 4])  # current feet positions according to WBC
        self.wbc_feet_err = np.zeros([logSize, 3, 4])  # error between feet positions and their reference
        self.wbc_feet_vel = np.zeros([logSize, 3, 4])  # current feet velocities according to WBC
        self.wbc_feet_pos_invkin = np.zeros([logSize, 3, 4])  # current feet positions according to InvKin
        self.wbc_feet_vel_invkin = np.zeros([logSize, 3, 4])  # current feet velocities according to InvKin

        # Timestamps
        self.tstamps = np.zeros(logSize)

    def sample(self, joystick, estimator, loop, planner, wbc):
        if (self.i >= self.logSize):
            if self.ringBuffer:
                self.i = 0
            else:
                return

        # Logging from joystick
        self.joy_v_ref[self.i] = joystick.v_ref[:, 0]

        # Logging from estimator
        self.esti_feet_status[self.i] = estimator.feet_status[:]
        self.esti_feet_goals[self.i] = estimator.feet_goals
        self.esti_q_filt[self.i] = estimator.q_filt[:, 0]
        self.esti_v_filt[self.i] = estimator.v_filt[:, 0]
        self.esti_v_secu[self.i] = estimator.v_secu[:]

        self.esti_FK_lin_vel[self.i] = estimator.FK_lin_vel[:]
        self.esti_FK_xyz[self.i] = estimator.FK_xyz[:]
        self.esti_xyz_mean_feet[self.i] = estimator.xyz_mean_feet[:]
        if not estimator.kf_enabled:
            self.esti_HP_x[self.i] = estimator.filter_xyz_vel.x
            self.esti_HP_dx[self.i] = estimator.filter_xyz_vel.dx
            self.esti_HP_alpha[self.i] = estimator.filter_xyz_vel.alpha
            self.esti_HP_filt_x[self.i] = estimator.filter_xyz_vel.filt_x

            self.esti_LP_x[self.i] = estimator.filter_xyz_pos.x
            self.esti_LP_dx[self.i] = estimator.filter_xyz_pos.dx
            self.esti_LP_alpha[self.i] = estimator.filter_xyz_pos.alpha
            self.esti_LP_filt_x[self.i] = estimator.filter_xyz_pos.filt_x
        else:
            self.esti_kf_X[self.i] = estimator.kf.X[:, 0]
            self.esti_kf_Z[self.i] = estimator.Z[:, 0]

        # Logging from the main loop
        self.loop_o_q_int[self.i] = loop.q_estim[:, 0]
        self.loop_o_v[self.i] = loop.v_estim[:, 0]

        # Logging from the planner
        self.planner_q_static[self.i] = planner.q_static[:, 0]
        self.planner_RPY_static[self.i] = planner.RPY_static[:, 0]
        self.planner_xref[self.i] = planner.xref
        self.planner_fsteps[self.i] = planner.fsteps
        self.planner_gait[self.i] = planner.gait
        self.planner_goals[self.i] = planner.goals
        self.planner_vgoals[self.i] = planner.vgoals
        self.planner_agoals[self.i] = planner.agoals
        self.planner_is_static[self.i] = planner.is_static
        self.planner_h_ref[self.i] = planner.h_ref

        # Logging from model predictive control
        self.mpc_x_f[self.i] = loop.x_f_mpc

        # Logging from whole body control
        self.wbc_x_f[self.i] = loop.x_f_wbc
        self.wbc_P[self.i] = loop.result.P
        self.wbc_D[self.i] = loop.result.D
        self.wbc_q_des[self.i] = loop.result.q_des
        self.wbc_v_des[self.i] = loop.result.v_des
        self.wbc_tau_ff[self.i] = loop.result.tau_ff
        self.wbc_f_ctc[self.i] = wbc.f_with_delta[:, 0]
        self.wbc_feet_pos[self.i] = wbc.feet_pos
        self.wbc_feet_err[self.i] = wbc.feet_err
        self.wbc_feet_vel[self.i] = wbc.feet_vel
        self.wbc_feet_pos_invkin[self.i] = wbc.invKin.cpp_posf.transpose()
        self.wbc_feet_vel_invkin[self.i] = wbc.invKin.cpp_vf.transpose()

        # Logging timestamp
        self.tstamps[self.i] = time()

        self.i += 1

    def processMocap(self, N, loggerSensors):

        self.mocap_b_v = np.zeros([N, 3])
        self.mocap_b_w = np.zeros([N, 3])
        self.mocap_RPY = np.zeros([N, 3])

        for i in range(N):
            oRb = loggerSensors.mocapOrientationMat9[i]

            """from IPython import embed
            embed()"""

            self.mocap_b_v[i] = (oRb.transpose() @ loggerSensors.mocapVelocity[i].reshape((3, 1))).ravel()
            self.mocap_b_w[i] = (oRb.transpose() @ loggerSensors.mocapAngularVelocity[i].reshape((3, 1))).ravel()
            self.mocap_RPY[i] = quaternionToRPY(loggerSensors.mocapOrientationQuat[i])[:, 0]

    def plotAll(self, loggerSensors):

        from matplotlib import pyplot as plt

        N = self.tstamps.shape[0]
        t_range = np.array([k*self.dt for k in range(N)])

        self.processMocap(N, loggerSensors)

        index6 = [1, 3, 5, 2, 4, 6]
        index12 = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]

        """plt.figure()
        for i in range(4):
            if i == 0:
                ax0 = plt.subplot(2, 2, i+1)
            else:
                plt.subplot(2, 2, i+1, sharex=ax0)
            switch = np.diff(self.esti_feet_status[:, i])
            tmp = self.wbc_feet_pos[:-1, 2, i]
            tmp_y = tmp[switch > 0]
            tmp_x = t_range[:-1]
            tmp_x = tmp_x[switch > 0]
            plt.plot(tmp_x, tmp_y, linewidth=3)"""

        lgd_X = ["FL", "FR", "HL", "HR"]
        lgd_Y = ["Pos X", "Pos Y", "Pos Z"]
        plt.figure()
        for i in range(12):
            if i == 0:
                ax0 = plt.subplot(3, 4, index12[i])
            else:
                plt.subplot(3, 4, index12[i], sharex=ax0)

            plt.plot(t_range, self.wbc_feet_pos[:, i % 3, np.int(i/3)], color='b', linewidth=3, marker='')
            plt.plot(t_range, self.wbc_feet_err[:, i % 3, np.int(i/3)], color='g', linewidth=3, marker='')
            plt.plot(t_range, self.planner_goals[:, i % 3, np.int(i/3)], color='r', linewidth=3, marker='')
            plt.plot(t_range, self.wbc_feet_pos_invkin[:, i % 3, np.int(i/3)],
                     color='darkviolet', linewidth=3, linestyle="--", marker='')
            if (i % 3) == 2:
                plt.plot(t_range, self.planner_gait[:, 0, 1+np.int(
                    i/3)] * np.max(self.wbc_feet_pos[:, i % 3, np.int(i/3)]), color='k', linewidth=3, marker='')
            plt.legend([lgd_Y[i % 3] + " " + lgd_X[np.int(i/3)]+"", "error",
                        lgd_Y[i % 3] + " " + lgd_X[np.int(i/3)]+" Ref", "Contact state"], prop={'size': 8})
        plt.suptitle("Measured & Reference feet positions (world frame)")

        lgd_X = ["FL", "FR", "HL", "HR"]
        lgd_Y = ["Vel X", "Vel Y", "Vel Z"]
        plt.figure()
        for i in range(12):
            if i == 0:
                ax0 = plt.subplot(3, 4, index12[i])
            else:
                plt.subplot(3, 4, index12[i], sharex=ax0)
            plt.plot(t_range, self.wbc_feet_vel[:, i % 3, np.int(i/3)], color='b', linewidth=3, marker='')
            plt.plot(t_range, self.planner_vgoals[:, i % 3, np.int(i/3)], color='r', linewidth=3, marker='')
            plt.plot(t_range, self.wbc_feet_vel_invkin[:, i % 3, np.int(i/3)],
                     color='darkviolet', linewidth=3, linestyle="--", marker='')
            plt.legend([lgd_Y[i % 3] + " " + lgd_X[np.int(i/3)], lgd_Y[i %
                                                                       3] + " " + lgd_X[np.int(i/3)]+" Ref"], prop={'size': 8})
        plt.suptitle("Measured and Reference feet velocities (world frame)")

        lgd_X = ["FL", "FR", "HL", "HR"]
        lgd_Y = ["Acc X", "Acc Y", "Acc Z"]
        plt.figure()
        for i in range(12):
            if i == 0:
                ax0 = plt.subplot(3, 4, index12[i])
            else:
                plt.subplot(3, 4, index12[i], sharex=ax0)
            plt.plot(t_range, self.planner_agoals[:, i % 3, np.int(i/3)], color='r', linewidth=3, marker='')
            plt.legend([lgd_Y[i % 3] + " " + lgd_X[np.int(i/3)]+" Ref"], prop={'size': 8})
        plt.suptitle("Reference feet accelerations (world frame)")

        # LOG_Q
        lgd = ["Position X", "Position Y", "Position Z", "Position Roll", "Position Pitch", "Position Yaw"]
        plt.figure()
        for i in range(6):
            if i == 0:
                ax0 = plt.subplot(3, 2, index6[i])
            else:
                plt.subplot(3, 2, index6[i], sharex=ax0)
            plt.plot(t_range, self.planner_xref[:, i, 0], "b", linewidth=2)
            plt.plot(t_range, self.planner_xref[:, i, 1], "r", linewidth=3)
            if i < 3:
                plt.plot(t_range, loggerSensors.mocapPosition[:, i], "k", linewidth=3)
            else:
                plt.plot(t_range, self.mocap_RPY[:, i-3], "k", linewidth=3)
            # plt.plot(t_range, self.log_q[i, :], "grey", linewidth=4)
            # plt.plot(t_range[:-2], self.log_x_invkin[i, :-2], "g", linewidth=2)
            # plt.plot(t_range[:-2], self.log_x_ref_invkin[i, :-2], "violet", linewidth=2, linestyle="--")
            plt.legend(["Robot state", "Robot reference state"], prop={'size': 8})
            plt.ylabel(lgd[i])
        plt.suptitle("Measured & Reference position and orientation")

        # LOG_V
        lgd = ["Linear vel X", "Linear vel Y", "Linear vel Z",
               "Angular vel Roll", "Angular vel Pitch", "Angular vel Yaw"]
        plt.figure()
        for i in range(6):
            if i == 0:
                ax0 = plt.subplot(3, 2, index6[i])
            else:
                plt.subplot(3, 2, index6[i], sharex=ax0)
            plt.plot(t_range, self.esti_v_filt[:, i], "b", linewidth=2)
            plt.plot(t_range, self.joy_v_ref[:, i], "r", linewidth=3)
            if i < 3:
                plt.plot(t_range, self.mocap_b_v[:, i], "k", linewidth=3)
                plt.plot(t_range, self.esti_FK_lin_vel[:, i], "violet", linewidth=3, linestyle="--")
            else:
                plt.plot(t_range, self.mocap_b_w[:, i-3], "k", linewidth=3)

            # plt.plot(t_range, self.log_dq[i, :], "g", linewidth=2)
            # plt.plot(t_range[:-2], self.log_dx_invkin[i, :-2], "g", linewidth=2)
            # plt.plot(t_range[:-2], self.log_dx_ref_invkin[i, :-2], "violet", linewidth=2, linestyle="--")
            plt.legend(["Robot state", "Robot reference state"], prop={'size': 8})
            plt.ylabel(lgd[i])
        plt.suptitle("Measured & Reference linear and angular velocities")

        """plt.figure()
        plt.plot(t_range[:-2], self.log_x[6, :-2], "b", linewidth=2)
        plt.plot(t_range[:-2], self.log_x_cmd[6, :-2], "r", linewidth=2)
        plt.plot(t_range[:-2], self.log_dx_invkin[0, :-2], "g", linewidth=2)
        plt.plot(t_range[:-2], self.log_dx_ref_invkin[0, :-2], "violet", linewidth=2)
        plt.legend(["WBC integrated output state", "Robot reference state",
                    "Task current state", "Task reference state"])"""

        lgd1 = ["HAA", "HFE", "Knee"]
        lgd2 = ["FL", "FR", "HL", "HR"]
        plt.figure()
        for i in range(12):
            if i == 0:
                ax0 = plt.subplot(3, 4, index12[i])
            else:
                plt.subplot(3, 4, index12[i], sharex=ax0)
            tau_fb = self.wbc_P[:, i] * (self.wbc_q_des[:, i] - self.esti_q_filt[:, 7+i]) + \
                self.wbc_D[:, i] * (self.wbc_v_des[:, i] - self.esti_v_filt[:, 6+i])
            h1, = plt.plot(t_range, self.wbc_tau_ff[:, i], "r", linewidth=3)
            h2, = plt.plot(t_range, tau_fb, "b", linewidth=3)
            h3, = plt.plot(t_range, self.wbc_tau_ff[:, i] + tau_fb, "g", linewidth=3)
            h4, = plt.plot(t_range[:-1], loggerSensors.torquesFromCurrentMeasurment[1:, i],
                           "violet", linewidth=3, linestyle="--")
            plt.xlabel("Time [s]")
            plt.ylabel(lgd1[i % 3]+" "+lgd2[int(i/3)]+" [Nm]")
            tmp = lgd1[i % 3]+" "+lgd2[int(i/3)]
            plt.legend([h1, h2, h3, h4], ["FF "+tmp, "FB "+tmp, "PD+ "+tmp, "Meas "+tmp], prop={'size': 8})
            plt.ylim([-8.0, 8.0])
        plt.suptitle("FF torques & FB torques & Sent torques & Meas torques")

        lgd1 = ["Ctct force X", "Ctct force Y", "Ctct force Z"]
        lgd2 = ["FL", "FR", "HL", "HR"]
        plt.figure()
        for i in range(12):
            if i == 0:
                ax0 = plt.subplot(3, 4, index12[i])
            else:
                plt.subplot(3, 4, index12[i], sharex=ax0)
            h1, = plt.plot(t_range, self.mpc_x_f[:, 12+i, 0], "r", linewidth=3)
            h2, = plt.plot(t_range, self.wbc_f_ctc[:, i], "b", linewidth=3, linestyle="--")
            plt.xlabel("Time [s]")
            plt.ylabel(lgd1[i % 3]+" "+lgd2[int(i/3)]+" [N]")
            plt.legend([h1, h2], ["MPC " + lgd1[i % 3]+" "+lgd2[int(i/3)],
                                  "WBC " + lgd1[i % 3]+" "+lgd2[int(i/3)]], prop={'size': 8})
            if (i % 3) == 2:
                plt.ylim([-0.0, 26.0])
            else:
                plt.ylim([-26.0, 26.0])
        plt.suptitle("Contact forces (MPC command) & WBC QP output")

        lgd1 = ["HAA", "HFE", "Knee"]
        lgd2 = ["FL", "FR", "HL", "HR"]
        plt.figure()
        for i in range(12):
            if i == 0:
                ax0 = plt.subplot(3, 4, index12[i])
            else:
                plt.subplot(3, 4, index12[i], sharex=ax0)
            h1, = plt.plot(t_range, self.wbc_q_des[:, i], color='r', linewidth=3)
            h2, = plt.plot(t_range, self.esti_q_filt[:, 7+i], color='b', linewidth=3)
            plt.xlabel("Time [s]")
            plt.ylabel(lgd1[i % 3]+" "+lgd2[int(i/3)]+" [rad]")
            plt.legend([h1, h2], ["Ref "+lgd1[i % 3]+" "+lgd2[int(i/3)],
                                  lgd1[i % 3]+" "+lgd2[int(i/3)]], prop={'size': 8})
        plt.suptitle("Desired actuator positions & Measured actuator positions")

        # Evolution of predicted trajectory along time
        log_t_pred = np.array([k*self.dt*10 for k in range(self.mpc_x_f.shape[2])])
        log_t_ref = np.array([k*self.dt*10 for k in range(self.planner_xref.shape[2])])

        """from IPython import embed
        embed()"""

        titles = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
        step = 2000
        plt.figure()
        for j in range(6):
            plt.subplot(3, 2, index6[j])
            c = [[i/(self.mpc_x_f.shape[0]+5), 0.0, i/(self.mpc_x_f.shape[0]+5)]
                 for i in range(0, self.mpc_x_f.shape[0], step)]
            for i in range(0, self.mpc_x_f.shape[0], step):
                h1, = plt.plot(log_t_pred+(i+10)*self.dt,
                               self.mpc_x_f[i, j, :], "b", linewidth=2, color=c[int(i/step)])
                h2, = plt.plot(log_t_ref+i*self.dt,
                               self.planner_xref[i, j, :], linestyle="--", marker='x', color="g", linewidth=2)
            h3, = plt.plot(np.array([k*self.dt for k in range(self.mpc_x_f.shape[0])]),
                           self.planner_xref[:, j, 0], linestyle=None, marker='x', color="r", linewidth=1)
            plt.xlabel("Time [s]")
            plt.legend([h1, h2, h3], ["Output trajectory of MPC",
                                      "Input trajectory of planner", "Actual robot trajectory"])
            plt.title("Predicted trajectory for " + titles[j])
        plt.suptitle("Analysis of trajectories in position and orientation computed by the MPC")

        plt.figure()
        for j in range(6):
            plt.subplot(3, 2, index6[j])
            c = [[i/(self.mpc_x_f.shape[0]+5), 0.0, i/(self.mpc_x_f.shape[0]+5)]
                 for i in range(0, self.mpc_x_f.shape[0], step)]
            for i in range(0, self.mpc_x_f.shape[0], step):
                h1, = plt.plot(log_t_pred+(i+10)*self.dt,
                               self.mpc_x_f[i, j+6, :], "b", linewidth=2, color=c[int(i/step)])
                h2, = plt.plot(log_t_ref+i*self.dt,
                               self.planner_xref[i, j+6, :], linestyle="--", marker='x', color="g", linewidth=2)
            h3, = plt.plot(np.array([k*self.dt for k in range(self.mpc_x_f.shape[0])]),
                           self.planner_xref[:, j+6, 0], linestyle=None, marker='x', color="r", linewidth=1)
            plt.xlabel("Time [s]")
            plt.legend([h1, h2, h3], ["Output trajectory of MPC",
                                      "Input trajectory of planner", "Actual robot trajectory"])
            plt.title("Predicted trajectory for velocity in " + titles[j])
        plt.suptitle("Analysis of trajectories of linear and angular velocities computed by the MPC")

        step = 1000
        lgd1 = ["Ctct force X", "Ctct force Y", "Ctct force Z"]
        lgd2 = ["FL", "FR", "HL", "HR"]
        plt.figure()
        for i in range(12):
            if i == 0:
                ax0 = plt.subplot(3, 4, index12[i])
            else:
                plt.subplot(3, 4, index12[i], sharex=ax0)
            h1, = plt.plot(t_range, self.mpc_x_f[:, 12+i, 0], "r", linewidth=3)
            h2, = plt.plot(t_range, self.wbc_f_ctc[:, i], "b", linewidth=3, linestyle="--")
            plt.xlabel("Time [s]")
            plt.ylabel(lgd1[i % 3]+" "+lgd2[int(i/3)]+" [N]")
            plt.legend([h1, h2], ["MPC " + lgd1[i % 3]+" "+lgd2[int(i/3)],
                                  "WBC " + lgd1[i % 3]+" "+lgd2[int(i/3)]], prop={'size': 8})
            if (i % 3) == 2:
                plt.ylim([-0.0, 26.0])
            else:
                plt.ylim([-26.0, 26.0])
        plt.suptitle("Contact forces (MPC command) & WBC QP output")

        lgd1 = ["Ctct force X", "Ctct force Y", "Ctct force Z"]
        lgd2 = ["FL", "FR", "HL", "HR"]
        plt.figure()
        for i in range(4):
            if i == 0:
                ax0 = plt.subplot(1, 4, i+1)
            else:
                plt.subplot(1, 4, i+1, sharex=ax0)

            for k in range(0, self.mpc_x_f.shape[0], step):
                h2, = plt.plot(log_t_pred+k*self.dt, self.mpc_x_f[k, 12+(3*i+2), :], linestyle="--", marker='x', linewidth=2)
            h1, = plt.plot(t_range, self.mpc_x_f[:, 12+(3*i+2), 0], "r", linewidth=3)
            # h3, = plt.plot(t_range, self.wbc_f_ctc[:, i], "b", linewidth=3, linestyle="--")
            plt.plot(t_range, self.esti_feet_status[:, i], "k", linestyle="--")
            plt.xlabel("Time [s]")
            plt.ylabel(lgd2[i]+" [N]")
            plt.legend([h1, h2], ["MPC "+lgd2[i],
                                  "MPC "+lgd2[i]+" trajectory"])
            plt.ylim([-1.0, 26.0])
        plt.suptitle("Contact forces trajectories & Actual forces trajectories")

        # Analysis of the complementary filter behaviour
        clr = ["b", "darkred", "forestgreen"]
        # Velocity complementary filter
        lgd_Y = ["dx", "ddx", "alpha dx", "dx_out", "dy", "ddy", "alpha dy", "dy_out", "dz", "ddz", "alpha dz", "dz_out"]
        plt.figure()
        for i in range(12):
            if i == 0:
                ax0 = plt.subplot(3, 4, i+1)
            else:
                plt.subplot(3, 4, i+1, sharex=ax0)
            if i % 4 == 0:
                plt.plot(t_range, self.esti_HP_x[:, int(i/4)], color=clr[int(i/4)], linewidth=3, marker='') # x input of the velocity complementary filter
            elif i % 4 == 1:
                plt.plot(t_range, self.esti_HP_dx[:, int(i/4)], color=clr[int(i/4)], linewidth=3, marker='') # dx input of the velocity complementary filter
            elif i % 4 == 2:
                plt.plot(t_range, self.esti_HP_alpha[:, int(i/4)], color=clr[int(i/4)], linewidth=3, marker='') # alpha parameter of the velocity complementary filter
            else:
                plt.plot(t_range, self.esti_HP_filt_x[:, int(i/4)], color=clr[int(i/4)], linewidth=3, marker='') # filtered output of the velocity complementary filter
            
            plt.legend([lgd_Y[i]], prop={'size': 8})
        plt.suptitle("Evolution of the quantities of the velocity complementary filter")

        # Position complementary filter
        lgd_Y = ["x", "dx", "alpha x", "x_out", "y", "dy", "alpha y", "y_out", "z", "dz", "alpha z", "z_out"]
        plt.figure()
        for i in range(12):
            if i == 0:
                ax0 = plt.subplot(3, 4, i+1)
            else:
                plt.subplot(3, 4, i+1, sharex=ax0)
            if i % 4 == 0:
                plt.plot(t_range, self.esti_LP_x[:, int(i/4)], color=clr[int(i/4)], linewidth=3, marker='') # x input of the position complementary filter
            elif i % 4 == 1:
                plt.plot(t_range, self.esti_LP_dx[:, int(i/4)], color=clr[int(i/4)], linewidth=3, marker='') # dx input of the position complementary filter
            elif i % 4 == 2:
                plt.plot(t_range, self.esti_LP_alpha[:, int(i/4)], color=clr[int(i/4)], linewidth=3, marker='') # alpha parameter of the position complementary filter
            else:
                plt.plot(t_range, self.esti_LP_filt_x[:, int(i/4)], color=clr[int(i/4)], linewidth=3, marker='') # filtered output of the position complementary filter
            
            plt.legend([lgd_Y[i]], prop={'size': 8})
        plt.suptitle("Evolution of the quantities of the position complementary filter")

        plt.show(block=True)

        from IPython import embed
        embed()

    def saveAll(self, loggerSensors, fileName="data"):
        date_str = datetime.now().strftime('_%Y_%m_%d_%H_%M')

        np.savez(fileName + date_str + ".npz",

                 joy_v_ref=self.joy_v_ref,

                 esti_feet_status=self.esti_feet_status,
                 esti_feet_goals=self.esti_feet_goals,
                 esti_q_filt=self.esti_q_filt,
                 esti_v_filt=self.esti_v_filt,
                 esti_v_secu=self.esti_v_secu,

                 esti_FK_lin_vel=self.esti_FK_lin_vel,
                 esti_FK_xyz=self.esti_FK_xyz,
                 esti_xyz_mean_feet=self.esti_xyz_mean_feet,

                 esti_HP_x=self.esti_HP_x,
                 esti_HP_dx=self.esti_HP_dx,
                 esti_HP_alpha=self.esti_HP_alpha,
                 esti_HP_filt_x=self.esti_HP_filt_x,

                 esti_LP_x=self.esti_LP_x,
                 esti_LP_dx=self.esti_LP_dx,
                 esti_LP_alpha=self.esti_LP_alpha,
                 esti_LP_filt_x=self.esti_LP_filt_x,

                 esti_kf_X=self.esti_kf_X,
                 esti_kf_Z=self.esti_kf_Z,

                 loop_o_q_int=self.loop_o_q_int,
                 loop_o_v=self.loop_o_v,

                 planner_q_static=self.planner_q_static,
                 planner_RPY_static=self.planner_RPY_static,
                 planner_xref=self.planner_xref,
                 planner_fsteps=self.planner_fsteps,
                 planner_gait=self.planner_gait,
                 planner_goals=self.planner_goals,
                 planner_vgoals=self.planner_vgoals,
                 planner_agoals=self.planner_agoals,
                 planner_is_static=self.planner_is_static,
                 planner_h_ref=self.planner_h_ref,

                 mpc_x_f=self.mpc_x_f,

                 wbc_x_f=self.wbc_x_f,
                 wbc_P=self.wbc_P,
                 wbc_D=self.wbc_D,
                 wbc_q_des=self.wbc_q_des,
                 wbc_v_des=self.wbc_v_des,
                 wbc_tau_ff=self.wbc_tau_ff,
                 wbc_f_ctc=self.wbc_f_ctc,
                 wbc_feet_pos=self.wbc_feet_pos,
                 wbc_feet_err=self.wbc_feet_err,
                 wbc_feet_vel=self.wbc_feet_vel,

                 tstamps=self.tstamps,

                 q_mes=loggerSensors.q_mes,
                 v_mes=loggerSensors.v_mes,
                 baseOrientation=loggerSensors.baseOrientation,
                 baseAngularVelocity=loggerSensors.baseAngularVelocity,
                 baseLinearAcceleration=loggerSensors.baseLinearAcceleration,
                 baseAccelerometer=loggerSensors.baseAccelerometer,
                 torquesFromCurrentMeasurment=loggerSensors.torquesFromCurrentMeasurment,
                 mocapPosition=loggerSensors.mocapPosition,
                 mocapVelocity=loggerSensors.mocapVelocity,
                 mocapAngularVelocity=loggerSensors.mocapAngularVelocity,
                 mocapOrientationMat9=loggerSensors.mocapOrientationMat9,
                 mocapOrientationQuat=loggerSensors.mocapOrientationQuat,
                 )
