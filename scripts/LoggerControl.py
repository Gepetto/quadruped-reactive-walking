'''This class will log 1d array in Nd matrix from device and qualisys object'''
import pickle
import numpy as np
from datetime import datetime as datetime
from time import time
import pinocchio as pin

class LoggerControl():
    def __init__(self, dt, N0_gait, type_MPC=None, joystick=None, estimator=None, loop=None, gait=None, statePlanner=None,
                 footstepPlanner=None, footTrajectoryGenerator=None, logSize=60e3, ringBuffer=False):
        self.ringBuffer = ringBuffer
        logSize = np.int(logSize)
        self.logSize = logSize
        self.i = 0

        self.dt = dt
        if type_MPC is not None:
            self.type_MPC = int(type_MPC)
        else:
            self.type_MPC = 0

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
        self.esti_filt_lin_vel = np.zeros([logSize, 3])  # estimated velocity of the base before low pass filter

        self.esti_HP_x = np.zeros([logSize, 3])  # x input of the velocity complementary filter
        self.esti_HP_dx = np.zeros([logSize, 3])  # dx input of the velocity complementary filter
        self.esti_HP_alpha = np.zeros([logSize, 3])  # alpha parameter of the velocity complementary filter
        self.esti_HP_filt_x = np.zeros([logSize, 3])  # filtered output of the velocity complementary filter

        self.esti_LP_x = np.zeros([logSize, 3])  # x input of the position complementary filter
        self.esti_LP_dx = np.zeros([logSize, 3])  # dx input of the position complementary filter
        self.esti_LP_alpha = np.zeros([logSize, 3])  # alpha parameter of the position complementary filter
        self.esti_LP_filt_x = np.zeros([logSize, 3])  # filtered output of the position complementary filter

        # Loop
        self.loop_o_q = np.zeros([logSize, 18])  # position in ideal world frame (esti_q_filt + dt * loop_o_v)
        self.loop_o_v = np.zeros([logSize, 18])  # estimated velocity in world frame
        self.loop_h_v = np.zeros([logSize, 18])  # estimated velocity in horizontal frame
        self.loop_h_v_windowed = np.zeros([logSize, 6])  # estimated velocity in horizontal frame
        self.loop_t_filter = np.zeros([logSize])  # time taken by the estimator
        self.loop_t_planner = np.zeros([logSize])  # time taken by the planning
        self.loop_t_mpc = np.zeros([logSize])  # time taken by the mcp
        self.loop_t_wbc = np.zeros([logSize])  # time taken by the whole body control
        self.loop_t_loop = np.zeros([logSize])  # time taken by the whole loop (without interface)
        self.loop_t_loop_if = np.zeros([logSize])  # time taken by the whole loop (with interface)
        self.loop_q_filt_mpc = np.zeros([logSize, 6])  # position in world frame filtered by 1st order low pass
        self.loop_h_v_filt_mpc = np.zeros([logSize, 6])  # vel in base frame filtered by 1st order low pass
        self.loop_vref_filt_mpc = np.zeros([logSize, 6])  # ref vel in base frame filtered by 1st order low pass

        # Gait
        self.planner_gait = np.zeros([logSize, N0_gait, 4])  # Gait sequence
        self.planner_is_static = np.zeros([logSize])  # if the planner is in static mode or not

        # State planner
        if statePlanner is not None:
            self.planner_xref = np.zeros([logSize, 12, 1+statePlanner.getNSteps()])  # Reference trajectory

        # Footstep planner
        if gait is not None:
            self.planner_fsteps = np.zeros([logSize, gait.getCurrentGait().shape[0], 12])  # Reference footsteps position
        self.planner_h_ref = np.zeros([logSize])  # reference height of the planner

        # Foot Trajectory Generator
        self.planner_goals = np.zeros([logSize, 3, 4])   # 3D target feet positions
        self.planner_vgoals = np.zeros([logSize, 3, 4])  # 3D target feet velocities
        self.planner_agoals = np.zeros([logSize, 3, 4])  # 3D target feet accelerations
        self.planner_jgoals = np.zeros([logSize, 3, 4])  # 3D target feet accelerations

        # Model Predictive Control
        # output vector of the MPC (next state + reference contact force)
        if statePlanner is not None:
            if loop.type_MPC == 3:
                self.mpc_x_f = np.zeros([logSize, 32, statePlanner.getNSteps()])
            else:
                self.mpc_x_f = np.zeros([logSize, 24, statePlanner.getNSteps()])
        self.mpc_solving_duration = np.zeros([logSize])

        # Whole body control
        self.wbc_P = np.zeros([logSize, 12])  # proportionnal gains of the PD+
        self.wbc_D = np.zeros([logSize, 12])  # derivative gains of the PD+
        self.wbc_q_des = np.zeros([logSize, 12])  # desired position of actuators
        self.wbc_v_des = np.zeros([logSize, 12])  # desired velocity of actuators
        self.wbc_FF = np.zeros([logSize, 12])  # gains for the feedforward torques
        self.wbc_tau_ff = np.zeros([logSize, 12])  # feedforward torques computed by the WBC
        self.wbc_ddq_IK = np.zeros([logSize, 18])  # joint accelerations computed by the IK
        self.wbc_f_ctc = np.zeros([logSize, 12])  # contact forces computed by the WBC
        self.wbc_ddq_QP = np.zeros([logSize, 18])  # joint accelerations computed by the QP
        self.wbc_feet_pos = np.zeros([logSize, 3, 4])  # current feet positions according to WBC
        self.wbc_feet_pos_target = np.zeros([logSize, 3, 4])  # current feet positions targets for WBC
        self.wbc_feet_err = np.zeros([logSize, 3, 4])  # error between feet positions and their reference
        self.wbc_feet_vel = np.zeros([logSize, 3, 4])  # current feet velocities according to WBC
        self.wbc_feet_vel_target = np.zeros([logSize, 3, 4])  # current feet velocities targets for WBC
        self.wbc_feet_acc_target = np.zeros([logSize, 3, 4])  # current feet accelerations targets for WBC

        # Timestamps
        self.tstamps = np.zeros(logSize)

    def sample(self, joystick, estimator, loop, gait, statePlanner, footstepPlanner, footTrajectoryGenerator, wbc, dT_whole):
        if (self.i >= self.logSize):
            if self.ringBuffer:
                self.i = 0
            else:
                return

        # Logging from joystick
        self.joy_v_ref[self.i] = joystick.v_ref[:, 0]

        # Logging from estimator
        self.esti_feet_status[self.i] = estimator.getFeetStatus()
        self.esti_feet_goals[self.i] = estimator.getFeetGoals()
        self.esti_q_filt[self.i] = estimator.getQFilt()
        self.esti_v_filt[self.i] = estimator.getVFilt()
        self.esti_v_secu[self.i] = estimator.getVSecu()

        self.esti_FK_lin_vel[self.i] = estimator.getFKLinVel()
        self.esti_FK_xyz[self.i] = estimator.getFKXYZ()
        self.esti_xyz_mean_feet[self.i] = estimator.getXYZMeanFeet()
        self.esti_filt_lin_vel[self.i] = estimator.getFiltLinVel()

        self.esti_HP_x[self.i] = estimator.getFilterVelX()
        self.esti_HP_dx[self.i] = estimator.getFilterVelDX()
        self.esti_HP_alpha[self.i] = estimator.getFilterVelAlpha()
        self.esti_HP_filt_x[self.i] = estimator.getFilterVelFiltX()

        self.esti_LP_x[self.i] = estimator.getFilterPosX()
        self.esti_LP_dx[self.i] = estimator.getFilterPosDX()
        self.esti_LP_alpha[self.i] = estimator.getFilterPosAlpha()
        self.esti_LP_filt_x[self.i] = estimator.getFilterPosFiltX()

        # Logging from the main loop
        self.loop_o_q[self.i] = loop.q[:, 0]
        self.loop_o_v[self.i] = loop.v[:, 0]
        self.loop_h_v[self.i] = loop.h_v[:, 0]
        self.loop_h_v_windowed[self.i] = loop.h_v_windowed[:, 0]
        self.loop_t_filter[self.i] = loop.t_filter
        self.loop_t_planner[self.i] = loop.t_planner
        self.loop_t_mpc[self.i] = loop.t_mpc
        self.loop_t_wbc[self.i] = loop.t_wbc
        self.loop_t_loop[self.i] = loop.t_loop
        self.loop_t_loop_if[self.i] = dT_whole
        self.loop_q_filt_mpc[self.i] = loop.q_filt_mpc[:6, 0]
        self.loop_h_v_filt_mpc[self.i] = loop.h_v_filt_mpc[:, 0]
        self.loop_vref_filt_mpc[self.i] = loop.vref_filt_mpc[:, 0]

        # Logging from the planner
        self.planner_xref[self.i] = statePlanner.getReferenceStates()
        self.planner_fsteps[self.i] = footstepPlanner.getFootsteps()
        self.planner_gait[self.i] = gait.getCurrentGait()
        self.planner_goals[self.i] = footTrajectoryGenerator.getFootPosition()
        self.planner_vgoals[self.i] = footTrajectoryGenerator.getFootVelocity()
        self.planner_agoals[self.i] = footTrajectoryGenerator.getFootAcceleration()
        self.planner_jgoals[self.i] = footTrajectoryGenerator.getFootJerk()
        self.planner_is_static[self.i] = gait.getIsStatic()
        self.planner_h_ref[self.i] = loop.h_ref

        # Logging from model predictive control
        self.mpc_x_f[self.i] = loop.x_f_mpc
        self.mpc_solving_duration[self.i] = loop.mpc_wrapper.t_mpc_solving_duration

        # Logging from whole body control
        self.wbc_P[self.i] = loop.result.P
        self.wbc_D[self.i] = loop.result.D
        self.wbc_q_des[self.i] = loop.result.q_des
        self.wbc_v_des[self.i] = loop.result.v_des
        self.wbc_FF[self.i] = loop.result.FF
        self.wbc_tau_ff[self.i] = loop.result.tau_ff
        self.wbc_ddq_IK[self.i] = wbc.ddq_cmd
        self.wbc_f_ctc[self.i] = wbc.f_with_delta
        self.wbc_ddq_QP[self.i] = wbc.ddq_with_delta
        self.wbc_feet_pos[self.i] = wbc.feet_pos
        self.wbc_feet_pos_target[self.i] = wbc.feet_pos_target
        self.wbc_feet_err[self.i] = wbc.feet_err
        self.wbc_feet_vel[self.i] = wbc.feet_vel
        self.wbc_feet_vel_target[self.i] = wbc.feet_vel_target
        self.wbc_feet_acc_target[self.i] = wbc.feet_acc_target

        # Logging timestamp
        self.tstamps[self.i] = time()

        self.i += 1

    def processMocap(self, N, loggerSensors):

        self.mocap_pos = np.zeros([N, 3])
        self.mocap_h_v = np.zeros([N, 3])
        self.mocap_b_w = np.zeros([N, 3])
        self.mocap_RPY = np.zeros([N, 3])
   
        for i in range(N):
            self.mocap_RPY[i] = pin.rpy.matrixToRpy(pin.Quaternion(loggerSensors.mocapOrientationQuat[i]).toRotationMatrix())

        # Robot world to Mocap initial translationa and rotation
        mTo = np.array([loggerSensors.mocapPosition[0, 0], loggerSensors.mocapPosition[0, 1], 0.02])  
        mRo = pin.rpy.rpyToMatrix(0.0, 0.0, self.mocap_RPY[0, 2])

        for i in range(N):
            oRb = loggerSensors.mocapOrientationMat9[i]

            oRh = pin.rpy.rpyToMatrix(0.0, 0.0, self.mocap_RPY[i, 2] - self.mocap_RPY[0, 2])

            self.mocap_h_v[i] = (oRh.transpose() @ mRo.transpose() @ loggerSensors.mocapVelocity[i].reshape((3, 1))).ravel()
            self.mocap_b_w[i] = (oRb.transpose() @ loggerSensors.mocapAngularVelocity[i].reshape((3, 1))).ravel()
            self.mocap_pos[i] = (mRo.transpose() @ (loggerSensors.mocapPosition[i, :] - mTo).reshape((3, 1))).ravel()

    def plotAll(self, loggerSensors):

        from matplotlib import pyplot as plt

        N = self.tstamps.shape[0]
        t_range = np.array([k*self.dt for k in range(N)])

        self.processMocap(N, loggerSensors)

        index6 = [1, 3, 5, 2, 4, 6]
        index12 = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]

        # Reconstruct pos and vel of feet in base frame to compare them with the
        # ones desired by the foot trajectory generator and whole-body control
        from example_robot_data.robots_loader import Solo12Loader
        Solo12Loader.free_flyer = True
        solo12 = Solo12Loader().robot
        FL_FOOT_ID = solo12.model.getFrameId('FL_FOOT')
        FR_FOOT_ID = solo12.model.getFrameId('FR_FOOT')
        HL_FOOT_ID = solo12.model.getFrameId('HL_FOOT')
        HR_FOOT_ID = solo12.model.getFrameId('HR_FOOT')
        foot_ids = np.array([FL_FOOT_ID, FR_FOOT_ID, HL_FOOT_ID, HR_FOOT_ID])
        q = np.zeros((19, 1))
        dq = np.zeros((18, 1))
        pin.computeAllTerms(solo12.model, solo12.data, q, np.zeros((18, 1)))
        feet_pos = np.zeros([self.esti_q_filt.shape[0], 3, 4])
        feet_vel = np.zeros([self.esti_q_filt.shape[0], 3, 4])
        for i in range(self.esti_q_filt.shape[0]):
            q[:3, 0] = self.loop_q_filt_mpc[i, :3]
            q[3:7, 0] = pin.Quaternion(pin.rpy.rpyToMatrix(self.loop_q_filt_mpc[i, 3:6])).coeffs() 
            q[7:, 0] = self.loop_o_q[i, 6:]
            dq[6:, 0] = self.loop_o_v[i, 6:]
            pin.forwardKinematics(solo12.model, solo12.data, q, dq)
            pin.updateFramePlacements(solo12.model, solo12.data)
            for j, idx in enumerate(foot_ids):
                feet_pos[i, :, j] = solo12.data.oMf[int(idx)].translation
                feet_vel[i, :, j] = pin.getFrameVelocity(solo12.model, solo12.data, int(idx), pin.LOCAL_WORLD_ALIGNED).linear

        ####
        # Measured & Reference feet positions (base frame)
        ####
        lgd_X = ["FL", "FR", "HL", "HR"]
        lgd_Y = ["Pos X", "Pos Y", "Pos Z"]
        plt.figure()
        for i in range(12):
            if i == 0:
                ax0 = plt.subplot(3, 4, index12[i])
            else:
                plt.subplot(3, 4, index12[i], sharex=ax0)

            plt.plot(t_range, self.wbc_feet_pos[:, i % 3, np.int(i/3)], color='b', linewidth=3, marker='')
            plt.plot(t_range, self.wbc_feet_pos_target[:, i % 3, np.int(i/3)], color='r', linewidth=3, marker='')
            plt.plot(t_range, feet_pos[:, i % 3, np.int(i/3)], color='rebeccapurple', linewidth=3, marker='')
            if (i % 3) == 2:
                mini = np.min(self.wbc_feet_pos[:, i % 3, np.int(i/3)])
                maxi = np.max(self.wbc_feet_pos[:, i % 3, np.int(i/3)])
                plt.plot(t_range, self.planner_gait[:, 0, np.int(
                    i/3)] * (maxi - mini) + mini, color='k', linewidth=3, marker='')
            plt.legend([lgd_Y[i % 3] + " " + lgd_X[np.int(i/3)]+" WBC",
                        lgd_Y[i % 3] + " " + lgd_X[np.int(i/3)]+" Ref",
                        lgd_Y[i % 3] + " " + lgd_X[np.int(i/3)]+" Robot", "Contact state"], prop={'size': 8})
        self.custom_suptitle("Feet positions (base frame)")

        ####
        # Measured & Reference feet velocities (base frame)
        ####
        lgd_X = ["FL", "FR", "HL", "HR"]
        lgd_Y = ["Vel X", "Vel Y", "Vel Z"]
        plt.figure()
        for i in range(12):
            if i == 0:
                ax0 = plt.subplot(3, 4, index12[i])
            else:
                plt.subplot(3, 4, index12[i], sharex=ax0)
            plt.plot(t_range, self.wbc_feet_vel[:, i % 3, np.int(i/3)], color='b', linewidth=3, marker='')
            plt.plot(t_range, self.wbc_feet_vel_target[:, i % 3, np.int(i/3)], color='r', linewidth=3, marker='')
            plt.plot(t_range, feet_vel[:, i % 3, np.int(i/3)], color='rebeccapurple', linewidth=3, marker='')
            plt.legend([lgd_Y[i % 3] + " WBC" + lgd_X[np.int(i/3)],
                        lgd_Y[i % 3] + " " + lgd_X[np.int(i/3)] + " Ref",
                        lgd_Y[i % 3] + " " + lgd_X[np.int(i/3)] + " Robot"], prop={'size': 8})
        self.custom_suptitle("Feet velocities (base frame)")

        ####
        # Reference feet accelerations (base frame)
        ####
        lgd_X = ["FL", "FR", "HL", "HR"]
        lgd_Y = ["Acc X", "Acc Y", "Acc Z"]
        plt.figure()
        for i in range(12):
            if i == 0:
                ax0 = plt.subplot(3, 4, index12[i])
            else:
                plt.subplot(3, 4, index12[i], sharex=ax0)
            plt.plot(t_range, self.wbc_feet_acc_target[:, i % 3, np.int(i/3)], color='r', linewidth=3, marker='')
            plt.legend([lgd_Y[i % 3] + " " + lgd_X[np.int(i/3)]+" Ref"], prop={'size': 8})
        self.custom_suptitle("Feet accelerations (base frame)")

        ####
        # Measured & Reference position and orientation (ideal world frame)
        ####
        lgd = ["Pos X", "Pos Y", "Pos Z", "Roll", "Pitch", "Yaw"]
        plt.figure()
        for i in range(6):
            if i == 0:
                ax0 = plt.subplot(3, 2, index6[i])
            else:
                plt.subplot(3, 2, index6[i], sharex=ax0)

            if i in [0, 1, 5]:
                plt.plot(t_range, self.loop_o_q[:, i], "b", linewidth=3)
            else:
                plt.plot(t_range, self.planner_xref[:, i, 0], "b", linewidth=2)
            if i < 3:
                plt.plot(t_range, self.mocap_pos[:, i], "k", linewidth=3)
            else:
                plt.plot(t_range, self.mocap_RPY[:, i-3], "k", linewidth=3)
            if i in [0, 1, 5]:
                plt.plot(t_range, self.loop_o_q[:, i], "r", linewidth=3)
            else:
                plt.plot(t_range, self.planner_xref[:, i, 1], "r", linewidth=3)
            plt.legend(["Robot state", "Ground truth", "Robot reference state"], prop={'size': 8})
            plt.ylabel(lgd[i])
        self.custom_suptitle("Position and orientation")

        ####
        # Measured & Reference linear and angular velocities (horizontal frame)
        ####
        lgd = ["Linear vel X", "Linear vel Y", "Linear vel Z",
               "Angular vel Roll", "Angular vel Pitch", "Angular vel Yaw"]
        plt.figure()
        for i in range(6):
            if i == 0:
                ax0 = plt.subplot(3, 2, index6[i])
            else:
                plt.subplot(3, 2, index6[i], sharex=ax0)

            plt.plot(t_range, self.loop_h_v[:, i], "b", linewidth=2)
            if i < 3:
                plt.plot(t_range, self.mocap_h_v[:, i], "k", linewidth=3)
                plt.plot(t_range, self.loop_h_v_filt_mpc[:, i], linewidth=3, color="forestgreen")
                plt.plot(t_range, self.loop_h_v_windowed[:, i], linewidth=3, color="rebeccapurple")
            else:
                plt.plot(t_range, self.mocap_b_w[:, i-3], "k", linewidth=3)
            plt.plot(t_range, self.joy_v_ref[:, i], "r", linewidth=3)
            if i < 3:
                plt.legend(["State", "Ground truth",
                        "State (LP 15Hz)", "State (windowed)", "Ref state"], prop={'size': 8})
            else:
                plt.legend(["State", "Ground truth", "Ref state"], prop={'size': 8})
            plt.ylabel(lgd[i])
        self.custom_suptitle("Linear and angular velocities")

        print("RMSE: ", np.sqrt(((self.joy_v_ref[:-1000, 0] - self.mocap_h_v[:-1000, 0])**2).mean()))

        # Analysis of the footstep locations (current and future) with a slider to move along time
        # self.slider_predicted_footholds()

        ####
        # Analysis of the footholds locations during the whole experiment
        ####
        """f_c = ["r", "b", "forestgreen", "rebeccapurple"]
        quat = np.zeros((4, 1))
        steps = np.zeros((12, 1))
        o_step = np.zeros((3, 1))
        plt.figure()
        plt.plot(self.loop_o_q[:, 0], self.loop_o_q[:, 1], linewidth=2, color="k")
        for i in range(self.planner_fsteps.shape[0]):
            fsteps = self.planner_fsteps[i]
            RPY = pin.rpy.matrixToRpy(pin.Quaternion(self.loop_o_q[0, 3:7]).toRotationMatrix())
            quat[:, 0] = pin.Quaternion(pin.rpy.rpyToMatrix(np.array([0.0, 0.0, RPY[2]]))).coeffs()
            oRh = pin.Quaternion(quat).toRotationMatrix()
            for j in range(4):
                #if np.any(fsteps[k, (j*3):((j+1)*3)]) and not np.array_equal(steps[(j*3):((j+1)*3), 0],
                #                                                                fsteps[k, (j*3):((j+1)*3)]):
                # steps[(j*3):((j+1)*3), 0] = fsteps[k, (j*3):((j+1)*3)]
                # o_step[:, 0:1] = oRh @ steps[(j*3):((j+1)*3), 0:1] + self.loop_o_q[i:(i+1), 0:3].transpose()
                o_step[:, 0:1] = oRh @ fsteps[0:1, (j*3):((j+1)*3)].transpose() + self.loop_o_q[i:(i+1), 0:3].transpose()
                plt.plot(o_step[0, 0], o_step[1, 0], linestyle=None, linewidth=1, marker="o", color=f_c[j])"""
        
        ####
        # FF torques & FB torques & Sent torques & Meas torques
        ####
        lgd1 = ["HAA", "HFE", "Knee"]
        lgd2 = ["FL", "FR", "HL", "HR"]
        plt.figure()
        for i in range(12):
            if i == 0:
                ax0 = plt.subplot(3, 4, index12[i])
            else:
                plt.subplot(3, 4, index12[i], sharex=ax0)
            tau_fb = self.wbc_P[:, i] * (self.wbc_q_des[:, i] - self.loop_o_q[:, 6+i]) + \
                self.wbc_D[:, i] * (self.wbc_v_des[:, i] - self.loop_o_v[:, 6+i])
            h1, = plt.plot(t_range, self.wbc_FF[:, i] * self.wbc_tau_ff[:, i], "r", linewidth=3)
            h2, = plt.plot(t_range, tau_fb, "b", linewidth=3)
            h3, = plt.plot(t_range, self.wbc_FF[:, i] * self.wbc_tau_ff[:, i] + tau_fb, "g", linewidth=3)
            h4, = plt.plot(t_range[:-1], loggerSensors.torquesFromCurrentMeasurment[1:, i],
                           "violet", linewidth=3, linestyle="--")
            plt.xlabel("Time [s]")
            plt.ylabel(lgd1[i % 3]+" "+lgd2[int(i/3)]+" [Nm]")
            tmp = lgd1[i % 3]+" "+lgd2[int(i/3)]
            plt.legend([h1, h2, h3, h4], ["FF "+tmp, "FB "+tmp, "PD+ "+tmp, "Meas "+tmp], prop={'size': 8})
            plt.ylim([-8.0, 8.0])
        self.custom_suptitle("Torques")

        ####
        # Contact forces (MPC command) & WBC QP output
        ####
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
        self.custom_suptitle("Contact forces (MPC command) & WBC QP output")

        ####
        # Desired & Measured actuator positions
        ####
        lgd1 = ["HAA", "HFE", "Knee"]
        lgd2 = ["FL", "FR", "HL", "HR"]
        plt.figure()
        for i in range(12):
            if i == 0:
                ax0 = plt.subplot(3, 4, index12[i])
            else:
                plt.subplot(3, 4, index12[i], sharex=ax0)
            h1, = plt.plot(t_range, self.wbc_q_des[:, i], color='r', linewidth=3)
            h2, = plt.plot(t_range, self.loop_o_q[:, 6+i], color='b', linewidth=3)
            plt.xlabel("Time [s]")
            plt.ylabel(lgd1[i % 3]+" "+lgd2[int(i/3)]+" [rad]")
            plt.legend([h1, h2], ["Ref "+lgd1[i % 3]+" "+lgd2[int(i/3)],
                                  lgd1[i % 3]+" "+lgd2[int(i/3)]], prop={'size': 8})
        self.custom_suptitle("Actuator positions")

        ####
        # Desired & Measured actuator velocity
        ####
        lgd1 = ["HAA", "HFE", "Knee"]
        lgd2 = ["FL", "FR", "HL", "HR"]
        plt.figure()
        for i in range(12):
            if i == 0:
                ax0 = plt.subplot(3, 4, index12[i])
            else:
                plt.subplot(3, 4, index12[i], sharex=ax0)
            h1, = plt.plot(t_range, self.wbc_v_des[:, i], color='r', linewidth=3)
            h2, = plt.plot(t_range, self.loop_o_v[:, 6+i], color='b', linewidth=3)
            plt.xlabel("Time [s]")
            plt.ylabel(lgd1[i % 3]+" "+lgd2[int(i/3)]+" [rad]")
            plt.legend([h1, h2], ["Ref "+lgd1[i % 3]+" "+lgd2[int(i/3)],
                                  lgd1[i % 3]+" "+lgd2[int(i/3)]], prop={'size': 8})
        self.custom_suptitle("Actuator velocities")

        ####
        # Evolution of trajectories in position and orientation computed by the MPC
        ####
        """
        log_t_pred = np.array([k*self.dt*10 for k in range(self.mpc_x_f.shape[2])])
        log_t_ref = np.array([k*self.dt*10 for k in range(self.planner_xref.shape[2])])

        titles = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
        step = 1000
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
            #h3, = plt.plot(np.array([k*self.dt for k in range(self.mpc_x_f.shape[0])]),
            #               self.planner_xref[:, j, 0], linestyle=None, marker='x', color="r", linewidth=1)
            plt.xlabel("Time [s]")
            plt.legend([h1, h2, h3], ["Output trajectory of MPC",
                                      "Input trajectory of planner"]) #, "Actual robot trajectory"])
            plt.title("Predicted trajectory for " + titles[j])
        self.custom_suptitle("Analysis of trajectories in position and orientation computed by the MPC")
        """

        ####
        # Evolution of trajectories of linear and angular velocities computed by the MPC
        ####
        """
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
        self.custom_suptitle("Analysis of trajectories of linear and angular velocities computed by the MPC")
        """

        ####
        # Evolution of contact force trajectories
        ####
        """
        step = 1000
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
        self.custom_suptitle("Contact forces trajectories & Actual forces trajectories")
        """

        ####
        # Analysis of the complementary filter behaviour
        ####
        """
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
        self.custom_suptitle("Evolution of the quantities of the velocity complementary filter")

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
        self.custom_suptitle("Evolution of the quantities of the position complementary filter")
        """

        ####
        # Power supply profile
        ####
        
        plt.figure()
        for i in range(3):
            if i == 0:
                ax0 = plt.subplot(3, 1, i+1)
            else:
                plt.subplot(3, 1, i+1, sharex=ax0)

            if i == 0:
                plt.plot(t_range, loggerSensors.current[:], linewidth=2)
                plt.ylabel("Bus current [A]")
            elif i == 1:
                plt.plot(t_range, loggerSensors.voltage[:], linewidth=2)
                plt.ylabel("Bus voltage [V]")
            else:
                plt.plot(t_range, loggerSensors.energy[:], linewidth=2)
                plt.ylabel("Bus energy [J]")
                plt.xlabel("Time [s]")
        

        ####
        # Estimated computation time for each step of the control architecture
        ####
        plt.figure()
        plt.plot(t_range, self.loop_t_filter, 'r+')
        plt.plot(t_range, self.loop_t_planner, 'g+')
        plt.plot(t_range, self.loop_t_mpc, 'b+')
        plt.plot(t_range, self.loop_t_wbc, '+', color="violet")
        plt.plot(t_range, self.loop_t_loop, 'k+')
        plt.plot(t_range, self.loop_t_loop_if, '+', color="rebeccapurple")
        plt.legend(["Estimator", "Planner", "MPC", "WBC", "Control loop", "Whole loop"])
        plt.xlabel("Time [s]")
        plt.ylabel("Time [s]")
        self.custom_suptitle("Computation time of each block")

        # Plot estimated solving time of the model prediction control
        fig = plt.figure()
        plt.plot(t_range[35:], self.mpc_solving_duration[35:], 'k+')
        plt.legend(["Solving duration"])
        plt.xlabel("Time [s]")
        plt.ylabel("Time [s]")
        self.custom_suptitle("MPC solving time")

        ####
        # Step in system time at each loop
        ####
        plt.figure()
        plt.plot(np.diff(self.tstamps))
        plt.legend(["System time step"])
        plt.xlabel("Loop []")
        plt.ylabel("Time [s]")
        self.custom_suptitle("System time step between 2 sucessive loops")

        ####
        # Comparison of raw and filtered quantities (15 Hz and windowed)
        ####
        """
        lgd = ["Position X", "Position Y", "Position Z", "Position Roll", "Position Pitch", "Position Yaw"]
        plt.figure()
        for i in range(6):
            if i == 0:
                ax0 = plt.subplot(3, 2, index6[i])
            else:
                plt.subplot(3, 2, index6[i], sharex=ax0)

            plt.plot(t_range, self.loop_o_q[:, i], linewidth=3)
            plt.plot(t_range, self.loop_q_filt_mpc[:, i], linewidth=3)

            plt.legend(["Estimated", "Estimated 15Hz filt"], prop={'size': 8})
            plt.ylabel(lgd[i])
        self.custom_suptitle("Comparison between position quantities before and after 15Hz low-pass filtering")

        lgd = ["Linear vel X", "Linear vel Y", "Linear vel Z",
               "Angular vel Roll", "Angular vel Pitch", "Angular vel Yaw"]
        plt.figure()
        for i in range(6):
            if i == 0:
                ax0 = plt.subplot(3, 2, index6[i])
            else:
                plt.subplot(3, 2, index6[i], sharex=ax0)

            plt.plot(t_range, self.loop_h_v[:, i], linewidth=3)
            plt.plot(t_range, self.loop_h_v_windowed[:, i], linewidth=3)
            plt.plot(t_range, self.loop_h_v_filt_mpc[:, i], linewidth=3)
            plt.plot(t_range, self.loop_vref_filt_mpc[:, i], linewidth=3)

            plt.legend(["Estimated", "Estimated 3Hz windowed", "Estimated 15Hz filt",
                        "Estimated 15Hz filt 3Hz windowed", "Reference 15Hz filt"], prop={'size': 8})
            plt.ylabel(lgd[i])
        self.custom_suptitle("Comparison between velocity quantities before and after 15Hz low-pass filtering")
        """

        ###############################
        # Display all graphs and wait #
        ###############################
        plt.show(block=True)

    def custom_suptitle(self, name):
        from matplotlib import pyplot as plt

        fig = plt.gcf()
        fig.suptitle(name)
        fig.canvas.manager.set_window_title(name)


    def saveAll(self, loggerSensors, fileName="data"):
        date_str = datetime.now().strftime('_%Y_%m_%d_%H_%M')

        np.savez_compressed(fileName + date_str + "_" + str(self.type_MPC) + ".npz",

                            joy_v_ref=self.joy_v_ref,

                            esti_feet_status=self.esti_feet_status,
                            esti_feet_goals=self.esti_feet_goals,
                            esti_q_filt=self.esti_q_filt,
                            esti_v_filt=self.esti_v_filt,
                            esti_v_secu=self.esti_v_secu,

                            esti_FK_lin_vel=self.esti_FK_lin_vel,
                            esti_FK_xyz=self.esti_FK_xyz,
                            esti_xyz_mean_feet=self.esti_xyz_mean_feet,
                            esti_filt_lin_vel=self.esti_filt_lin_vel,

                            esti_HP_x=self.esti_HP_x,
                            esti_HP_dx=self.esti_HP_dx,
                            esti_HP_alpha=self.esti_HP_alpha,
                            esti_HP_filt_x=self.esti_HP_filt_x,

                            esti_LP_x=self.esti_LP_x,
                            esti_LP_dx=self.esti_LP_dx,
                            esti_LP_alpha=self.esti_LP_alpha,
                            esti_LP_filt_x=self.esti_LP_filt_x,

                            loop_o_q=self.loop_o_q,
                            loop_o_v=self.loop_o_v,
                            loop_h_v=self.loop_h_v,
                            loop_h_v_windowed=self.loop_h_v_windowed,
                            loop_t_filter=self.loop_t_filter,
                            loop_t_planner=self.loop_t_planner,
                            loop_t_mpc=self.loop_t_mpc,
                            loop_t_wbc=self.loop_t_wbc,
                            loop_t_loop=self.loop_t_loop,
                            loop_t_loop_if=self.loop_t_loop_if,
                            loop_q_filt_mpc=self.loop_q_filt_mpc,
                            loop_h_v_filt_mpc=self.loop_h_v_filt_mpc,
                            loop_vref_filt_mpc=self.loop_vref_filt_mpc,

                            planner_xref=self.planner_xref,
                            planner_fsteps=self.planner_fsteps,
                            planner_gait=self.planner_gait,
                            planner_goals=self.planner_goals,
                            planner_vgoals=self.planner_vgoals,
                            planner_agoals=self.planner_agoals,
                            planner_jgoals=self.planner_jgoals,
                            planner_is_static=self.planner_is_static,
                            planner_h_ref=self.planner_h_ref,

                            mpc_x_f=self.mpc_x_f,
                            mpc_solving_duration=self.mpc_solving_duration,

                            wbc_P=self.wbc_P,
                            wbc_D=self.wbc_D,
                            wbc_q_des=self.wbc_q_des,
                            wbc_v_des=self.wbc_v_des,
                            wbc_FF=self.wbc_FF,
                            wbc_tau_ff=self.wbc_tau_ff,
                            wbc_ddq_IK=self.wbc_ddq_IK,
                            wbc_f_ctc=self.wbc_f_ctc,
                            wbc_ddq_QP=self.wbc_ddq_QP,
                            wbc_feet_pos=self.wbc_feet_pos,
                            wbc_feet_pos_target=self.wbc_feet_pos_target,
                            wbc_feet_err=self.wbc_feet_err,
                            wbc_feet_vel=self.wbc_feet_vel,
                            wbc_feet_vel_target=self.wbc_feet_vel_target,
                            wbc_feet_acc_target=self.wbc_feet_acc_target,

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
                            current=loggerSensors.current,
                            voltage=loggerSensors.voltage,
                            energy=loggerSensors.energy,
                            )

    def loadAll(self, loggerSensors, fileName=None):

        if fileName is None:
            import glob
            fileName = np.sort(glob.glob('data_2021_*.npz'))[-1]  # Most recent file

        data = np.load(fileName, allow_pickle = True)

        # Load LoggerControl arrays
        self.joy_v_ref = data["joy_v_ref"]

        self.logSize = self.joy_v_ref.shape[0]

        self.esti_feet_status = data["esti_feet_status"]
        self.esti_feet_goals = data["esti_feet_goals"]
        self.esti_q_filt = data["esti_q_filt"]
        self.esti_v_filt = data["esti_v_filt"]
        self.esti_v_secu = data["esti_v_secu"]

        self.esti_FK_lin_vel = data["esti_FK_lin_vel"]
        self.esti_FK_xyz = data["esti_FK_xyz"]
        self.esti_xyz_mean_feet = data["esti_xyz_mean_feet"]
        self.esti_filt_lin_vel = data["esti_filt_lin_vel"]

        self.esti_HP_x = data["esti_HP_x"]
        self.esti_HP_dx = data["esti_HP_dx"]
        self.esti_HP_alpha = data["esti_HP_alpha"]
        self.esti_HP_filt_x = data["esti_HP_filt_x"]

        self.esti_LP_x = data["esti_LP_x"]
        self.esti_LP_dx = data["esti_LP_dx"]
        self.esti_LP_alpha = data["esti_LP_alpha"]
        self.esti_LP_filt_x = data["esti_LP_filt_x"]

        self.loop_o_q = data["loop_o_q"]
        self.loop_o_v = data["loop_o_v"]
        self.loop_h_v = data["loop_h_v"]
        self.loop_h_v_windowed = data["loop_h_v_windowed"]
        self.loop_t_filter = data["loop_t_filter"]
        self.loop_t_planner = data["loop_t_planner"]
        self.loop_t_mpc = data["loop_t_mpc"]
        self.loop_t_wbc = data["loop_t_wbc"]
        self.loop_t_loop = data["loop_t_loop"]
        self.loop_t_loop_if = data["loop_t_loop_if"]
        self.loop_q_filt_mpc = data["loop_q_filt_mpc"]
        self.loop_h_v_filt_mpc = data["loop_h_v_filt_mpc"]
        self.loop_vref_filt_mpc = data["loop_vref_filt_mpc"]

        self.planner_xref = data["planner_xref"]
        self.planner_fsteps = data["planner_fsteps"]
        self.planner_gait = data["planner_gait"]
        self.planner_goals = data["planner_goals"]
        self.planner_vgoals = data["planner_vgoals"]
        self.planner_agoals = data["planner_agoals"]
        self.planner_jgoals = data["planner_jgoals"]
        self.planner_is_static = data["planner_is_static"]
        self.planner_h_ref = data["planner_h_ref"]

        self.mpc_x_f = data["mpc_x_f"]
        self.mpc_solving_duration = data["mpc_solving_duration"]

        self.wbc_P = data["wbc_P"]
        self.wbc_D = data["wbc_D"]
        self.wbc_q_des = data["wbc_q_des"]
        self.wbc_v_des = data["wbc_v_des"]
        self.wbc_FF = data["wbc_FF"]
        self.wbc_tau_ff = data["wbc_tau_ff"]
        self.wbc_ddq_IK = data["wbc_ddq_IK"]
        self.wbc_f_ctc = data["wbc_f_ctc"]
        self.wbc_ddq_QP = data["wbc_ddq_QP"]
        self.wbc_feet_pos = data["wbc_feet_pos"]
        self.wbc_feet_pos_target = data["wbc_feet_pos_target"]
        self.wbc_feet_err = data["wbc_feet_err"]
        self.wbc_feet_vel = data["wbc_feet_vel"]
        self.wbc_feet_vel_target = data["wbc_feet_vel_target"]
        self.wbc_feet_acc_target = data["wbc_feet_acc_target"]

        self.tstamps = data["tstamps"]

        # Load LoggerSensors arrays
        loggerSensors.q_mes = data["q_mes"]
        loggerSensors.v_mes = data["v_mes"]
        loggerSensors.baseOrientation = data["baseOrientation"]
        loggerSensors.baseAngularVelocity = data["baseAngularVelocity"]
        loggerSensors.baseLinearAcceleration = data["baseLinearAcceleration"]
        loggerSensors.baseAccelerometer = data["baseAccelerometer"]
        loggerSensors.torquesFromCurrentMeasurment = data["torquesFromCurrentMeasurment"]
        loggerSensors.mocapPosition = data["mocapPosition"]
        loggerSensors.mocapVelocity = data["mocapVelocity"]
        loggerSensors.mocapAngularVelocity = data["mocapAngularVelocity"]
        loggerSensors.mocapOrientationMat9 = data["mocapOrientationMat9"]
        loggerSensors.mocapOrientationQuat = data["mocapOrientationQuat"]
        loggerSensors.logSize = loggerSensors.q_mes.shape[0]
        loggerSensors.current = data["current"]
        loggerSensors.voltage = data["voltage"]
        loggerSensors.energy = data["energy"]
        

    def slider_predicted_trajectory(self):

        from matplotlib import pyplot as plt
        from matplotlib.widgets import Slider, Button

        # The parametrized function to be plotted
        def f(t, time):
            return np.sin(2 * np.pi * t) + time

        index6 = [1, 3, 5, 2, 4, 6]
        log_t_pred = np.array([(k+1)*self.dt*10 for k in range(self.mpc_x_f.shape[2])])
        log_t_ref = np.array([k*self.dt*10 for k in range(self.planner_xref.shape[2])])
        trange = np.max([np.max(log_t_pred), np.max(log_t_ref)])
        h1s = []
        h2s = []
        axs = []
        h1s_vel = []
        h2s_vel = []
        axs_vel = []

        # Define initial parameters
        init_time = 0.0

        # Create the figure and the line that we will manipulate
        fig = plt.figure()
        ax = plt.gca()
        for j in range(6):
            ax = plt.subplot(3, 2, index6[j])
            h1, = plt.plot(log_t_pred, self.mpc_x_f[0, j, :], "b", linewidth=2)
            h2, = plt.plot(log_t_ref, self.planner_xref[0, j, :], linestyle="--", marker='x', color="g", linewidth=2)
            h3, = plt.plot(np.array([k*self.dt for k in range(self.mpc_x_f.shape[0])]),
                           self.planner_xref[:, j, 0], linestyle=None, marker='x', color="r", linewidth=1)
            axs.append(ax)
            h1s.append(h1)
            h2s.append(h2)

        #ax.set_xlabel('Time [s]')
        axcolor = 'lightgoldenrodyellow'
        #ax.margins(x=0)

        # Make a horizontal slider to control the time.
        axtime = plt.axes([0.25, 0.03, 0.65, 0.03], facecolor=axcolor)
        time_slider = Slider(
            ax=axtime,
            label='Time [s]',
            valmin=0.0,
            valmax=self.logSize*self.dt,
            valinit=init_time,
        )

        # Create the figure and the line that we will manipulate (for velocities)
        fig_vel = plt.figure()
        ax = plt.gca()
        for j in range(6):
            ax = plt.subplot(3, 2, index6[j])
            h1, = plt.plot(log_t_pred, self.mpc_x_f[0, j, :], "b", linewidth=2)
            h2, = plt.plot(log_t_ref, self.planner_xref[0, j, :], linestyle="--", marker='x', color="g", linewidth=2)
            h3, = plt.plot(np.array([k*self.dt for k in range(self.mpc_x_f.shape[0])]),
                           self.planner_xref[:, j+6, 0], linestyle=None, marker='x', color="r", linewidth=1)
            axs_vel.append(ax)
            h1s_vel.append(h1)
            h2s_vel.append(h2)

        #axcolor = 'lightgoldenrodyellow'
        #ax.margins(x=0)

        # Make a horizontal slider to control the time.
        axtime_vel = plt.axes([0.25, 0.03, 0.65, 0.03], facecolor=axcolor)
        time_slider_vel = Slider(
            ax=axtime_vel,
            label='Time [s]',
            valmin=0.0,
            valmax=self.logSize*self.dt,
            valinit=init_time,
        )

        # The function to be called anytime a slider's value changes
        def update(val, recursive=False):
            time_slider.val = np.round(val / (self.dt*10), decimals=0) * (self.dt*10)
            rounded = int(np.round(time_slider.val / self.dt, decimals=0))
            for j in range(6):
                h1s[j].set_xdata(log_t_pred + time_slider.val)
                h2s[j].set_xdata(log_t_ref + time_slider.val)
                y1 = self.mpc_x_f[rounded, j, :] - self.planner_xref[rounded, j, 1:]
                y2 = self.planner_xref[rounded, j, :] - self.planner_xref[rounded, j, :]
                h1s[j].set_ydata(y1)
                h2s[j].set_ydata(y2)
                axs[j].set_xlim([time_slider.val - self.dt * 3, time_slider.val+trange+self.dt * 3])
                ymin = np.min([np.min(y1), np.min(y2)])
                ymax = np.max([np.max(y1), np.max(y2)])
                axs[j].set_ylim([ymin - 0.05 * (ymax - ymin), ymax + 0.05 * (ymax - ymin)])
            fig.canvas.draw_idle()
            if not recursive:
                update_vel(time_slider.val, True)

        def update_vel(val, recursive=False):
            time_slider_vel.val = np.round(val / (self.dt*10), decimals=0) * (self.dt*10)
            rounded = int(np.round(time_slider_vel.val / self.dt, decimals=0))
            for j in range(6):
                h1s_vel[j].set_xdata(log_t_pred + time_slider.val)
                h2s_vel[j].set_xdata(log_t_ref + time_slider.val)
                y1 = self.mpc_x_f[rounded, j+6, :]
                y2 = self.planner_xref[rounded, j+6, :]
                h1s_vel[j].set_ydata(y1)
                h2s_vel[j].set_ydata(y2)
                axs_vel[j].set_xlim([time_slider.val - self.dt * 3, time_slider.val+trange+self.dt * 3])
                ymin = np.min([np.min(y1), np.min(y2)])
                ymax = np.max([np.max(y1), np.max(y2)])
                axs_vel[j].set_ylim([ymin - 0.05 * (ymax - ymin), ymax + 0.05 * (ymax - ymin)])
            fig_vel.canvas.draw_idle()
            if not recursive:
                update(time_slider_vel.val, True)

        # register the update function with each slider
        time_slider.on_changed(update)
        time_slider_vel.on_changed(update)

        plt.show()

    def slider_predicted_footholds(self):

        from matplotlib import pyplot as plt
        from matplotlib.widgets import Slider, Button
        import utils_mpc
        import pinocchio as pin

        self.planner_fsteps

        # Define initial parameters
        init_time = 0.0

        # Create the figure and the line that we will manipulate
        fig = plt.figure()
        ax = plt.gca()
        h1s = []

        f_c = ["r", "b", "forestgreen", "rebeccapurple"]
        quat = np.zeros((4, 1))

        fsteps = self.planner_fsteps[0]
        o_step = np.zeros((3*int(fsteps.shape[0]), 1))
        RPY = pin.rpy.matrixToRpy(pin.Quaternion(self.loop_o_q[0, 3:7]).toRotationMatrix())
        quat[:, 0] = pin.Quaternion(pin.rpy.rpyToMatrix(np.array([0.0, 0.0, RPY[2]]))).coeffs()
        oRh = pin.Quaternion(quat).toRotationMatrix()
        for j in range(4):
            o_step[0:3, 0:1] = oRh @ fsteps[0:1, (j*3):((j+1)*3)].transpose() + self.loop_o_q[0:1, 0:3].transpose()
            h1, = plt.plot(o_step[0::3, 0], o_step[1::3, 0], linestyle=None, linewidth=0, marker="o", color=f_c[j])
            h1s.append(h1)

        axcolor = 'lightgoldenrodyellow'

        # Make a horizontal slider to control the time.
        axtime = plt.axes([0.25, 0.03, 0.65, 0.03], facecolor=axcolor)
        time_slider = Slider(
            ax=axtime,
            label='Time [s]',
            valmin=0.0,
            valmax=self.logSize*self.dt,
            valinit=init_time,
        )

        ax.set_xlim([-0.3, 0.5])
        ax.set_ylim([-0.3, 0.5])

        # The function to be called anytime a slider's value changes
        def update(val):
            time_slider.val = np.round(val / (self.dt*10), decimals=0) * (self.dt*10)
            rounded = int(np.round(time_slider.val / self.dt, decimals=0))
            fsteps = self.planner_fsteps[rounded]
            o_step = np.zeros((3*int(fsteps.shape[0]), 1))
            RPY = pin.rpy.matrixToRpy(pin.Quaternion(self.loop_o_q[rounded, 3:7]).toRotationMatrix())
            quat[:, 0] = pin.Quaternion(pin.rpy.rpyToMatrix(np.array([0.0, 0.0, RPY[2]]))).coeffs()
            oRh = pin.Quaternion(quat).toRotationMatrix()
            for j in range(4):
                for k in range(int(fsteps.shape[0])):
                    o_step[(3*k):(3*(k+1)), 0:1] = oRh @ fsteps[(k):(k+1), (j*3):((j+1)*3)].transpose() + self.loop_o_q[rounded:(rounded+1), 0:3].transpose()
                h1s[j].set_xdata(o_step[0::3, 0].copy())
                h1s[j].set_ydata(o_step[1::3, 0].copy())
            fig.canvas.draw_idle()

        # register the update function with each slider
        time_slider.on_changed(update)

        plt.show()


if __name__ == "__main__":

    import LoggerSensors
    import sys
    import os
    from sys import argv
    sys.path.insert(0, os.getcwd()) # adds current directory to python path

    # Data file name to load
    file_name = "/home/thomas_cbrs/Desktop/edin/quadruped-reactive-walking/scripts/crocoddyl_eval/logs/explore_weight_acc/data_2021_09_16_15_10_3.npz"

    # Create loggers
    loggerSensors = LoggerSensors.LoggerSensors(logSize=20000-3)
    logger = LoggerControl(0.001, 30, logSize=20000-3)

    # Load data from .npz file
    logger.loadAll(loggerSensors, fileName= file_name)

    # Call all ploting functions
    logger.plotAll(loggerSensors)

    # logger.slider_predicted_trajectory()
