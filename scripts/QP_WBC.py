
from utils_mpc import quaternionToRPY
import numpy as np
import scipy as scipy
import pinocchio as pin
from example_robot_data import load
import osqp as osqp
from solo12InvKin import Solo12InvKin
from time import clock, time
import libquadruped_reactive_walking as lrw

class controller():

    def __init__(self, dt, N_SIMULATION):

        self.dt = dt  # Time step

        self.invKin = Solo12InvKin(dt)  # Inverse Kinematics object
       
        self.box_qp = lrw.QPWBC()
        self.M = np.zeros((18, 18))
        self.Jc = np.zeros((12, 18))

        self.x = np.zeros(18)  # solution of WBC QP

        self.error = False  # Set to True when an error happens in the controller

        self.k_since_contact = np.zeros((1, 4))

        # Arrays to store results (for solo12)
        self.qdes = np.zeros((19, ))
        self.vdes = np.zeros((18, 1))
        self.tau_ff = np.zeros(12)
        self.qint = np.zeros((19, ))
        self.vint = np.zeros((18, 1))

        # Logging
        self.k_log = 0
        self.log_feet_pos = np.zeros((3, 4, N_SIMULATION))
        self.log_feet_err = np.zeros((3, 4, N_SIMULATION))
        self.log_feet_vel = np.zeros((3, 4, N_SIMULATION))
        self.log_feet_pos_target = np.zeros((3, 4, N_SIMULATION))
        self.log_feet_vel_target = np.zeros((3, 4, N_SIMULATION))
        self.log_feet_acc_target = np.zeros((3, 4, N_SIMULATION))
        self.log_x_cmd = np.zeros((12, N_SIMULATION))
        self.log_f_cmd = np.zeros((12, N_SIMULATION))
        self.log_f_out = np.zeros((12, N_SIMULATION))
        self.log_x = np.zeros((12, N_SIMULATION))
        self.log_q = np.zeros((18, N_SIMULATION))
        self.log_dq = np.zeros((18, N_SIMULATION))
        self.log_x_ref_invkin = np.zeros((6, N_SIMULATION))
        self.log_x_invkin = np.zeros((6, N_SIMULATION))
        self.log_dx_ref_invkin = np.zeros((6, N_SIMULATION))
        self.log_dx_invkin = np.zeros((6, N_SIMULATION))

        self.log_tau_ff = np.zeros((12, N_SIMULATION))
        self.log_qdes = np.zeros((12, N_SIMULATION))
        self.log_vdes = np.zeros((12, N_SIMULATION))
        self.log_q_pyb = np.zeros((19, N_SIMULATION))
        self.log_v_pyb = np.zeros((18, N_SIMULATION))

        self.log_contacts = np.zeros((4, N_SIMULATION))
        self.log_tstamps = np.zeros((N_SIMULATION))

    def compute(self, q, dq, x_cmd, f_cmd, contacts, planner):
        """ Call Inverse Kinematics to get an acceleration command then
        solve a QP problem to get the feedforward torques

        Args:
            q (19x1): Current state of the base
            dq (18x1): Current velocity of the base (in base frame)
            x_cmd (1x12): Position and velocity references from the mpc
            f_cmd (1x12): Contact forces references from the mpc
            contacts (1x4): Contact status of feet
            planner (object): Object that contains the pos, vel and acc references for feet
        """

        # Update nb of iterations since contact
        self.k_since_contact += contacts  # Increment feet in stance phase
        self.k_since_contact *= contacts  # Reset feet in swing phase

        # self.tic = time()
        # Compute Inverse Kinematics
        ddq_cmd = np.array([self.invKin.refreshAndCompute(q.copy(), dq.copy(), x_cmd, contacts, planner)]).T

        # Log position, velocity and acceleration references for the feet
        indexes = [10, 18, 26, 34]

        for i in range(4):
            self.log_feet_pos[:, i, self.k_log] = self.invKin.rdata.oMf[indexes[i]].translation
            self.log_feet_err[:, i, self.k_log] = self.invKin.feet_position_ref[i] - self.invKin.rdata.oMf[indexes[i]].translation # self.invKin.pfeet_err[i]
            self.log_feet_vel[:, i, self.k_log] = pin.getFrameVelocity(self.invKin.rmodel, self.invKin.rdata,
                                                                       indexes[i], pin.LOCAL_WORLD_ALIGNED).linear
        self.feet_pos = self.log_feet_pos[:, :, self.k_log]
        self.feet_err = self.log_feet_err[:, :, self.k_log]
        self.feet_vel = self.log_feet_vel[:, :, self.k_log]

        # + np.array([[0.0, 0.0, q[2, 0] - planner.h_ref]]).T
        self.log_feet_pos_target[:, :, self.k_log] = planner.goals[:, :]
        self.log_feet_vel_target[:, :, self.k_log] = planner.vgoals[:, :]
        self.log_feet_acc_target[:, :, self.k_log] = planner.agoals[:, :]

        #self.tac = time()

        # Solve QP problem with Python version
        """self.qp_wbc.compute(self.invKin.rmodel, self.invKin.rdata,
                            q.copy(), dq.copy(), ddq_cmd, np.array([f_cmd]).T, contacts)"""

        # Compute the joint space inertia matrix M by using the Composite Rigid Body Algorithm
        self.M = pin.crba(self.invKin.rmodel, self.invKin.rdata, q)

        # Compute Jacobian of contact points
        # Indexes of feet frames in this order: [FL, FR, HL, HR]
        indexes = [10, 18, 26, 34]
        self.Jc = np.zeros((12, 18))
        for i in range(4):
            if contacts[i]:
                self.Jc[(3*i):(3*(i+1)), :] = pin.getFrameJacobian(self.invKin.rmodel, self.invKin.rdata, indexes[i],
                                                                   pin.LOCAL_WORLD_ALIGNED)[:3, :]

        # Compute joint torques according to the current state of the system and the desired joint accelerations
        RNEA = pin.rnea(self.invKin.rmodel, self.invKin.rdata, q, dq, ddq_cmd)[:6]

        # Solve the QP problem with C++ bindings
        self.box_qp.run(self.M, self.Jc, f_cmd.reshape((-1, 1)), RNEA.reshape((-1, 1)), self.k_since_contact)

        # Add deltas found by the QP problem to reference quantities
        deltaddq = self.box_qp.get_ddq_res()
        self.f_with_delta = self.box_qp.get_f_res().reshape((-1, 1))
        ddq_with_delta = ddq_cmd.copy()
        ddq_with_delta[:6, 0] += deltaddq

        # Compute joint torques from contact forces and desired accelerations
        RNEA_delta = pin.rnea(self.invKin.rmodel, self.invKin.rdata, q, dq, ddq_with_delta)[6:]
        self.tau_ff[:] = RNEA_delta - ((self.Jc[:, 6:].transpose()) @ self.f_with_delta).ravel()

        """print("f python:", self.qp_wbc.f_cmd.ravel() + np.array(self.qp_wbc.x[6:]))
        print("f cpp   :", f_with_delta.ravel())

        print("ddq python:", (self.qp_wbc.ddq_cmd + self.qp_wbc.delta_ddq).ravel())
        print("ddq cpp   :", ddq_with_delta.ravel())"""

        #self.toc = time()

        # Retrieve desired positions and velocities
        self.vdes[:, 0] = self.invKin.dq_cmd  # (dq + ddq_cmd * self.dt).ravel()  # v des in world frame
        self.qdes[:] = self.invKin.q_cmd  # pin.integrate(self.invKin.robot.model, q, self.vdes * self.dt)

        # Double integration of ddq_cmd + delta_ddq
        # dq[2, 0] = 0.0 self.qdes[2]
        self.vint[:, 0] = (dq + ddq_cmd * self.dt).ravel()  # in world frame
        # self.vint[0:3, 0:1] = self.invKin.rot.transpose() @ self.vint[0:3, 0:1]  # velocity needs to be in base frame for pin.integrate
        # self.vint[3:6, 0:1] = self.invKin.rot.transpose() @ self.vint[3:6, 0:1]
        self.qint[:] = pin.integrate(self.invKin.robot.model, q, self.vint * self.dt)
        # self.qint[2] = planner.h_ref

        self.log_x_cmd[:, self.k_log] = x_cmd[:]  # Input of the WBC block (reference pos/ori/linvel/angvel)
        self.log_f_cmd[:, self.k_log] = f_cmd[:]  # Input of the WBC block (contact forces)
        self.log_f_out[:, self.k_log] = self.f_with_delta[:, 0]  # Input of the WBC block (contact forces)
        self.log_x[0:3, self.k_log] = self.qint[0:3]  # Output of the WBC block (pos)
        self.log_x[3:6, self.k_log] = quaternionToRPY(self.qint[3:7]).ravel()  # Output of the WBC block (ori)
        oMb = pin.SE3(pin.Quaternion(np.array([self.qint[3:7]]).transpose()), np.zeros((3, 1)))
        self.log_x[6:9, self.k_log] = oMb.rotation @ self.vint[0:3, 0]  # Output of the WBC block (lin vel)
        self.log_x[9:12, self.k_log] = oMb.rotation @ self.vint[3:6, 0]  # Output of the WBC block (ang vel)
        self.log_q[0:3, self.k_log] = q[0:3, 0]  # Input of the WBC block (current pos)
        self.log_q[3:6, self.k_log] = quaternionToRPY(q[3:7, 0]).ravel()  # Input of the WBC block (current ori)
        self.log_dq[:6, self.k_log] = dq[0:6, 0]  # Input of the WBC block (current linvel/angvel)

        self.log_q[6:, self.k_log] = q[7:, 0].copy()
        self.log_dq[6:, self.k_log] = dq[6:, 0].copy()

        self.log_x_ref_invkin[:, self.k_log] = self.invKin.x_ref[:, 0]  # Position task reference
        # Position task state (reconstruct with pin.forwardKinematics)
        self.log_x_invkin[:, self.k_log] = self.invKin.x[:, 0]
        self.log_dx_ref_invkin[:, self.k_log] = self.invKin.dx_ref[:, 0]  # Velocity task reference
        # Velocity task state (reconstruct with pin.forwardKinematics)
        self.log_dx_invkin[:, self.k_log] = self.invKin.dx[:, 0]

        self.log_tau_ff[:, self.k_log] = self.tau_ff[:]
        self.log_qdes[:, self.k_log] = self.qdes[7:]
        self.log_vdes[:, self.k_log] = self.vdes[6:, 0]
        self.log_contacts[:, self.k_log] = contacts

        self.log_tstamps[self.k_log] = clock()

        """from IPython import embed
        embed()"""

        self.k_log += 1

        #self.tuc = time()

        self.tic = 0.0
        self.tac = 0.0
        self.toc = 0.0
        self.tuc = 0.0

        return 0

    def show_logs(self):

        from matplotlib import pyplot as plt

        N = self.log_x_cmd.shape[1]
        t_range = np.array([k*self.dt for k in range(N)])

        index6 = [1, 3, 5, 2, 4, 6]
        index12 = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]

        lgd_X = ["FL", "FR", "HL", "HR"]
        lgd_Y = ["Pos X", "Pos Y", "Pos Z"]
        plt.figure()
        for i in range(12):
            if i == 0:
                ax0 = plt.subplot(3, 4, index12[i])
            else:
                plt.subplot(3, 4, index12[i], sharex=ax0)

            plt.plot(t_range, self.log_feet_pos[i % 3, np.int(i/3), :], color='b', linewidth=3, marker='')
            plt.plot(t_range, self.log_feet_err[i % 3, np.int(i/3), :], color='g', linewidth=3, marker='')
            plt.plot(t_range, self.log_feet_pos_target[i % 3, np.int(i/3), :], color='r', linewidth=3, marker='')
            plt.plot(t_range, self.log_contacts[np.int(
                i/3), :] * np.max(self.log_feet_pos[i % 3, np.int(i/3), :]), color='k', linewidth=3, marker='')
            plt.legend([lgd_Y[i % 3] + " " + lgd_X[np.int(i/3)]+"", "error",
                        lgd_Y[i % 3] + " " + lgd_X[np.int(i/3)]+" Ref", "Contact state"])
        plt.suptitle("Reference positions of feet (world frame)")

        lgd_X = ["FL", "FR", "HL", "HR"]
        lgd_Y = ["Vel X", "Vel Y", "Vel Z"]
        plt.figure()
        for i in range(12):
            if i == 0:
                ax0 = plt.subplot(3, 4, index12[i])
            else:
                plt.subplot(3, 4, index12[i], sharex=ax0)
                plt.plot(t_range, self.log_feet_vel[i % 3, np.int(i/3), :], color='b', linewidth=3, marker='')
            plt.plot(t_range, self.log_feet_vel_target[i % 3, np.int(i/3), :], color='r', linewidth=3, marker='')
            plt.legend([lgd_Y[i % 3] + " " + lgd_X[np.int(i/3)]+" Ref"])
        plt.suptitle("Current and reference velocities of feet (world frame)")

        lgd_X = ["FL", "FR", "HL", "HR"]
        lgd_Y = ["Acc X", "Acc Y", "Acc Z"]
        plt.figure()
        for i in range(12):
            if i == 0:
                ax0 = plt.subplot(3, 4, index12[i])
            else:
                plt.subplot(3, 4, index12[i], sharex=ax0)
            plt.plot(t_range, self.log_feet_acc_target[i % 3, np.int(i/3), :], color='r', linewidth=3, marker='')
            plt.legend([lgd_Y[i % 3] + " " + lgd_X[np.int(i/3)]+" Ref"])
        plt.suptitle("Current and reference accelerations of feet (world frame)")

        # LOG_Q
        lgd = ["Position X", "Position Y", "Position Z", "Position Roll", "Position Pitch", "Position Yaw"]
        plt.figure()
        for i in range(6):
            if i == 0:
                ax0 = plt.subplot(3, 2, index6[i])
            else:
                plt.subplot(3, 2, index6[i], sharex=ax0)
            plt.plot(t_range[:-2], self.log_x[i, :-2], "b", linewidth=2)
            plt.plot(t_range[:-2], self.log_x_cmd[i, :-2], "r", linewidth=3)
            # plt.plot(t_range, self.log_q[i, :], "grey", linewidth=4)
            plt.plot(t_range[:-2], self.log_x_invkin[i, :-2], "g", linewidth=2)
            plt.plot(t_range[:-2], self.log_x_ref_invkin[i, :-2], "violet", linewidth=2, linestyle="--")
            plt.legend(["WBC integrated output state", "Robot reference state",
                        "Task current state", "Task reference state"])
            plt.ylabel(lgd[i])

        # LOG_V
        lgd = ["Linear vel X", "Linear vel Y", "Linear vel Z",
               "Angular vel Roll", "Angular vel Pitch", "Angular vel Yaw"]
        plt.figure()
        for i in range(6):
            if i == 0:
                ax0 = plt.subplot(3, 2, index6[i])
            else:
                plt.subplot(3, 2, index6[i], sharex=ax0)
            plt.plot(t_range[:-2], self.log_x[i+6, :-2], "b", linewidth=2)
            plt.plot(t_range[:-2], self.log_x_cmd[i+6, :-2], "r", linewidth=3)
            # plt.plot(t_range, self.log_dq[i, :], "g", linewidth=2)
            plt.plot(t_range[:-2], self.log_dx_invkin[i, :-2], "g", linewidth=2)
            plt.plot(t_range[:-2], self.log_dx_ref_invkin[i, :-2], "violet", linewidth=2, linestyle="--")
            plt.legend(["WBC integrated output state", "Robot reference state",
                        "Task current state", "Task reference state"])
            plt.ylabel(lgd[i])

        plt.figure()
        plt.plot(t_range[:-2], self.log_x[6, :-2], "b", linewidth=2)
        plt.plot(t_range[:-2], self.log_x_cmd[6, :-2], "r", linewidth=2)
        plt.plot(t_range[:-2], self.log_dx_invkin[0, :-2], "g", linewidth=2)
        plt.plot(t_range[:-2], self.log_dx_ref_invkin[0, :-2], "violet", linewidth=2)
        plt.legend(["WBC integrated output state", "Robot reference state",
                    "Task current state", "Task reference state"])

        index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
        lgd1 = ["Tau 1", "Tau 2", "Tau 3"]
        lgd2 = ["FL", "FR", "HL", "HR"]
        plt.figure()
        for i in range(12):
            if i == 0:
                ax0 = plt.subplot(3, 4, index12[i])
            else:
                plt.subplot(3, 4, index12[i], sharex=ax0)
            h1, = plt.plot(t_range[:-2], self.log_tau_ff[i, :-2], "b", linewidth=3)
            plt.xlabel("Time [s]")
            plt.ylabel(lgd1[i % 3]+" "+lgd2[int(i/3)])
            plt.legend([h1], [lgd1[i % 3]+" "+lgd2[int(i/3)]])
            plt.ylim([-8.0, 8.0])
        plt.suptitle("Feedforward torques")

        lgd1 = ["Ctct force X", "Ctct force Y", "Ctct force Z"]
        lgd2 = ["FL", "FR", "HL", "HR"]
        plt.figure()
        for i in range(12):
            if i == 0:
                ax0 = plt.subplot(3, 4, index12[i])
            else:
                plt.subplot(3, 4, index12[i], sharex=ax0)
            h1, = plt.plot(t_range[:-2], self.log_f_cmd[i, :-2], "r", linewidth=3)
            h2, = plt.plot(t_range[:-2], self.log_f_out[i, :-2], "b", linewidth=3, linestyle="--")
            plt.xlabel("Time [s]")
            plt.ylabel(lgd1[i % 3]+" "+lgd2[int(i/3)])
            plt.legend([h1], [lgd1[i % 3]+" "+lgd2[int(i/3)]])
            if (i % 3) == 2:
                plt.ylim([-0.0, 26.0])
            else:
                plt.ylim([-26.0, 26.0])
        plt.suptitle("Contact forces (MPC command) & WBC QP output")

        index12 = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
        plt.figure()
        for i in range(12):
            if i == 0:
                ax0 = plt.subplot(3, 4, index12[i])
            else:
                plt.subplot(3, 4, index12[i], sharex=ax0)
            plt.plot(t_range, self.log_qdes[i, :], color='r', linewidth=3)
            plt.plot(t_range, self.log_q[6+i, :], color='b', linewidth=3)
            plt.legend(["Qdes"], prop={'size': 8})

        plt.show(block=True)

    def saveAll(self, fileName="data_QP", log_date=True):
        from datetime import datetime as datetime
        if log_date:
            date_str = datetime.now().strftime('_%Y_%m_%d_%H_%M')
        else:
            date_str = ""

        np.savez(fileName + date_str + ".npz",
                 log_feet_pos=self.log_feet_pos,
                 log_feet_pos_target=self.log_feet_pos_target,
                 log_feet_vel_target=self.log_feet_vel_target,
                 log_feet_acc_target=self.log_feet_acc_target,
                 log_x_cmd=self.log_x_cmd,
                 log_f_cmd=self.log_f_cmd,
                 log_f_out=self.log_f_out,
                 log_x=self.log_x,
                 log_q=self.log_q,
                 log_dq=self.log_dq,
                 log_x_ref_invkin=self.log_x_ref_invkin,
                 log_x_invkin=self.log_x_invkin,
                 log_dx_ref_invkin=self.log_dx_ref_invkin,
                 log_dx_invkin=self.log_dx_invkin,
                 log_tau_ff=self.log_tau_ff,
                 log_qdes=self.log_qdes,
                 log_vdes=self.log_vdes,
                 log_q_pyb=self.log_q_pyb,
                 log_v_pyb=self.log_v_pyb,
                 log_tstamps=self.log_tstamps)
