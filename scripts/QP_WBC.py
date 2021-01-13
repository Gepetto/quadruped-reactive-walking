
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
        self.qp_wbc = QP_WBC()  # QP of the WBC
        
        self.box_qp = lrw.QPWBC()
        self.M = np.zeros((18, 18))
        self.Jc = np.zeros((12, 18))

        self.x = np.zeros(18)  # solution of WBC QP

        self.error = False  # Set to True when an error happens in the controller

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
        self.log_q = np.zeros((6, N_SIMULATION))
        self.log_dq = np.zeros((6, N_SIMULATION))
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

        #self.tic = time()
        # Compute Inverse Kinematics
        ddq_cmd = np.array([self.invKin.refreshAndCompute(q.copy(), dq.copy(), x_cmd, contacts, planner)]).T

        # Log position, velocity and acceleration references for the feet
        """indexes = [10, 18, 26, 34]
        for i in range(4):
            self.log_feet_pos[:, i, self.k_log] = self.invKin.rdata.oMf[indexes[i]].translation
            self.log_feet_err[:, i, self.k_log] = self.invKin.pfeet_err[i]
            self.log_feet_vel[:, i, self.k_log] = pin.getFrameVelocity(self.invKin.rmodel, self.invKin.rdata,
                                                                       indexes[i], pin.LOCAL_WORLD_ALIGNED).linear
        # + np.array([[0.0, 0.0, q[2, 0] - planner.h_ref]]).T
        self.log_feet_pos_target[:, :, self.k_log] = planner.goals[:, :]
        self.log_feet_vel_target[:, :, self.k_log] = planner.vgoals[:, :]
        self.log_feet_acc_target[:, :, self.k_log] = planner.agoals[:, :]"""

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
        self.box_qp.run(self.M, self.Jc, f_cmd.reshape((-1, 1)), RNEA.reshape((-1, 1)))

        # Add deltas found by the QP problem to reference quantities
        deltaddq = self.box_qp.get_ddq_res()
        f_with_delta = self.box_qp.get_f_res().reshape((-1, 1))
        ddq_with_delta = ddq_cmd.copy()
        ddq_with_delta[:6, 0] += deltaddq

        # Compute joint torques from contact forces and desired accelerations
        RNEA_delta = pin.rnea(self.invKin.rmodel, self.invKin.rdata, q, dq, ddq_with_delta)[6:]
        self.tau_ff[:] = RNEA_delta - ((self.Jc[:, 6:].transpose()) @ f_with_delta).ravel()

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
        self.log_f_out[:, self.k_log] = f_cmd[:] + np.array(self.x[6:])  # Input of the WBC block (contact forces)
        self.log_x[0:3, self.k_log] = self.qint[0:3]  # Output of the WBC block (pos)
        self.log_x[3:6, self.k_log] = quaternionToRPY(self.qint[3:7]).ravel()  # Output of the WBC block (ori)
        oMb = pin.SE3(pin.Quaternion(np.array([self.qint[3:7]]).transpose()), np.zeros((3, 1)))
        self.log_x[6:9, self.k_log] = oMb.rotation @ self.vint[0:3, 0]  # Output of the WBC block (lin vel)
        self.log_x[9:12, self.k_log] = oMb.rotation @ self.vint[3:6, 0]  # Output of the WBC block (ang vel)
        self.log_q[0:3, self.k_log] = q[0:3, 0]  # Input of the WBC block (current pos)
        self.log_q[3:6, self.k_log] = quaternionToRPY(q[3:7, 0]).ravel()  # Input of the WBC block (current ori)
        self.log_dq[:, self.k_log] = dq[0:6, 0]  # Input of the WBC block (current linvel/angvel)

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


class QP_WBC():

    def __init__(self):

        # Set to True after the creation of the QP problem during the first call of the solver
        self.initialized = False

        # Friction coefficient
        self.mu = 0.9

        # QP OSQP object
        self.prob = osqp.OSQP()

        # ML matrix
        self.ML = scipy.sparse.csc.csc_matrix(
            np.ones((6 + 20, 18)), shape=(6 + 20, 18))
        self.ML_full = np.zeros((6 + 20, 18))

        self.C = np.zeros((5, 3))  # Force constraints
        self.C[[0, 1, 2, 3] * 2 + [4], [0, 0, 1, 1, 2, 2, 2, 2, 2]
               ] = np.array([1, -1, 1, -1, -self.mu, -self.mu, -self.mu, -self.mu, -1])
        for i in range(4):
            self.ML_full[(6+5*i): (6+5*(i+1)), (6+3*i): (6+3*(i+1))] = self.C

        # Relaxation of acceleration
        self.delta_ddq = np.zeros((18, 1))

        # NK matrix
        self.NK = np.zeros((6 + 20, 1))

        # NK_inf is the lower bound
        self.NK_inf = np.zeros((6 + 20, ))
        self.inf_lower_bound = -np.inf * np.ones((20,))
        self.inf_lower_bound[4:: 5] = - 25.0  # - maximum normal force
        self.NK_inf[: 6] = self.NK[: 6, 0]
        self.NK_inf[6:] = self.inf_lower_bound

        # Mass matrix
        self.A = np.zeros((18, 18))

        # Create weight matrices
        self.create_weight_matrices()

    def create_weight_matrices(self):
        """Create the weight matrices P and q in the cost function x^T.P.x + x^T.q of the QP problem
        """

        # Number of states
        n_x = 18

        # Declaration of the P matrix in "x^T.P.x + x^T.q"
        # P_row, _col and _data satisfy the relationship P[P_row[k], P_col[k]] = P_data[k]
        P_row = np.array([], dtype=np.int64)
        P_col = np.array([], dtype=np.int64)
        P_data = np.array([], dtype=np.float64)

        # Define weights for the x-x_ref components of the optimization vector
        P_row = np.arange(0, n_x, 1)
        P_col = np.arange(0, n_x, 1)
        P_data = 0.1 * np.ones((n_x,))
        P_data[6:] = 1  # weight for forces

        # Convert P into a csc matrix for the solver
        self.P = scipy.sparse.csc.csc_matrix(
            (P_data, (P_row, P_col)), shape=(n_x, n_x))

        # Declaration of the Q matrix in "x^T.P.x + x^T.Q"
        self.Q = np.zeros(n_x,)

        return 0

    def update_ML(self, model, data, q, contacts):
        """Update the M and L matrices involved in the MPC constraint equations M.X = N and L.X <= K
        """

        # Compute the joint space inertia matrix M by using the Composite Rigid Body Algorithm
        self.A = pin.crba(model, data, q)

        # Indexes of feet frames in this order: [FL, FR, HL, HR]
        indexes = [10, 18, 26, 34]

        # Contact Jacobian
        self.JcT = np.zeros((18, 12))
        for i in range(4):
            if contacts[i]:
                self.JcT[:, (3*i):(3*(i+1))] = pin.getFrameJacobian(model, data, indexes[i],
                                                                    pin.LOCAL_WORLD_ALIGNED)[:3, :].transpose()
        self.ML_full[:6, :6] = - self.A[:6, :6]
        self.ML_full[:6, 6:] = self.JcT[:6, :]

        # Update solver matrix
        self.ML.data[:] = self.ML_full.ravel(order="F")

        return 0

    def update_NK(self, model, data, q, v, ddq_cmd, f_cmd):
        """Update the N and K matrices involved in the MPC constraint equations M.X = N and L.X <= K
        """

        # Save acceleration command for torque computation
        self.ddq_cmd = ddq_cmd

        # Save force command for torque computation
        self.f_cmd = f_cmd

        # Compute the non-linear effects (Corriolis, centrifual and gravitationnal effects)
        self.NLE = pin.nonLinearEffects(model, data, q, v)
        self.NK[:6, 0] = self.NLE[:6]

        # Add reference accelerations
        self.NK[:6, :] += self.A[:6, :] @ ddq_cmd

        # Remove reference forces
        self.NK[:6, :] -= self.JcT[:6, :12] @ f_cmd

        return 0

    def call_solver(self):
        """Create an initial guess and call the solver to solve the QP problem

        Args:
            k (int): number of MPC iterations since the start of the simulation
        """

        # Initial guess
        self.warmxf = np.zeros((18, ))

        # OSQP solves a problem with constraints of the form inf_bound <= A X <= sup_bound
        # We defined the problem as having constraints M.X = N and L.X <= K

        # Copy the "equality" part of NK on the other side of the constaint
        self.NK_inf[:6] = self.NK[:6, 0]

        # That way we have N <= M.X <= N for the upper part
        # And - infinity <= L.X <= K for the lower part

        # Setup the solver (first iteration) then just update it
        if not self.initialized:  # Setup the solver with the matrices
            self.prob.setup(P=self.P, q=self.Q, A=self.ML,
                            l=self.NK_inf, u=self.NK.ravel(), verbose=False)
            self.prob.update_settings(eps_abs=1e-5)
            self.prob.update_settings(eps_rel=1e-5)
            # self.prob.update_settings(time_limit=5e-4)
            self.initialized = True
        else:  # Code to update the QP problem without creating it again
            try:
                self.prob.update(
                    Ax=self.ML.data, l=self.NK_inf, u=self.NK.ravel())
            except ValueError:
                print("Bound Problem")
            self.prob.warm_start(x=self.x)

        # Run the solver to solve the QP problem
        self.sol = self.prob.solve()
        self.x = self.sol.x

        """from IPython import embed
        embed()"""

        return 0

    def compute(self, model, data, q, dq, ddq_cmd, f_cmd, contacts):

        self.update_ML(model, data, q, contacts)
        self.update_NK(model, data, q, dq, ddq_cmd, f_cmd)
        self.call_solver()

        return 0

    def get_joint_torques(self):

        self.delta_ddq[:6, 0] = self.x[:6]

        return (self.A @ (self.ddq_cmd + self.delta_ddq) + np.array([self.NLE]).transpose()
                - self.JcT @ (self.f_cmd + np.array([self.x[6:]]).transpose()))[6:, ]


class BOX_QP_WBC():

    def __init__(self):

        # Set to True after the creation of the QP problem during the first call of the solver
        self.initialized = False

        # Weight matrices
        self.Q1 = 0.1 * np.eye(6)
        self.Q2 = 1.0 * np.eye(12)

        # Friction coefficient
        self.mu = 0.9

        # Generatrix matrix
        self.Gk = np.array([[self.mu, self.mu, -self.mu, -self.mu],
                           [self.mu, -self.mu, self.mu, -self.mu],
                           [1.0, 1.0, 1.0, 1.0]])
        self.G = np.zeros((12, 16))
        for k in range(4):
            self.G[(3*k):(3*(k+1)), (4*k):(4*(k+1))] = self.Gk

        # QP OSQP object
        self.prob = osqp.OSQP()

        # ML matrix
        self.ML = scipy.sparse.csc.csc_matrix(
            np.ones((6 + 20, 18)), shape=(6 + 20, 18))
        self.ML_full = np.zeros((6 + 20, 18))

        self.C = np.zeros((5, 3))  # Force constraints
        self.C[[0, 1, 2, 3] * 2 + [4], [0, 0, 1, 1, 2, 2, 2, 2, 2]
               ] = np.array([1, -1, 1, -1, -self.mu, -self.mu, -self.mu, -self.mu, -1])
        for i in range(4):
            self.ML_full[(6+5*i): (6+5*(i+1)), (6+3*i): (6+3*(i+1))] = self.C

        # Relaxation of acceleration
        self.delta_ddq = np.zeros((18, 1))

        # NK matrix
        self.NK = np.zeros((6 + 20, 1))

        # NK_inf is the lower bound
        self.NK_inf = np.zeros((6 + 20, ))
        self.inf_lower_bound = -np.inf * np.ones((20,))
        self.inf_lower_bound[4:: 5] = - 25.0  # - maximum normal force
        self.NK_inf[: 6] = self.NK[: 6, 0]
        self.NK_inf[6:] = self.inf_lower_bound

        # Mass matrix
        self.M = np.zeros((18, 18))
        self.RNEA = np.zeros((6, 1))

        # Indexes of feet frames in this order: [FL, FR, HL, HR]
        self.indexes = [10, 18, 26, 34]

        # Create weight matrices
        self.create_weight_matrices()

    def compute_matrices(self, model, data, q, v, ddq_cmd, f_cmd, contacts):
        """TODO
        """

        # Compute the joint space inertia matrix M by using the Composite Rigid Body Algorithm
        self.M = pin.crba(model, data, q)

        # Contact Jacobian
        self.Jc = np.zeros((12, 18))
        for i in range(4):
            if contacts[i]:
                self.Jc[(3*i):(3*(i+1)), :] = pin.getFrameJacobian(model, data, self.indexes[i],
                                                                   pin.LOCAL_WORLD_ALIGNED)[:3, :]

        # Compute all matrices of the Box QP problem
        self.f_cmd = f_cmd
        self.Y = self.M[:6, :6]
        self.X = self.Jc[:, :6].transpose()
        self.Yinv = np.linalg.pinv(self.Y)
        self.A = self.Yinv @ self.X
        self.RNEA[:, 0] = pin.rnea(model, data, q, v, ddq_cmd)[:6]
        self.gamma = self.Yinv @ ((self.X @ f_cmd) - self.RNEA)
        self.H = (self.A.transpose() @ self.Q1) @ self.A + self.Q2
        self.g = (self.A.transpose() @ self.Q1) @ self.gamma
        self.Pw = (self.G.transpose() @ self.H) @ self.G
        self.Qw = (self.G.transpose() @ self.g) - ((self.G.transpose() @ self.H) @ f_cmd)

        """from IPython import embed
        embed()"""

        return 0

    def create_weight_matrices(self):
        """Create the weight matrices P and q in the cost function 1/2 x^T.P.x + x^T.q of the QP problem
        """

        # Number of states
        n_x = 16

        # Declaration of the P matrix in "x^T.P.x + x^T.q"
        # P_row, _col and _data satisfy the relationship P[P_row[k], P_col[k]] = P_data[k]
        P_row = np.array([], dtype=np.int64)
        P_col = np.array([], dtype=np.int64)
        P_data = np.array([], dtype=np.float64)

        # Define weights for the x-x_ref components of the optimization vector
        P_row = np.tile(np.arange(0, n_x, 1), n_x)
        P_col = np.repeat(np.arange(0, n_x, 1), n_x)
        P_data = np.ones((n_x*n_x,))

        # Convert P into a csc matrix for the solver
        self.P = scipy.sparse.csc.csc_matrix(
            (P_data, (P_row, P_col)), shape=(n_x, n_x))

        # Declaration of the Q matrix in "x^T.P.x + x^T.Q"
        self.Q = np.zeros(n_x,)

        self.box_sup = 25.0 * np.ones((16, ))
        self.box_low = np.zeros((16, ))

        return 0

    def update_matrices(self):
        """Update the weight matrices P and q in the cost function 1/2 x^T.P.x + x^T.q of the QP problem
        """

        #if self.initialized:
        self.Pw[np.abs(self.Pw) < 1e-6] = 0.0
        self.Qw[np.abs(self.Qw) < 1e-6] = 0.0

        self.P.data[:] = self.Pw.ravel(order="F")
        self.Q[:] = self.Qw.ravel()

    def call_solver(self):
        """Create an initial guess and call the solver to solve the QP problem

        Args:
            k (int): number of MPC iterations since the start of the simulation
        """

        """from IPython import embed
        embed()"""
        # Setup the solver (first iteration) then just update it
        if not self.initialized:  # Setup the solver with the matrices
            self.prob.setup(P=self.P, q=self.Q, A=scipy.sparse.csc.csc_matrix(np.eye(16), shape=(16, 16)),
                            l=self.box_low, u=self.box_sup, verbose=False)
            #self.prob.update_settings(eps_abs=1e-5)
            #self.prob.update_settings(eps_rel=1e-5)
            # self.prob.update_settings(time_limit=5e-4)
            self.initialized = True
            """from IPython import embed
            embed()
            self.update_matrices()
            self.prob.update(Px=self.P.data, Px_idx=np.arange(0, len(self.P.data), 1), q=self.Q[:])"""
        else:  # Code to update the QP problem without creating it again
            try:
                #self.prob.update(Px=self.P.data, Px_idx=np.arange(0, len(self.P.data), 1), q=self.Q[:])
                self.prob.setup(P=self.P, q=self.Q, A=scipy.sparse.csc.csc_matrix(np.eye(16), shape=(16, 16)),
                            l=self.box_low, u=self.box_sup, verbose=False)
                a = 1
            except ValueError:
                print("Bound Problem")
            # self.prob.warm_start(x=self.x)

        # Run the solver to solve the QP problem
        self.sol = self.prob.solve()
        self.x = self.sol.x

        self.f_res = self.G @ self.x.reshape((16, 1))
        self.ddq_res = self.A @ (self.f_res - self.f_cmd) + self.gamma
        """from IPython import embed
        embed()"""
        print(self.f_res.ravel())
        print(self.x.ravel())
        """from IPython import embed
        embed()"""

        return 0

    def compute(self, model, data, q, dq, ddq_cmd, f_cmd, contacts):

        self.compute_matrices(model, data, q, dq, ddq_cmd, f_cmd, contacts)
        self.update_matrices()
        self.call_solver()

        return 0

    def get_joint_torques(self, model, data, q, v, ddq_cmd):

        self.delta_ddq[:6, 0:1] = self.ddq_res

        return (pin.rnea(model, data, q, v, ddq_cmd+self.delta_ddq) + (self.Jc.transpose() @ self.f_res).ravel())[6:, ]

        """self.delta_ddq[:6, 0] = self.x[:6]

        return (self.A @ (self.ddq_cmd + self.delta_ddq) + np.array([self.NLE]).transpose()
                - self.JcT @ (self.f_cmd + np.array([self.x[6:]]).transpose()))[6:, ]"""


if __name__ == "__main__":

    # Load the URDF model
    robot = load('solo12')
    model = robot.model
    data = robot.data

    # Create the QP object
    qp_wbc = QP_WBC()

    # Position and velocity
    q = np.array([[0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0, 0.8, -1.6,
                   0, 0.8, -1.6, 0, -0.8, 1.6, 0, -0.8, 1.6]]).transpose()
    v = np.zeros((18, 1))

    # Commands
    ddq_cmd = np.zeros((18, 1))
    f_cmd = np.array([[0.0, 0.0, 6.0] * 4]).transpose()

    # Update the matrices, the four feet are in contact
    qp_wbc.update_ML(model, data, q, np.array([1, 1, 1, 1]))
    qp_wbc.update_NK(model, data, q, v, ddq_cmd, f_cmd)
    qp_wbc.call_solver()

    # Display results
    print("######")
    print("ddq_cmd: ", ddq_cmd.ravel())
    print("ddq_out: ", qp_wbc.x[:6].ravel())
    print("f_cmd: ", f_cmd.ravel())
    print("f_out: ", f_cmd.ravel() + qp_wbc.x[6:].ravel())
    print("torques: ", qp_wbc.get_joint_torques().ravel())
