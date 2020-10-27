
from utils_mpc import quaternionToRPY
import numpy as np
import scipy as scipy
import pinocchio as pin
from example_robot_data import load
import osqp as osqp
from solo12InvKin import Solo12InvKin


class controller():

    def __init__(self, dt, N_SIMULATION):

        self.dt = dt  # Time step

        self.invKin = Solo12InvKin(dt)  # Inverse Kinematics object
        self.qp_wbc = QP_WBC()  # QP of the WBC

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
        self.log_feet_pos_target = np.zeros((3, 4, N_SIMULATION))
        self.log_feet_vel_target = np.zeros((3, 4, N_SIMULATION))
        self.log_feet_acc_target = np.zeros((3, 4, N_SIMULATION))
        self.log_x_cmd = np.zeros((12, N_SIMULATION))
        self.log_x = np.zeros((12, N_SIMULATION))
        self.log_q = np.zeros((6, N_SIMULATION))
        self.log_dq = np.zeros((6, N_SIMULATION))
        self.log_x_ref_invkin = np.zeros((6, N_SIMULATION))
        self.log_x_invkin = np.zeros((6, N_SIMULATION))
        self.log_dx_ref_invkin = np.zeros((6, N_SIMULATION))
        self.log_dx_invkin = np.zeros((6, N_SIMULATION))

    def compute(self, q, dq, o_dq, x_cmd, f_cmd, contacts, planner):
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

        # Compute Inverse Kinematics
        ddq_cmd = np.array([self.invKin.refreshAndCompute(q.copy(), dq.copy(), x_cmd, contacts, planner)]).T

        # Solve QP problem
        self.qp_wbc.compute(self.invKin.robot.model, self.invKin.robot.data,
                            q.copy(), dq.copy(), ddq_cmd, np.array([f_cmd]).T, contacts)

        """if dq[0, 0] > 0.4:
            from IPython import embed
            embed()"""

        # Retrieve joint torques
        self.tau_ff[:] = self.qp_wbc.get_joint_torques().ravel()

        # Retrieve desired positions and velocities
        self.vdes[:, 0] = self.invKin.dq_cmd  # (dq + ddq_cmd * self.dt).ravel()  # v des in world frame
        self.qdes[:] = self.invKin.q_cmd  # pin.integrate(self.invKin.robot.model, q, self.vdes * self.dt)

        # Double integration of ddq_cmd + delta_ddq
        self.vint[:, 0] = (dq + (ddq_cmd + 0.0 * self.qp_wbc.delta_ddq) * self.dt).ravel()  # in world frame
        # self.vint[0:3, 0:1] = self.invKin.rot.transpose() @ self.vint[0:3, 0:1]  # velocity needs to be in base frame for pin.integrate
        # self.vint[3:6, 0:1] = self.invKin.rot.transpose() @ self.vint[3:6, 0:1]
        self.qint[:] = pin.integrate(self.invKin.robot.model, q, self.vint * self.dt)

        # Log position, velocity and acceleration references for the feet
        indexes = [10, 18, 26, 34]
        for i in range(4):
            self.log_feet_pos[:, i, self.k_log] = self.invKin.robot.data.oMf[indexes[i]].translation
        self.log_feet_pos_target[:, :, self.k_log] = planner.goals[:, :]
        self.log_feet_vel_target[:, :, self.k_log] = planner.vgoals[:, :]
        self.log_feet_acc_target[:, :, self.k_log] = planner.agoals[:, :]

        self.log_x_cmd[:, self.k_log] = x_cmd[:]  # Input of the WBC block (reference pos/ori/linvel/angvel)
        self.log_x[0:3, self.k_log] = self.qint[0:3]  # Output of the WBC block (pos)
        self.log_x[3:6, self.k_log] = quaternionToRPY(self.qint[3:7]).ravel()  # Output of the WBC block (ori)
        oMb = pin.SE3(pin.Quaternion(np.array([self.qint[3:7]]).transpose()), np.zeros((3, 1)))
        self.log_x[6:9, self.k_log] = oMb.rotation @ self.vint[0:3, 0]  # Output of the WBC block (lin vel)
        self.log_x[9:12, self.k_log] = oMb.rotation @ self.vint[3:6, 0]  # Output of the WBC block (ang vel)
        self.log_q[0:3, self.k_log] = q[0:3, 0]  # Input of the WBC block (current pos)
        self.log_q[3:6, self.k_log] = quaternionToRPY(q[3:7]).ravel()  # Input of the WBC block (current ori)
        self.log_dq[:, self.k_log] = dq[0:6, 0]  # Input of the WBC block (current linvel/angvel)

        self.log_x_ref_invkin[:, self.k_log] = self.invKin.x_ref[:, 0]  # Position task reference
        # Position task state (reconstruct with pin.forwardKinematics)
        self.log_x_invkin[:, self.k_log] = self.invKin.x[:, 0]
        self.log_dx_ref_invkin[:, self.k_log] = self.invKin.dx_ref[:, 0]  # Velocity task reference
        # Velocity task state (reconstruct with pin.forwardKinematics)
        self.log_dx_invkin[:, self.k_log] = self.invKin.dx[:, 0]

        """if dq[0, 0] > 0.02:
            from IPython import embed
            embed()"""

        self.k_log += 1

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
            plt.plot(t_range, self.log_feet_pos_target[i % 3, np.int(i/3), :], color='r', linewidth=3, marker='')
            plt.legend([lgd_Y[i % 3] + " " + lgd_X[np.int(i/3)]+"", lgd_Y[i % 3] + " " + lgd_X[np.int(i/3)]+" Ref"])
        plt.suptitle("Reference positions of feet (world frame)")

        lgd_X = ["FL", "FR", "HL", "HR"]
        lgd_Y = ["Vel X", "Vel Y", "Vel Z"]
        plt.figure()
        for i in range(12):
            if i == 0:
                ax0 = plt.subplot(3, 4, index12[i])
            else:
                plt.subplot(3, 4, index12[i], sharex=ax0)
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
            # plt.plot(t_range, self.log_q[i, :], "g", linewidth=2)
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

        plt.show(block=True)


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
                self.JcT[:, (3*i):(3*(i+1))] = pin.computeFrameJacobian(model,
                                                                        data, q, indexes[i], pin.LOCAL_WORLD_ALIGNED)[:3, :].transpose()

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
            self.initialized = True
        else:  # Code to update the QP problem without creating it again
            try:
                self.prob.update(
                    Ax=self.ML.data, l=self.NK_inf, u=self.NK.ravel())
            except ValueError:
                print("Bound Problem")
            self.prob.warm_start(x=self.warmxf)

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


if __name__ == "__main__":

    # Load the URDF model
    robot = load('solo12')
    model = robot.model
    data = robot.data

    # Create the QP object
    qp_wbc = QP_WBC(model, data)

    # Position and velocity
    q = np.array([[0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0, 0.8, -1.6,
                   0, 0.8, -1.6, 0, -0.8, 1.6, 0, -0.8, 1.6]]).transpose()
    v = np.zeros((18, 1))

    # Commands
    ddq_cmd = np.zeros((18, 1))
    f_cmd = np.array([[0.0, 0.0, 6.0] * 4]).transpose()

    # Update the matrices, the four feet are in contact
    qp_wbc.update_ML(q, [0, 1, 2, 3])
    qp_wbc.update_NK(q, v, ddq_cmd, f_cmd)
    qp_wbc.call_solver()

    # Display results
    print("######")
    print("ddq_cmd: ", ddq_cmd.ravel())
    print("ddq_out: ", qp_wbc.x[:6].ravel())
    print("f_cmd: ", f_cmd.ravel())
    print("f_out: ", qp_wbc.x[6:].ravel())
    print("torques: ", qp_wbc.get_joint_torques().ravel())
