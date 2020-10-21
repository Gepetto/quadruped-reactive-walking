
import numpy as np
import scipy as scipy
import pinocchio as pin
from example_robot_data import load
import osqp as osqp


class QP_WBC():

    def __init__(self, model, data):

        self.model = model
        self.data = data

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
            self.ML_full[(6+5*i):(6+5*(i+1)), (6+3*i):(6+3*(i+1))] = self.C

        # NK matrix
        self.NK = np.zeros((6 + 20, 1))

        # NK_inf is the lower bound
        self.NK_inf = np.zeros((6 + 20, ))
        self.inf_lower_bound = -np.inf * np.ones((20,))
        self.inf_lower_bound[4::5] = - 25.0  # - maximum normal force
        self.NK_inf[:6] = self.NK[:6, 0]
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
        P_data = 1.0 * np.ones((n_x,))
        P_data[6:] = 0.01  # weight for forces

        # Convert P into a csc matrix for the solver
        self.P = scipy.sparse.csc.csc_matrix(
            (P_data, (P_row, P_col)), shape=(n_x, n_x))

        # Declaration of the Q matrix in "x^T.P.x + x^T.Q"
        self.Q = np.zeros(n_x,)

        return 0

    def update_ML(self, q, contacts):
        """Update the M and L matrices involved in the MPC constraint equations M.X = N and L.X <= K
        """

        # Compute the joint space inertia matrix M by using the Composite Rigid Body Algorithm
        self.A = pin.crba(self.model, self.data, q)

        # Indexes of feet frames in this order: [FL, FR, HL, HR]
        indexes = [10, 18, 26, 34]

        # Contact Jacobian
        self.JcT = np.zeros((18, 12))
        for i in contacts:
            self.JcT[:, (3*i):(3*(i+1))] = pin.computeFrameJacobian(model,
                                                                    data, q, indexes[i], pin.WORLD)[:3, :].transpose()

        self.ML_full[:6, :6] = - self.A[:6, :6]
        self.ML_full[:6, 6:] = self.JcT[:6, :]

        # Update solver matrix
        self.ML.data[:] = self.ML_full.ravel(order="F")

        return 0

    def update_NK(self, q, v, ddq_cmd, f_cmd):
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

    def call_solver(self, k):
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
        if k == 0:  # Setup the solver with the matrices
            self.prob.setup(P=self.P, q=self.Q, A=self.ML,
                            l=self.NK_inf, u=self.NK.ravel(), verbose=False)
            self.prob.update_settings(eps_abs=1e-5)
            self.prob.update_settings(eps_rel=1e-5)

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

        return 0

    def get_joint_torques(self):

        delta_q = np.zeros((18, 1))
        delta_q[:6, 0] = qp_wbc.x[:6]

        return (self.A @ (self.ddq_cmd + delta_q) + np.array([self.NLE]).transpose()
                - self.JcT @ (self.f_cmd + np.array([qp_wbc.x[6:]]).transpose()))[6:, :]


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
    qp_wbc.call_solver(0)

    # Display results
    print("######")
    print("ddq_cmd: ", ddq_cmd.ravel())
    print("ddq_out: ", qp_wbc.x[:6].ravel())
    print("f_cmd: ", f_cmd.ravel())
    print("f_out: ", qp_wbc.x[6:].ravel())
    print("torques: ", qp_wbc.get_joint_torques().ravel())
