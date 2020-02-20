# coding: utf8

import numpy as np
from time import clock
import scipy as scipy
import osqp as osqp
from matplotlib import pyplot as plt


class MpcSolver:
    """Wrapper for the MPC to create constraint matrices, call the QP solver and
    retrieve the result.

    """

    def __init__(self, dt, S, k_max_loop):

        # Time step of the MPC solver
        self.dt = dt

        # Reference trajectory matrix of size 12 by (1 + N)  with the current state of
        # the robot in column 0 and the N steps of the prediction horizon in the others
        self.xref = np.zeros((12, 1 + S.shape[0]))

        # Number of states (6 positions + 6 velocity)
        self.n_x = 12

        # Forces have 3 components (fx, fy, fz)
        self.n_f = 3

        # Current state vector of the robot
        self.x0 = np.zeros((self.n_x, 1))

        # Constraints of the QP solver are A.X == b and G.X <= h
        self.A = 0
        self.b = 0
        self.G = 0
        self.h = 0

        # Weight matrices of the QP solver associated with the cost x^T.P.x + x^T.q
        self.P = 0
        self.q = 0

        # Solution of the QP problem
        self.x = 0

        # "robot state vector" part of the solution of the QP problem
        self.x_robot = np.zeros(self.xref.shape)

        # "contact forces" part of the solution of the QP problem
        self.f_applied = 0

        # Predicted position and velocity of the robot during the next time step
        self.qu = np.zeros((6, 1))
        self.vu = np.zeros((6, 1))

        # To display a trail behind the robot in the viewer as it moves around
        self.trail = np.zeros((3, k_max_loop))
        self.k_trail = 0

        # Avoid redundant computation for the creation of matrices

        # Inertia matrix of the robot in body frame (found in urdf)
        gI = np.diag([0.00578574, 0.01938108, 0.02476124])

        # Inverting the inertia matrix in local frame
        self.gI_inv = np.linalg.inv(gI)

        # Mass of the quadruped in [kg] (found in urdf)
        self.m = 3.0  # 2.2 in urdf for the trunk

    def getRefStatesDuringTrajectory(self, settings):
        """Returns the reference trajectory of the robot for each time step of the
        predition horizon. The ouput is a matrix of size 12 by N with N the number
        of time steps (around T_gait / dt) and 12 the position / orientation /
        linear velocity / angular velocity vertically stacked.

        Keyword arguments:
        qu -- current position/orientation of the robot (6 by 1)
        v_ref -- reference velocity vector of the flying base (6 by 1, linear and angular stacked)
        dt -- time step
        T_gait -- period of the current gait
        """

        n_steps = int(np.round(settings.T_gait/self.dt))
        qu_ref = np.zeros((6, n_steps))

        dt_vector = np.linspace(settings.dt, settings.T_gait, n_steps)
        qu_ref = settings.v_ref_world * dt_vector

        yaw = np.linspace(0, settings.T_gait-self.dt, n_steps) * settings.v_ref_world[5, 0]
        qu_ref[0, :] = self.dt * np.cumsum(settings.v_ref_world[0, 0] * np.cos(yaw) -
                                           settings.v_ref_world[1, 0] * np.sin(yaw))
        qu_ref[1, :] = self.dt * np.cumsum(settings.v_ref_world[0, 0] * np.sin(yaw) +
                                           settings.v_ref_world[1, 0] * np.cos(yaw))

        # Stack the reference velocity to the reference position to get the reference state vector
        settings.x_ref = np.vstack((qu_ref, np.tile(settings.v_ref_world, (1, n_steps))))

        # Desired height is supposed constant
        settings.x_ref[2, :] = settings.h_ref

        # Stack the reference trajectory (future states) with the current state
        self.xref[6:, 0:1] = settings.v_ref
        self.xref[:, 1:] = settings.x_ref
        self.xref[2, 0] = settings.h_ref

        # Current state vector of the robot
        self.x0 = np.vstack((settings.qu_m, settings.vu_m))

        return 0

    def retrieve_data(self, fstep_planner, ftraj_gen):
        """Retrieve footsteps information from the FootstepPlanner
        and the FootTrajectoryGenerator

        Keyword arguments:
        fstep_planner -- FootstepPlanner object
        ftraj_gen -- FootTrajectoryGenerator object
        """

        self.footholds = ftraj_gen.desired_pos[0:2, :]
        self.footholds_lock = ftraj_gen.footsteps_lock
        self.footholds_no_lock = fstep_planner.footsteps

        # Information in world frame for visualisation purpose
        self.footholds_world = ftraj_gen.desired_pos_world

        return 0

    def create_constraints_matrices(self, settings, solo, k_loop):

        enable_timer = False

        t_test = clock()

        footholds_m = self.footholds.copy()

        update = np.array(settings.S[0]).ravel() == 0
        if np.any(update):
            footholds_m[:, update] = self.footholds_lock[:, update]

        # Number of timesteps in the prediction horizon
        nb_xf = settings.n_contacts.shape[0]

        """# Inertia matrix of the robot in body frame (found in urdf)
        self.gI = np.diag([0.00578574, 0.01938108, 0.02476124])

        # Inverting the inertia matrix in the global frame
        # R_gI = getRotMatrix(xref[3:6, 0:1])
        # gI_inv = np.linalg.inv(R_gI * gI)

        # Inverting the inertia matrix in local frame
        self.gI_inv = np.linalg.inv(gI)

        # Mass of the quadruped in [kg] (found in urdf)
        self.m = 2.2"""

        # Initialization of the constant part of matrix A
        A = np.zeros((12, 12))
        A[0:3, 0:3] = np.eye(3)
        A[0:3, 6:9] = self.dt * np.eye(3)
        A[3:6, 3:6] = np.eye(3)
        A[6:9, 6:9] = np.eye(3)
        A[9:12, 9:12] = np.eye(3)

        # A_row, A_col and A_data satisfy the relationship A[A_row[k], A_col[k]] = A_data[k]
        A_row = np.array([i for i in range(12)] + [0, 1, 2] + [3, 4, 5])
        A_col = np.array([i for i in range(12)] + [6, 7, 8] + [9, 10, 11])
        A_data = np.array([1 for i in range(12)] + [self.dt, self.dt, self.dt] + [self.dt, self.dt, self.dt])

        # Initialization of the slipping cone constraint matrix C
        # Simplified friction condition with checks only along X and Y axes
        # For instance C = np.array([[1, 0, -nu], [-1, 0, -nu], [0, 1, -nu], [0, -1, -nu]])
        # To check that abs(Fx) <= (nu * Fz) and abs(Fy) <= (nu * Fz)
        nu = 2

        # C_row, C_col and C_data satisfy the relationship C[C_row[k], C_col[k]] = C_data[k]
        C_row = np.array([0, 1, 2, 3] * 2)
        C_col = np.array([0, 0, 1, 1, 2, 2, 2, 2])
        C_data = np.array([1, -1, 1, -1, -nu, -nu, -nu, -nu])

        # Cumulative number of footholds. For instance if two feet touch the ground during 10
        # steps then 4 feets during 6 steps then nb_tot = 2 * 10 + 4 * 6
        nb_tot = np.sum(settings.n_contacts)

        # Matrix M used for the equality constraints (M.X = N)
        # with dimensions (nb_xf * n_x, nb_xf * n_x + nb_tot * n_f)
        # nb_xf * n_x rows for the constraints x(k+1) = A * x(k) + B * f(k) + g. M is basically
        # [ -1  0  0  0  B  0  0  0
        #    A -1  0  0  0  B  0  0
        #    0  A -1  0  0  0  B  0
        #    0  0  A -1  0  0  0  B ] so nb_of_timesteps * nb_of_states lines

        # X vector is [X1 X2 X3 X4 F0 F1 F2 F3] with X1 = A(0) * X0 + B(0) * F(0)
        # A(0) being the current state of the robot
        # So we have nb_xf * n_x columns to store the Xi and nb_tot * n_f columns to store the Fi

        if enable_timer:
            t_test_diff = clock() - t_test
            print("Initialization stuff:", t_test_diff)

            t_test = clock()

        # M_row, _col and _data satisfy the relationship M[M_row[k], M_col[k]] = M_data[k]
        M_row = np.array([], dtype=np.int64)
        M_col = np.array([], dtype=np.int64)
        M_data = np.array([], dtype=np.float64)

        # Fill M with minus identity matrices
        M_row = np.arange(0, nb_xf*self.n_x, 1)
        M_col = np.arange(0, nb_xf*self.n_x, 1)
        M_data = - np.ones((nb_xf*self.n_x,))

        # Fill M with A(k) matrices
        # Looped version:
        """ for i in range(nb_xf-1):
            # Dynamic part of A is related to the dt * R term
            c, s = np.cos(xref[5, i]), np.sin(xref[5, i])
            # R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
            # A[3:6, 9:12] = dt * R
            # Then we just put A in M, here in numpy style
            # M[((i+1)*n_x):((i+2)*n_x), (i*n_x):((i+1)*n_x)] = A

            M_row = np.hstack((M_row, A_row + ((i+1)*n_x)))
            M_col = np.hstack((M_col, A_col + (i*n_x)))
            M_data = np.hstack((M_data, A_data))

            M_row = np.hstack((M_row, np.array([3,  3,  4,  4,  5]) + ((i+1)*n_x)))
            M_col = np.hstack((M_col, np.array([9, 10,  9, 10, 11]) + (i*n_x)))
            M_data = np.hstack((M_data, dt * np.array([c, s, -s, c, 1])))"""

        # Non-looped version:
        M_row = np.hstack((M_row, np.tile(A_row, (nb_xf-1,)) +
                           np.repeat(np.arange(self.n_x, self.n_x*(nb_xf), self.n_x), 18)))
        M_col = np.hstack((M_col, np.tile(A_col, (nb_xf-1,)) +
                           np.repeat(np.arange(0, self.n_x*(nb_xf-1), self.n_x), 18)))
        M_data = np.hstack((M_data, np.tile(A_data, (nb_xf-1,))))

        if enable_timer:
            t_test_diff = clock() - t_test
            print("Fill A in M:", t_test_diff)

        # Matrix L used for the equality constraints (L.X <= K)
        # with dimensions (nb_tot * 5, nb_xf * n_x + nb_tot * n_f)
        # nb_tot * 4 rows for the slipping constraints nu fz > abs(fx) and nu fz > abs(fy) (see C matrix)
        # nb_tot rows for the ground reaction constraints fz > 0
        # L is basically a lot of C and -1 stacked depending on the number of footholds during each timestep.
        # The basic bloc is [ 1  0  -nu
        #                    -1  0  -nu
        #                     0  1  -nu
        #                     0 -1  -nu
        #                     0  0   -1 ] for one foothold

        # L_row, _col and _data satisfy the relationship L[L_row[k], L_col[k]] = L_data[k]
        L_row = np.array([], dtype=np.int64)
        L_col = np.array([], dtype=np.int64)
        L_data = np.array([], dtype=np.float64)

        # Fill M with B(k) matrices
        # and fill L with slipping cone constraints
        # and fill L with ground reaction force constraints (fz > 0)
        nb_tot = 0
        n_tmp = np.sum(settings.n_contacts)
        S_prev = settings.S[0, :].copy()
        for i in range(nb_xf):
            update = (S_prev == 1) & (settings.S[i, :] == 0)  # Detect if one of the feet just left the ground
            if np.any(update):
                for up in range(update.shape[1]):
                    if (update[0, up] == True):  # Considering only feet that just left the ground
                        # Updating position of the foothold for this leg.
                        # no need if only one period in the prediction horizon
                        footholds_m[:, up:(up+1)] = self.footholds_no_lock[0:2, up:(up+1)]

            if False and np.any(update):  # If any foot left the ground (start of swing phase)
                # Get the future position of footholds
                print("UPDATE")
                S_tmp = np.vstack((settings.S[i:, :], settings.S[0:i, :]))
                print(self.xref[6:12, i:(i+1)])
                print(self.xref[0:6, i:(i+1)])
                print(self.x0[6:12, 0:1])
                future_footholds = update_target_footholds_no_lock(
                    self.xref[6:12, i:(i+1)], self.xref[0:6, i:(i+1)], self.x0[6:12, 0:1], settings.t_stance, S_tmp, dt, settings.T_gait, h=x0[2, 0], k=0.03)
                print("Futur footholds local")
                print(future_footholds)

                # As the output of update_target_footholds_no_lock is in local frame then
                # the future position and orientation has to be taken into account to be in optimisation frame
                indexes = (S_tmp != 0).argmax(axis=0)

                for j in range(S_tmp.shape[1]):
                    if (i+indexes[0, j]) < settings.S.shape[0]:
                        c, s = np.cos(self.xref[5, i+indexes[0, j]]), np.sin(self.xref[5, i+indexes[0, j]])
                        R = np.array([[c, -s], [s, c]])
                        future_footholds[:, j] = self.xref[0:2, (i+indexes[0, j])] + np.dot(R, future_footholds[:, j])
                # c, s = np.cos(xref[5, i]), np.sin(xref[5, i])
                # R = np.array([[c, -s], [s, c]])
                #future_footholds = np.tile(xref[0:2, i:(i+1)], (1, 4)) + np.dot(R, future_footholds)

                for up in range(update.shape[1]):
                    if (update[0, up] == True):  # Considering only feet that just touched the ground
                        # Updating position of the foothold for this leg.
                        #  no need if only one period in the prediction horizon
                        footholds_m[:, up:(up+1)] = future_footholds[0:2, up:(up+1)]
            # Saving current state of feet (touching or not) for the next step
            S_prev = (settings.S[i, :]).copy()

            # Number of feet touching the ground during this timestep
            nb_contacts = settings.n_contacts[i, 0]

            # B(k) matrix related to x(k+1) = A * x(k) + B * f(k) + g
            # n_x rows (number of row of x) and n_f * nb_contacts columns (depends on the number of footholds)

            # B_row, _col and _data satisfy the relationship B[B_row[k], B_col[k]] = B_data[k]
            B_row = np.array([], dtype=np.int64)
            B_col = np.array([], dtype=np.int64)
            B_data = np.array([], dtype=np.float64)

            # Position of footholds in the global frame
            pos_contacts = footholds_m[:, (settings.S[i, :] == 1).getA()[0, :]]
            # print(pos_contacts)
            # For each foothold during this timestep
            """for j in range(nb_contacts):
                # Relative position of the foothold compared to the center of mass (here center of the base)
                # contact_foot = np.array([[pos_contacts[0, j]], [pos_contacts[1, j]], [0]]) - xref[0:3, i:(i+1)]

                # Filling the B matrix. In numpy style, just like in the paper:
                # B[9:12, (3*j):(3*j+3)] = dt * np.dot(gI_inv, getSkew(contact_foot))
                # B[6:9, (3*j):(3*j+3)] = dt / m * np.eye(3)

                # Looped version of filling B
                tmp = dt * np.dot(gI_inv, getSkew(contact_foot))
                B_row = np.hstack((B_row, np.array([9, 9, 9, 10, 10, 10, 11, 11, 11])))
                B_col = np.hstack((B_col, np.tile(np.arange(3*j, 3*j+3, 1), (3,))))
                B_data = np.hstack((B_data, tmp.reshape((-1,))))

                B_row = np.hstack((B_row, np.array([6, 7, 8])))
                B_col = np.hstack((B_col, np.arange(3*j, 3*j+3, 1)))
                B_data = np.hstack((B_data, dt / m * np.ones((3,))))

                # Filling the L matrix as explained above. In numpy style:
                # L[(4*nb_tot+4*j):(4*nb_tot+4*(j+1)), (nb_xf*n_x+n_f*(nb_tot+j)):(nb_xf*n_x+n_f*(nb_tot+(j+1)))] = C
                # L[(n_tmp * 4 + nb_tot + j), (nb_xf*n_x+n_f*(nb_tot+j)+(n_f-1))] = -1

                # Looped version of filling L
                L_row = np.hstack((L_row, C_row + (4*nb_tot+4*j)))
                L_col = np.hstack((L_col, C_col + (nb_xf*n_x+n_f*(nb_tot+j))))
                L_data = np.hstack((L_data, C_data))

                L_row = np.hstack((L_row, (n_tmp * 4 + nb_tot + j)))
                L_col = np.hstack((L_col, (nb_xf*n_x+n_f*(nb_tot+j)+(n_f-1))))
                L_data = np.hstack((L_data, -1))"""

            # Non-looped version of filling B
            contact_foot = np.vstack((pos_contacts, np.zeros((1, pos_contacts.shape[1])))) - self.xref[0:3, i:(i+1)]
            # print("Footholds_m: \n", footholds_m)

            """print("### Pos_contacts:")
            print(pos_contacts)
            print(contact_foot)
            print(xref[0:3, i:(i+1)].transpose())"""

            enable_support_lines = False
            if enable_support_lines:
                if (i == 0):
                    for j_c in range(nb_xf):
                        if j_c % 1 == 0:
                            for i_c in range(4):
                                solo.viewer.gui.addCurve("world/support_"+str(j_c)+"_"+str(i_c),
                                                         [[0., 0., 0.], [0., 0., 0.]], [0.0, 0.0, 0.0, 0.0])
                                # solo.viewer.gui.setCurveLineWidth("world/support_"+str(i)+"_"+str(i_c), 0.0)
                                # solo.viewer.gui.setColor("world/support_"+str(i)+"_"+str(i_c), [0.0,0.0,0.0,0.0])
                    solo.viewer.gui.refresh()

                if (i % 1) == 0:

                    num_foot = 0
                    for i_c in range(4):
                        if settings.S[i, i_c] == 1:
                            c, s = np.cos(settings.q_w[5, 0]), np.sin(settings.q_w[5, 0])
                            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 0]])
                            curvePoints_1 = np.dot(R, self.xref[0:3, i:(i+1)]) + \
                                settings.q_w[0:3, 0:1] + np.array([[0], [0], [0.05]])
                            curvePoints_2 = np.dot(R[0:2, 0:2], pos_contacts) + \
                                np.tile(settings.q_w[0:2, 0:1], (1, pos_contacts.shape[1]))
                            solo.viewer.gui.addCurve("world/support_"+str(i)+"_"+str(i_c),
                                                     [[0., 0., 0.], [1., 1., 1]], [i/nb_xf, 0.0, i/nb_xf, 0.5])
                            # solo.viewer.gui.setCurvePoints("world/support_"+str(i)+"_"+str(i_c), [xref[0:3, i].tolist(), [pos_contacts[0,num_foot],pos_contacts[1,num_foot],0.]])
                            solo.viewer.gui.setCurvePoints("world/support_"+str(i)+"_"+str(i_c), [curvePoints_1[:, 0].tolist(), [
                                                           curvePoints_2[0, num_foot], curvePoints_2[1, num_foot], 0.]])
                            solo.viewer.gui.setCurveLineWidth("world/support_"+str(i)+"_"+str(i_c), 8.0)
                            solo.viewer.gui.setColor("world/support_"+str(i)+"_"+str(i_c),
                                                     [i/nb_xf, 0.0, i/nb_xf, 0.5])
                            num_foot += 1
                        else:
                            solo.viewer.gui.setCurveLineWidth("world/support_"+str(i)+"_"+str(i_c), 0.0)
                            solo.viewer.gui.setColor("world/support_"+str(i)+"_"+str(i_c), [0.0, 0.0, 0.0, 0.0])
                    solo.viewer.gui.refresh()

            # tmp = np.reshape(np.array([np.zeros((nb_contacts,)), -contact_foot[2, :], contact_foot[1, :],
            #                  contact_foot[2, :], np.zeros((nb_contacts,)), -contact_foot[0, :],
            #                  -contact_foot[1, :], contact_foot[0, :], np.zeros(nb_contacts,)]), (-1,), order='F')
            tmp = self.dt * np.dot(self.gI_inv, np.array([[np.zeros((nb_contacts,)), -contact_foot[2, :],
                                                           contact_foot[1, :]],
                                                          [contact_foot[2, :], np.zeros(
                                                              (nb_contacts,)), -contact_foot[0, :]],
                                                          [-contact_foot[1, :], contact_foot[0, :],
                                                           np.zeros(nb_contacts,)]]))
            # print("Contact_foot: \n", contact_foot)
            B_row = np.hstack((B_row, np.tile(np.array([9, 9, 9, 10, 10, 10, 11, 11, 11]), (nb_contacts,))))
            B_col = np.hstack((B_col, np.tile(np.array([0, 1, 2]), (3*nb_contacts,)
                                              ) + np.repeat(3 * np.arange(0, nb_contacts, 1), 9)))
            B_data = np.hstack((B_data, np.moveaxis(-tmp, 2, 0).reshape((-1,))))

            B_row = np.hstack((B_row, np.tile(np.array([6, 7, 8]), (nb_contacts,))))
            B_col = np.hstack((B_col, np.arange(0, 3*nb_contacts, 1)))
            B_data = np.hstack((B_data, self.dt / self.m * np.ones((3*nb_contacts,))))

            # Non-looped version of filling L
            L_row = np.hstack((L_row, np.tile(C_row, (nb_contacts,))
                               + np.repeat(4 * nb_tot + np.arange(0, 4 * nb_contacts, 4), len(C_row))))
            L_col = np.hstack((L_col, np.tile(C_col, (nb_contacts,))
                               + np.repeat(nb_xf*self.n_x + self.n_f*nb_tot
                                           + np.arange(0, self.n_f * nb_contacts, self.n_f), len(C_col))))
            L_data = np.hstack((L_data, np.tile(C_data, (nb_contacts,))))

            L_row = np.hstack((L_row, n_tmp * 4 + nb_tot + np.arange(0, nb_contacts, 1)))
            L_col = np.hstack((L_col, nb_xf*self.n_x+self.n_f*nb_tot+(self.n_f-1) +
                               np.arange(0, self.n_f*nb_contacts, self.n_f)))
            L_data = np.hstack((L_data, - np.ones((nb_contacts,))))

            # Filling M with B(k) associated to timestep k. In numpy style:
            # M[(i*n_x):((i+1)*n_x), (nb_xf*n_x+n_f*nb_tot):(nb_xf*n_x+n_f*(nb_tot+nb_contacts))] = B

            M_row = np.hstack((M_row, B_row + (i*self.n_x)))
            M_col = np.hstack((M_col, B_col + (nb_xf*self.n_x+self.n_f*nb_tot)))
            M_data = np.hstack((M_data, B_data))

            # Cumulative number of footholds during the previous step to fill M and L at the correct place
            nb_tot += nb_contacts

        # Matrix N on the other side of the equal sign (M.X = N)
        # n_x * nb_xf rows since there is nb_xf equations A * X + B * F + g
        # Only 1 column
        # N = - g [0 0 0 0 0 0 0 0 1 0 0 0]^T - A*X0 + ( - A*X0ref + X1ref) for the first row
        # N = - g [0 0 0 0 0 0 0 0 1 0 0 0]^T        + ( - A*Xkref + Xk+1ref) for the other rows

        if enable_timer:
            t_test_diff = clock() - t_test
            print("Create L and M:", t_test_diff)

            t_test = clock()

        N = np.zeros((self.n_x*nb_xf, 1))

        # N_row, _col and _data satisfy the relationship N[N_row[k], N_col[k]] = N_data[k]
        N_row = np.array([], dtype=np.int64)
        N_col = np.array([], dtype=np.int64)
        N_data = np.array([], dtype=np.float64)

        # Matrix K on the other side of the inequal sign (L.X <= K)
        # np.sum(settings.n_contacts) * 5 rows to be consistent with L, all coefficients are 0

        # K_row, _col and _data satisfy the relationship K[K_row[k], K_col[k]] = K_data[k]
        K_row = np.array([], dtype=np.int64)
        K_col = np.array([], dtype=np.int64)
        K_data = np.array([], dtype=np.float64)

        # The gravity vector is included in N
        # It has an effect on the 8th coefficient of the state vector for each timestep (linear velocity along Z)
        # In numpy style:
        g = np.zeros((self.n_x, 1))
        g[8, 0] = -9.81 * self.dt
        for i in range(nb_xf):
            N[self.n_x*i:(self.n_x*(i+1)), 0:1] = - g

        # Including - A*X0 in the first row of N
        c, s = np.cos(self.xref[5, 0]), np.sin(self.xref[5, 0])
        R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        A[3:6, 9:12] = self.dt * R
        # Numpy style: N[0:n_x, 0:1] += np.dot(A, - x0)
        N[0:self.n_x, 0:1] += np.dot(A, - self.x0)

        # Here we include both the gravity and -A*X0 at the same time
        tmp = np.dot(A, - self.x0)
        tmp = np.vstack((tmp, np.zeros((self.n_x*(nb_xf-1), 1))))
        tmp[np.arange(0, nb_xf, 1)*self.n_x + 8, 0:1] += 9.81*self.dt

        N_row = np.hstack((N_row, np.arange(0, self.n_x*nb_xf, 1)))
        N_col = np.hstack((N_col, np.zeros((self.n_x*nb_xf,), dtype=np.int64)))
        N_data = np.hstack((N_data, tmp.reshape((-1,))))

        # D is the third term of the sum of N that includes (- A*Xk-1ref + Xkref). For instance
        # [  1   0   0
        #    A   1   0
        #    0   A   1 ] that will be used to do a matrix product with Xref vector
        D = np.zeros((self.n_x*nb_xf, self.n_x*nb_xf))

        # D_row, _col and _data satisfy the relationship D[D_row[k], D_col[k]] = D_data[k]
        D_row = np.array([], dtype=np.int64)
        D_col = np.array([], dtype=np.int64)
        D_data = np.array([], dtype=np.float64)

        # Fill D with identity matrices
        for i in range(nb_xf):
            D[(i*self.n_x):((i+1)*self.n_x), (i*self.n_x):((i+1)*self.n_x)] = np.eye(self.n_x)

        D_row = np.arange(0, nb_xf*self.n_x, 1)
        D_col = np.arange(0, nb_xf*self.n_x, 1)
        D_data = + np.ones((nb_xf*self.n_x,))

        # Fill D with A(k) matrices
        for i in range(nb_xf-1):
            # Dynamic part of A
            c, s = np.cos(self.xref[5, (i+1)]), np.sin(self.xref[5, (i+1)])
            R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
            A[3:6, 9:12] = self.dt * R
            D[((i+1)*self.n_x):((i+2)*self.n_x), (i*self.n_x):((i+1)*self.n_x)] = -A

            D_row = np.hstack((D_row, A_row + ((i+1)*self.n_x)))
            D_col = np.hstack((D_col, A_col + (i*self.n_x)))
            D_data = np.hstack((D_data, A_data))

            D_row = np.hstack((D_row, np.array([3,  3,  4,  4,  5]) + ((i+1)*self.n_x)))
            D_col = np.hstack((D_col, np.array([9, 10,  9, 10, 11]) + (i*self.n_x)))
            D_data = np.hstack((D_data, np.array([c, s, -s, c, 1])))

        if enable_timer:
            t_test_diff = clock() - t_test
            print("Create K and N:", t_test_diff)

            t_test = clock()

        # Convert _row, _col and _data into Compressed Sparse Column matrices (Scipy)
        M_csc = scipy.sparse.csc.csc_matrix((M_data, (M_row, M_col)), shape=(
            nb_xf * self.n_x, nb_xf * self.n_x + nb_tot * self.n_f))
        N_csc = scipy.sparse.csc.csc_matrix((N_data, (N_row, N_col)), shape=(self.n_x*nb_xf, 1))
        L_csc = scipy.sparse.csc.csc_matrix((L_data, (L_row, L_col)), shape=(
            nb_tot * 5, nb_xf * self.n_x + nb_tot * self.n_f))
        K_csc = scipy.sparse.csc.csc_matrix((K_data, (K_row, K_col)), shape=(n_tmp*5, 1))

        if enable_timer:
            t_test_diff = clock() - t_test
            print("Conversion to Csc:", t_test_diff)

        # Include D*xref to N which already contains - g - A*X0 (first row) or just - g (other rows)
        N = N + np.dot(D, (self.xref[:, 1:]).reshape((-1, 1), order='F'))
        N_csc += scipy.sparse.csc.csc_matrix(np.dot(D, (self.xref[:, 1:]
                                                        ).reshape((-1, 1), order='F')), shape=(self.n_x*nb_xf, 1))

        # print("N: ", np.array_equal(N_csc.toarray(), N))
        # Check if matrices are properly created
        """print("M: ", np.array_equal(M_csc.toarray(), M))
        print("N: ", np.array_equal(N_csc.toarray(), N))
        print("L: ", np.array_equal(L_csc.toarray(), L))
        print("K: ", np.array_equal(K_csc.toarray(), K))"""

        # print(gI_inv)

        # Store results
        self.A = M_csc
        self.b = N_csc
        self.G = L_csc
        self.h = K_csc

        # Reshaping h and b because the solver wants them as numpy array with only one dimension
        self.h = (self.h.toarray()).reshape((-1, ))
        self.b = (self.b.toarray()).reshape((-1, ))

        return 0

    def create_weight_matrices(self, settings):
        """Create the weight matrices in the cost x^T.P.x + x^T.q of the QP problem

        """

        # Declaration of the P matrix in "x^T.P.x + x^T.q"
        # P_row, _col and _data satisfy the relationship P[P_row[k], P_col[k]] = P_data[k]
        P_row = np.array([], dtype=np.int64)
        P_col = np.array([], dtype=np.int64)
        P_data = np.array([], dtype=np.float64)

        # Define weights for the x-x_ref components of the optimization vector
        P_row = np.arange(0, self.n_x * (settings.n_contacts).shape[0], 1)
        P_col = np.arange(0, self.n_x * (settings.n_contacts).shape[0], 1)
        P_data = 0.0 * np.ones((self.n_x * (settings.n_contacts).shape[0],))

        # Hand-tuning of parameters if you want to give more weight to specific components
        P_data[0::12] = 1000  # position along x
        P_data[1::12] = 1000  # position along y
        P_data[2::12] = 300  # position along z
        P_data[3::12] = 300  # roll
        P_data[4::12] = 300  # pitch
        P_data[5::12] = 100  # yaw
        P_data[6::12] = 30  # linear velocity along x
        P_data[7::12] = 30  # linear velocity along y
        P_data[8::12] = 300  # linear velocity along z
        P_data[9::12] = 100  # angular velocity along x
        P_data[10::12] = 100  # angular velocity along y
        P_data[11::12] = 30  # angular velocity along z

        # Define weights for the force components of the optimization vector
        P_row = np.hstack((P_row, np.arange(self.n_x * (settings.n_contacts).shape[0], self.G.shape[1], 1)))
        P_col = np.hstack((P_col, np.arange(self.n_x * (settings.n_contacts).shape[0], self.G.shape[1], 1)))
        P_data = np.hstack((P_data, 0.0*np.ones((self.G.shape[1] - self.n_x * (settings.n_contacts).shape[0],))))

        P_data[(self.n_x * (settings.n_contacts).shape[0])::3] = 0.01  # force along x
        P_data[(self.n_x * (settings.n_contacts).shape[0] + 1)::3] = 0.01  # force along y
        P_data[(self.n_x * (settings.n_contacts).shape[0] + 2)::3] = 0.01  # force along z

        # Convert P into a csc matrix for the solver
        self.P = scipy.sparse.csc.csc_matrix((P_data, (P_row, P_col)), shape=(self.G.shape[1], self.G.shape[1]))

        # Declaration of the q matrix in "x^T.P.x + x^T.q"
        self.q = np.hstack((np.zeros(self.n_x * (settings.n_contacts).shape[0],), 0.00 *
                            np.ones((self.G.shape[1]-self.n_x * (settings.n_contacts).shape[0], ))))

        # Weight for the z component of contact forces (fz > 0 so with a positive weight it tries to minimize fz)
        # q[(n_x * (settings.n_contacts).shape[0]+2)::3] = 0.01

        return 0

    def call_solver(self, settings):
        """Create an initial guess and call the solver to solve the QP problem
        """

        # Initial guess for forces (mass evenly supported by all legs in contact)
        f_temp = np.zeros((3*np.sum(settings.n_contacts)))
        # f_temp[2::3] = 2.2 * 9.81 / np.sum(settings.S[0,:])
        tmp = np.array(np.sum(settings.S, axis=1)).ravel().astype(int)
        f_temp[2::3] = (np.repeat(tmp, tmp)-4) / (2 - 4) * (3.0 * 9.81 * 0.5) + \
            (np.repeat(tmp, tmp)-2) / (4 - 2) * (3.0 * 9.81 * 0.25)

        # Initial guess (current state + guess for forces) to warm start the solver
        initx = np.hstack((np.zeros((self.xref.shape[0] * (self.xref.shape[1]-1),)), f_temp))

        # Create the QP solver object
        prob = osqp.OSQP()

        # Stack equality and inequality matrices
        inf_lower_bound = -np.inf * np.ones(len(self.h))
        qp_A = scipy.sparse.vstack([self.G, self.A]).tocsc()
        qp_l = np.hstack([inf_lower_bound, self.b])
        qp_u = np.hstack([self.h, self.b])

        # Setup the solver with the matrices and a warm start
        prob.setup(P=self.P, q=self.q, A=qp_A, l=qp_l, u=qp_u, verbose=False)
        prob.warm_start(x=initx)
        """else:
            qp_A = scipy.sparse.vstack([G, A]).tocsc()
            qp_l = np.hstack([l, b])
            qp_u = np.hstack([h, b])
            prob.update(A=qp_A, l=qp_l, u=qp_u)"""

        # Run the solver to solve the QP problem
        # x = solve_qp(P, q, G, h, A, b, solver='osqp')
        self.x = prob.solve().x

        return 0

    def check_solution(self):
        """Check if the solution returned by the solver respects the constraints
        """

        # Code of checkQPSolution but need to be updated to work in local frame

        return 0

    def retrieve_result(self, settings):
        """Extract relevant information from the output of the QP solver
        """

        # Retrieve the "robot state vector" part of the solution of the QP problem
        self.x_robot = (self.x[0:(self.xref.shape[0]*(self.xref.shape[1]-1))]
                        ).reshape((self.xref.shape[0], self.xref.shape[1]-1), order='F')

        # Retrieve the "contact forces" part of the solution of the QP problem
        self.f_applied = self.x[self.xref.shape[0]*(self.xref.shape[1]-1):(self.xref.shape[0] *
                                                                           (self.xref.shape[1]-1)
                                                                           + settings.n_contacts[0, 0]*3)]

        # As the QP problem is solved for (x_robot - x_ref), we need to add x_ref to the result to get x_robot
        self.x_robot += self.xref[:, 1:]

        # Predicted position and velocity of the robot during the next time step
        self.qu = self.x_robot[0:6, 0:1]
        self.vu = self.x_robot[6:12, 0:1]

        return 0

    def update_viewer(self, viewer, initialisation, settings):
        """Update display for visualization purpose

        Keyword arguments:
        :param viewer: A gepetto viewer object
        :param initialisation: A bool, is it the first iteration of the main loop
        """

        # Display reference trajectory with a red curve (gepetto gui)
        c, s = np.cos(settings.q_w[5, 0]), np.sin(settings.q_w[5, 0])
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 0]])
        curvePoints = np.dot(R, settings.x_ref[0:3, :]) + settings.q_w[0:3, 0:1] + np.array([[0], [0], [0.05]])
        if initialisation:
            viewer.gui.addCurve("world/refTraj", [[0., 0., 0.], [1., 1., 1.]], [1., .0, 0., 0.5])
        viewer.gui.setCurvePoints("world/refTraj", (curvePoints.transpose()).tolist())
        viewer.gui.setCurveLineWidth("world/refTraj", 8.0)

        # Display contact forces with a cyan curves (gepetto gui)
        c_forces = np.zeros((3, 4))
        cpt = 0
        update = np.array(settings.S[0]).ravel()
        for i in range(4):
            if update[i]:
                c_forces[:, i] = self.f_applied[(cpt*3):((cpt+1)*3)]
                cpt += 1

        c, s = np.cos(settings.q_w[5, 0]), np.sin(settings.q_w[5, 0])
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        curvePoints = 0.01 * np.dot(R, c_forces[0:3, :]) + self.footholds_world
        if initialisation:
            for i in range(4):
                viewer.gui.addCurve("world/cForce"+str(i), [[0., 0., 0.], [0., 0., 0.]], [0., 1., 1., 0.5])
        else:
            for i in range(4):
                viewer.gui.setCurvePoints("world/cForce"+str(i), [self.footholds_world[:, i].tolist(),
                                                                  curvePoints[:, i].tolist()])
                viewer.gui.setCurveLineWidth("world/cForce"+str(i), 8.0)

        # Display predicted trajectory with a blue curve (gepetto gui)
        c, s = np.cos(settings.q_w[5, 0]), np.sin(settings.q_w[5, 0])
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 0]])
        curvePoints = np.dot(R, self.x_robot[0:3, :]) + settings.q_w[0:3, 0:1] + np.array([[0.], [0.], [0.05]])
        if initialisation:
            viewer.gui.addCurve("world/optTraj", [[0., 0., 0.], [1., 1., 1.]], [0., .0, 1., 0.5])
        viewer.gui.setCurvePoints("world/optTraj", (curvePoints.transpose()).tolist())
        viewer.gui.setCurveLineWidth("world/optTraj", 8.0)

        # Display current velocity (gepetto gui)
        c, s = np.cos(settings.q_w[5, 0]), np.sin(settings.q_w[5, 0])
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 0]])
        curvePoints = np.dot(R, settings.x_ref[0:3, 0:1]) + settings.q_w[0:3, 0:1] + np.array([[0], [0], [0.05]])
        if initialisation:
            viewer.gui.addCurve("world/velTraj", [[0., 0., 0.], [1., 1., 1.]], [0., 1.0, 0., 0.5])
        viewer.gui.setCurvePoints("world/velTraj", [[curvePoints[0, 0], curvePoints[1, 0], curvePoints[2, 0]],
                                                    [curvePoints[0, 0]+1.5*(c*self.x0[6, 0]-s*self.x0[7, 0]),
                                                     curvePoints[1, 0]+1.5*(c*self.x0[7, 0]+s*self.x0[6, 0]),
                                                     curvePoints[2, 0]+1.5*self.x0[8, 0]]])
        viewer.gui.setCurveLineWidth("world/velTraj", 8.0)

        # Display a trail behind the robot
        self.trail[:, self.k_trail] = settings.q_w[0:3, 0] + np.array([[0.0, 0.0, 0.05]])
        curvePoints = np.dot(R, settings.x_ref[0:3, :]) + settings.q_w[0:3, 0:1] + np.array([[0], [0], [0.05]])
        if initialisation:
            viewer.gui.addCurve("world/trail", [[0., 0., 0.], [1., 1., 1.]], [0., .0, 1., 0.5])
        elif self.k_trail > 1:
            viewer.gui.setCurvePoints("world/trail", (self.trail[:, 0:self.k_trail].transpose()).tolist())
            viewer.gui.setCurveLineWidth("world/trail", 8.0)
        self.k_trail += 1

        return 0

    def plot_graphs(self, settings):

        # Display the predicted trajectory along X, Y and Z for the current iteration
        log_t = self.dt * np.arange(0, self.x_robot.shape[1], 1)

        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(log_t, self.x_robot[0, :], "b", linewidth=2)
        plt.plot(log_t, self.xref[0, 1:], "r", linewidth=2)
        plt.xlabel("Time [s]")
        plt.ylabel("Output trajectory along X [m]")
        plt.legend(["Robot", "Reference"])
        plt.subplot(3, 1, 2)
        plt.plot(log_t, self.x_robot[1, :], "b", linewidth=2)
        plt.plot(log_t, self.xref[1, 1:], "r", linewidth=2)
        plt.xlabel("Time [s]")
        plt.ylabel("Output trajectory along Y [m]")
        plt.legend(["Robot", "Reference"])
        plt.subplot(3, 1, 3)
        plt.plot(log_t, self.x_robot[2, :], "b", linewidth=2)
        plt.plot(log_t, self.xref[2, 1:], "r", linewidth=2)
        plt.xlabel("Time [s]")
        plt.ylabel("Output trajectory along Z [m]")
        plt.legend(["Robot", "Reference"])
        plt.show(block=True)

        # Display the desired contact forces for each foot over the prediction horizon for the current iteration
        f_1 = np.zeros((3, (self.xref.shape[1]-1)))
        f_2 = np.zeros((3, (self.xref.shape[1]-1)))
        f_3 = np.zeros((3, (self.xref.shape[1]-1)))
        f_4 = np.zeros((3, (self.xref.shape[1]-1)))
        fs = [f_1, f_2, f_3, f_4]
        cpt_tot = 0
        for i_f in range((self.xref.shape[1]-1)):
            up = (settings.S[i_f, :] == 1)
            for i_up in range(4):
                if up[0, i_up] == True:
                    (fs[i_up])[:, i_f] = self.x[(self.xref.shape[0]*(self.xref.shape[1]-1) + 3 * cpt_tot):
                                                (self.xref.shape[0]*(self.xref.shape[1]-1) + 3 * cpt_tot + 3)]
                    cpt_tot += 1

        plt.close()
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.title("Front left")
        plt.plot(f_1[0, :], linewidth=2)
        plt.plot(f_1[1, :], linewidth=2)
        plt.plot(f_1[2, :], linewidth=2)
        plt.subplot(2, 2, 2)
        plt.title("Front right")
        plt.plot(f_2[0, :], linewidth=2)
        plt.plot(f_2[1, :], linewidth=2)
        plt.plot(f_2[2, :], linewidth=2)
        plt.subplot(2, 2, 3)
        plt.title("Hindleft")
        plt.plot(f_3[0, :], linewidth=2)
        plt.plot(f_3[1, :], linewidth=2)
        plt.plot(f_3[2, :], linewidth=2)
        plt.subplot(2, 2, 4)
        plt.title("Hind right")
        plt.plot(f_4[0, :], linewidth=2)
        plt.plot(f_4[1, :], linewidth=2)
        plt.plot(f_4[2, :], linewidth=2)
        plt.show(block=True)

        return 0
