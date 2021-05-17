# coding: utf8

import Joystick
#from matplotlib import pyplot as plt
import time
import numpy as np
import math
import pinocchio as pin
import tsid
import utils_mpc
import FootTrajectoryGenerator as ftg
import libquadruped_reactive_walking as la
np.set_printoptions(precision=3, linewidth=300)

class PyPlanner:
    """Planner that outputs current and future locations of footsteps, the reference trajectory of the base and
    the position, velocity, acceleration commands for feet in swing phase based on the reference velocity given by
    the user and the current position/velocity of the base in TSID world
    """

    def __init__(self, dt, dt_tsid, T_gait, T_mpc, k_mpc, on_solo8, h_ref, fsteps_init):

        # Time step of the contact sequence
        self.dt = dt

        # Time step of TSID
        self.dt_tsid = dt_tsid

        # Gait duration
        self.T_gait = T_gait
        self.T_mpc = T_mpc

        # Whether we are working on solo8 or not
        self.on_solo8 = on_solo8

        # Reference height for the trunk
        self.h_ref = h_ref

        # Number of TSID iterations for one iteration of the MPC
        self.k_mpc = k_mpc

        # Feedback gain for the feedback term of the planner
        self.k_feedback = 0.03

        # Position of shoulders in local frame
        self.shoulders = np.array([[0.1946, 0.1946, -0.1946, -0.1946],
                                   [0.14695, -0.14695, 0.14695, -0.14695],
                                   [0.0, 0.0, 0.0, 0.0]])

        # Value of the gravity acceleartion
        self.g = 9.81

        # Value of the maximum allowed deviation due to leg length
        self.L = 0.155

        # Number of time steps in the prediction horizon
        self.n_steps = np.int(self.T_gait/self.dt)

        self.dt_vector = np.linspace(self.dt, self.T_gait, self.n_steps)

        # Reference trajectory matrix of size 12 by (1 + N)  with the current state of
        # the robot in column 0 and the N steps of the prediction horizon in the others
        self.xref = np.zeros((12, 1 + self.n_steps))

        # Gait matrix
        self.gait = np.zeros((20, 5))
        self.fsteps = np.full((self.gait.shape[0], 13), np.nan)
        self.is_static = False  # Flag for static gait
        self.q_static = np.zeros((19, 1))
        self.RPY_static = np.zeros((3, 1))

        # Feet matrix
        self.o_feet_contact = self.shoulders.ravel(order='F').copy()

        # To store the result of the compute_next_footstep function
        self.next_footstep = np.zeros((3, 4))

        # Predefined matrices for compute_footstep function
        self.R = np.zeros((3, 3, self.gait.shape[0]))
        self.R[2, 2, :] = 1.0

        self.flag_rotation_command = int(0)
        self.h_rotation_command = h_ref

        # Create gait matrix
        # self.create_walking_trot()
        self.create_trot()
        """self.create_static()
        self.is_static = True"""
        # self.create_custom()

        self.desired_gait = self.gait.copy()
        self.new_desired_gait = self.gait.copy()

        # Foot trajectory generator
        max_height_feet = 0.05
        t_lock_before_touchdown = 0.00
        self.ftgs = [ftg.Foot_trajectory_generator(
            max_height_feet, t_lock_before_touchdown, self.shoulders[0, i], self.shoulders[1, i]) for i in range(4)]

        # Variables for foot trajectory generator
        self.i_end_gait = -1
        self.t_stance = np.zeros((4, ))  # Total duration of current stance phase for each foot
        self.t_swing = np.zeros((4, ))  # Total duration of current swing phase for each foot
        self.footsteps_target = (self.shoulders[0:2, :]).copy()
        self.goals = fsteps_init.copy()  # Store 3D target position for feet
        self.vgoals = np.zeros((3, 4))  # Store 3D target velocity for feet
        self.agoals = np.zeros((3, 4))  # Store 3D target acceleration for feet
        self.mgoals = np.zeros((6, 4))  # Storage variable for the trajectory generator
        self.mgoals[0, :] = fsteps_init[0, :]
        self.mgoals[3, :] = fsteps_init[1, :]

        # C++ class
        self.Cplanner = la.Planner(dt, dt_tsid, T_gait, T_mpc, k_mpc, on_solo8, h_ref, fsteps_init)

        self.log_debug1 = np.zeros((10001, 3))
        self.log_debug2 = np.zeros((10001, 3))

    def create_static(self):
        """Create the matrices used to handle the gait and initialize them to keep the 4 feet in contact

        self.gait and self.fsteps matrices contains information about the gait
        """

        # Number of timesteps in a half period of gait
        N = np.int(0.5 * self.T_gait/self.dt)

        # Starting status of the gait
        # 4-stance phase, 2-stance phase, 4-stance phase, 2-stance phase
        self.gait = np.zeros((self.fsteps.shape[0], 5))
        self.gait[0:4, 0] = np.array([2*N, 0, 0, 0])
        self.fsteps[0:4, 0] = self.gait[0:4, 0]

        # Set stance and swing phases
        # Coefficient (i, j) is equal to 0.0 if the j-th feet is in swing phase during the i-th phase
        # Coefficient (i, j) is equal to 1.0 if the j-th feet is in stance phase during the i-th phase
        self.gait[0, 1:] = np.ones((4,))

        return 0

    def static_gait(self):
        """
        For a static gait (4 feet on the ground)
        Set stance and swing phases and their duration
        Coefficient (i, j) is equal to 0.0 if the j-th feet is in swing phase during the i-th phase
        Coefficient (i, j) is equal to 1.0 if the j-th feet is in stance phase during the i-th phase
        """

        # Number of timesteps in a half period of gait
        N = np.int(0.5 * self.T_gait/self.dt)

        # Gait matrix
        new_desired_gait = np.zeros((self.fsteps.shape[0], 5))
        new_desired_gait[0:4, 0] = np.array([2*N, 0, 0, 0])
        new_desired_gait[0:4, 1:] = np.ones((4, 4))

        return new_desired_gait

    def create_walking_trot(self):
        """Create the matrices used to handle the gait and initialize them to perform a walking trot

        self.gait and self.fsteps matrices contains information about the walking trot
        """

        # Number of timesteps in a half period of gait
        N = np.int(0.5 * self.T_gait/self.dt)

        # Starting status of the gait
        # 4-stance phase, 2-stance phase, 4-stance phase, 2-stance phase
        self.gait = np.zeros((self.fsteps.shape[0], 5))
        i = 1
        self.gait[(4*i):(4*(i+1)), 0] = np.array([4, N-4, 4, N-4])
        self.fsteps[(4*i):(4*(i+1)), 0] = self.gait[(4*i):(4*(i+1)), 0]

        # Set stance and swing phases
        # Coefficient (i, j) is equal to 0.0 if the j-th feet is in swing phase during the i-th phase
        # Coefficient (i, j) is equal to 1.0 if the j-th feet is in stance phase during the i-th phase
        self.gait[4*i+0, 1:] = np.ones((4,))
        self.gait[4*i+1, [1, 4]] = np.ones((2,))
        self.gait[4*i+2, 1:] = np.ones((4,))
        self.gait[4*i+3, [2, 3]] = np.ones((2,))

        return 0

    def create_trot(self):
        """Create the matrices used to handle the gait and initialize them to perform a trot

        self.gait and self.fsteps matrices contains information about the walking trot
        """

        # Number of timesteps in a half period of gait
        N = np.int(0.5 * self.T_gait/self.dt)

        # Starting status of the gait
        # 4-stance phase, 2-stance phase, 4-stance phase, 2-stance phase
        self.gait = np.zeros((self.fsteps.shape[0], 5))
        i = 1
        self.gait[(2*i):(2*(i+1)), 0] = np.array([N, N])
        self.fsteps[(2*i):(2*(i+1)), 0] = self.gait[(2*i):(2*(i+1)), 0]

        # Set stance and swing phases
        # Coefficient (i, j) is equal to 0.0 if the j-th feet is in swing phase during the i-th phase
        # Coefficient (i, j) is equal to 1.0 if the j-th feet is in stance phase during the i-th phase
        self.gait[2*i+0, [1, 4]] = np.ones((2,))
        self.gait[2*i+1, [2, 3]] = np.ones((2,))

        return 0

    def one_swing_gait(self):
        """
        For a gait with only one leg in swing phase at a given time
        Set stance and swing phases and their duration
        Coefficient (i, j) is equal to 0.0 if the j-th feet is in swing phase during the i-th phase
        Coefficient (i, j) is equal to 1.0 if the j-th feet is in stance phase during the i-th phase
        """

        # Number of timesteps in a half period of gait
        N = np.int(0.5 * self.T_gait/self.dt)

        # Gait matrix
        new_desired_gait = np.zeros((self.fsteps.shape[0], 5))
        new_desired_gait[0:4, 0] = np.array([N/2, N/2, N/2, N/2])
        new_desired_gait[0, 1] = 1
        new_desired_gait[1, 4] = 1
        new_desired_gait[2, 2] = 1
        new_desired_gait[3, 3] = 1

        return new_desired_gait

    def trot_gait(self):
        """
        For a walking trot gait
        Set stance and swing phases and their duration
        Coefficient (i, j) is equal to 0.0 if the j-th feet is in swing phase during the i-th phase
        Coefficient (i, j) is equal to 1.0 if the j-th feet is in stance phase during the i-th phase
        """

        # Number of timesteps in a half period of gait
        N = np.int(0.5 * self.T_gait/self.dt)

        # Gait matrix
        new_desired_gait = np.zeros((self.fsteps.shape[0], 5))
        new_desired_gait[0:2, 0] = np.array([N, N])
        new_desired_gait[0, [1, 4]] = np.ones((2,))
        new_desired_gait[1, [2, 3]] = np.ones((2,))

        return new_desired_gait

    def pacing_gait(self):
        """
        For a pacing gait
        Set stance and swing phases and their duration
        Coefficient (i, j) is equal to 0.0 if the j-th feet is in swing phase during the i-th phase
        Coefficient (i, j) is equal to 1.0 if the j-th feet is in stance phase during the i-th phase
        """

        # Number of timesteps in a half period of gait
        N = np.int(0.5 * self.T_gait/self.dt)

        # Gait matrix
        new_desired_gait = np.zeros((self.fsteps.shape[0], 5))
        new_desired_gait[0:4, 0] = np.array([1, N-1, 1, N-1])
        new_desired_gait[0, 1:] = np.ones((4,))
        new_desired_gait[1, [1, 3]] = np.ones((2,))
        new_desired_gait[2, 1:] = np.ones((4,))
        new_desired_gait[3, [2, 4]] = np.ones((2,))

        return new_desired_gait

    def bounding_gait(self):
        """
        For a bounding gait
        Set stance and swing phases and their duration
        Coefficient (i, j) is equal to 0.0 if the j-th feet is in swing phase during the i-th phase
        Coefficient (i, j) is equal to 1.0 if the j-th feet is in stance phase during the i-th phase
        """

        # Number of timesteps in a half period of gait
        N = np.int(0.5 * self.T_gait/self.dt)

        # Gait matrix
        new_desired_gait = np.zeros((self.fsteps.shape[0], 5))
        new_desired_gait[0:4, 0] = np.array([1, N-1, 1, N-1])
        new_desired_gait[0, 1:] = np.ones((4,))
        new_desired_gait[1, [1, 2]] = np.ones((2,))
        new_desired_gait[2, 1:] = np.ones((4,))
        new_desired_gait[3, [3, 4]] = np.ones((2,))

        return new_desired_gait

    def pronking_gait(self):
        """
        For a pronking gait
        Set stance and swing phases and their duration
        Coefficient (i, j) is equal to 0.0 if the j-th feet is in swing phase during the i-th phase
        Coefficient (i, j) is equal to 1.0 if the j-th feet is in stance phase during the i-th phase
        """

        # Number of timesteps in a half period of gait
        N = np.int(0.5 * self.T_gait/self.dt)

        # Gait matrix
        new_desired_gait = np.zeros((self.fsteps.shape[0], 5))
        new_desired_gait[0:2, 0] = np.array([N-1, N+1])
        new_desired_gait[0, 1:] = np.zeros((4,))
        new_desired_gait[1, 1:] = np.ones((4,))

        return new_desired_gait

    def create_custom(self):
        """Create the matrices used to handle the gait and initialize them to perform a trot

        self.gait and self.fsteps matrices contains information about the walking trot
        """

        # Number of timesteps in a half period of gait
        N = np.int(0.5 * self.T_gait/self.dt)

        # Starting status of the gait
        # 4-stance phase, 2-stance phase, 4-stance phase, 2-stance phase
        self.gait = np.zeros((self.fsteps.shape[0], 5))
        i = 1
        self.gait[(4*i):(4*(i+1)), 0] = np.array([N/2, N/2, N/2, N/2])
        self.fsteps[(4*i):(4*(i+1)), 0] = self.gait[(4*i):(4*(i+1)), 0]

        # Set stance and swing phases
        # Coefficient (i, j) is equal to 0.0 if the j-th feet is in swing phase during the i-th phase
        # Coefficient (i, j) is equal to 1.0 if the j-th feet is in stance phase during the i-th phase
        self.gait[4*i+0, [2, 3, 4]] = np.ones((3,))
        self.gait[4*i+1, [1, 3, 4]] = np.ones((3,))
        self.gait[4*i+2, [1, 2, 4]] = np.ones((3,))
        self.gait[4*i+3, [1, 2, 3]] = np.ones((3,))

        return 0

    def roll_experimental(self, k, k_mpc):
        """Move one step further in the gait cycle

        Decrease by 1 the number of remaining step for the current phase of the gait and increase
        by 1 the number of remaining step for the last phase of the gait (periodic motion)

        Args:
            k (int): number of MPC iterations since the start of the simulation
            k_mpc (int): Number of inv dynamics iterations for one iteration of the MPC
        """

        # Retrieve the new desired gait pattern. Most of the time new_desired_gait will be equal to desired_gait since
        # we want to keep the same gait pattern. However if we want to change then the new gait pattern is temporarily
        # stored inside new_desired_gait before being stored inside desired_gait
        if k % (np.int(self.T_gait/self.dt)*k_mpc) == 0:
            self.desired_gait = self.new_desired_gait.copy()
            self.pt_line = 0
            self.pt_sum = self.desired_gait[0, 0]

        # Index of the first empty line
        index = next((idx for idx, val in np.ndenumerate(self.gait[:, 0]) if val == 0.0), 0.0)[0]

        # Create a new phase if needed or increase the last one by 1 step
        pt = (k / k_mpc) % np.int(self.T_gait/self.dt)
        while pt >= self.pt_sum:
            self.pt_line += 1
            self.pt_sum += self.desired_gait[self.pt_line, 0]
        if np.array_equal(self.desired_gait[self.pt_line, 1:], self.gait[index-1, 1:]):
            self.gait[index-1, 0] += 1.0
        else:
            self.gait[index, 1:] = self.desired_gait[self.pt_line, 1:]
            self.gait[index, 0] = 1.0

        # Decrease the current phase by 1 step and delete it if it has ended
        if self.gait[0, 0] > 1.0:
            self.gait[0, 0] -= 1.0
        else:
            self.gait = np.roll(self.gait, -1, axis=0)
            self.gait[-1, :] = np.zeros((5, ))

            # Store positions of feet that are now in contact
            if (k != 0):
                for i in range(4):
                    if self.gait[0, 1+i] == 1:
                        self.o_feet_contact[(3*i):(3*(i+1))] = self.fsteps[1, (3*i+1):(3*(i+1)+1)]

        return 0

    def compute_footsteps(self, q_cur, v_cur, v_ref, reduced):
        """Compute the desired location of footsteps over the prediction horizon

        Compute a X by 13 matrix containing the remaining number of steps of each phase of the gait (first column)
        and the [x, y, z]^T desired position of each foot for each phase of the gait (12 other columns).
        For feet currently touching the ground the desired position is where they currently are.

        Args:
            q_cur (7x1 array): current position vector of the flying base in world frame (linear and angular stacked)
            v_cur (6x1 array): current velocity vector of the flying base in world frame (linear and angular stacked)
            v_ref (6x1 array): desired velocity vector of the flying base in world frame (linear and angular stacked)
        """

        self.fsteps[:, 0] = self.gait[:, 0]
        self.fsteps[:, 1:] = np.nan

        i = 1

        rpt_gait = np.repeat(self.gait[:, 1:] == 1, 3, axis=1)

        # Set current position of feet for feet in stance phase
        (self.fsteps[0, 1:])[rpt_gait[0, :]] = (self.o_feet_contact)[rpt_gait[0, :]]

        # Get future desired position of footsteps
        self.compute_next_footstep(q_cur, v_cur, v_ref)

        """if reduced:  # Reduce size of support polygon
            self.next_footstep[0:2, :] -= np.array([[0.00, 0.00, -0.00, -0.00],
                                                    [0.04, -0.04, 0.04, -0.04]])"""

        # Cumulative time by adding the terms in the first column (remaining number of timesteps)
        dt_cum = np.cumsum(self.gait[:, 0]) * self.dt

        # Get future yaw angle compared to current position
        angle = v_ref[5, 0] * dt_cum + self.RPY[2, 0]
        c = np.cos(angle)
        s = np.sin(angle)
        self.R[0:2, 0:2, :] = np.array([[c, -s], [s, c]])

        # Displacement following the reference velocity compared to current position
        if v_ref[5, 0] != 0:
            dx = (v_cur[0, 0] * np.sin(v_ref[5, 0] * dt_cum) +
                  v_cur[1, 0] * (np.cos(v_ref[5, 0] * dt_cum) - 1)) / v_ref[5, 0]
            dy = (v_cur[1, 0] * np.sin(v_ref[5, 0] * dt_cum) -
                  v_cur[0, 0] * (np.cos(v_ref[5, 0] * dt_cum) - 1)) / v_ref[5, 0]
        else:
            dx = v_cur[0, 0] * dt_cum
            dy = v_cur[1, 0] * dt_cum

        """print(v_cur.ravel())
        print(v_ref.ravel())
        print(dt_cum.ravel())"""

        # Update the footstep matrix depending on the different phases of the gait (swing & stance)
        while (self.gait[i, 0] != 0):

            # Feet that were in stance phase and are still in stance phase do not move
            A = rpt_gait[i-1, :] & rpt_gait[i, :]
            if np.any(rpt_gait[i-1, :] & rpt_gait[i, :]):
                (self.fsteps[i, 1:])[A] = (self.fsteps[i-1, 1:])[A]

            # Feet that are in swing phase are NaN whether they were in stance phase previously or not
            # Commented as self.fsteps is already filled by np.nan by default
            """if np.any(rpt_gait[i, :] == False):
                (self.fsteps[i, 1:])[rpt_gait[i, :] == False] = np.nan * np.ones((12,))[rpt_gait[i, :] == False]"""

            # Feet that were in swing phase and are now in stance phase need to be updated
            A = np.logical_not(rpt_gait[i-1, :]) & rpt_gait[i, :]
            q_tmp = np.array([[q_cur[0, 0]], [q_cur[1, 0]], [0.0]])  # current position without height
            if np.any(A):
                # self.compute_next_footstep(i, q_cur, v_cur, v_ref)

                # Get desired position of footstep compared to current position
                next_ft = (np.dot(self.R[:, :, i-1], self.next_footstep) + q_tmp +
                           np.array([[dx[i-1]], [dy[i-1]], [0.0]])).ravel(order='F')
                # next_ft = (self.next_footstep + q_tmp + np.array([[dx[i-1]], [dy[i-1]], [0.0]])).ravel(order='F')

                # next_ft = (self.next_footstep).ravel(order='F')

                # Assignement only to feet that have been in swing phase
                (self.fsteps[i, 1:])[A] = next_ft[A]

            i += 1

        # print(self.fsteps[0:2, 2::3])
        return 0

    def compute_next_footstep(self, q_cur, v_cur, v_ref):
        """Compute the desired location of footsteps for a given pair of current/reference velocities

        Compute a 3 by 1 matrix containing the desired location of each feet considering the current velocity of the
        robot and the reference velocity

        Args:
            q_cur (7x1 array): current position vector of the flying base in world frame (linear and angular stacked)
            v_cur (6x1 array): current velocity vector of the flying base in world frame (linear and angular stacked)
            v_ref (6x1 array): desired velocity vector of the flying base in world frame (linear and angular stacked)
        """

        """c, s = math.cos(self.RPY[2, 0]), math.sin(self.RPY[2, 0])
        R = np.array([[c, -s, 0], [s, c, 0], [0.0, 0.0, 0.0]])
        self.b_v_cur = R.transpose() @ v_cur[0:3, 0:1]
        self.b_v_ref = R.transpose() @ v_ref[0:3, 0:1]"""

        # TODO: Automatic detection of t_stance to handle arbitrary gaits
        t_stance = self.T_gait * 0.5
        # self.t_stance[:] = np.sum(self.gait[:, 0:1] * self.gait[:, 1:], axis=0) * self.dt
        """for j in range(4):
            i = 0
            t_stance = 0.0
            while self.gait[i, 1+j] == 1:
                t_stance += self.gait[i, 0]
                i += 1
            if i > 0:
                self.t_stance[j] = t_stance * self.dt"""

        # for j in range(4):
        """if self.gait[i, 1+j] == 1:
            t_stance = 0.0
            while self.gait[i, 1+j] == 1:
                t_stance += self.gait[i, 0]
                i += 1
            i_end = self.gait.shape[0] - 1
            while self.gait[i_end, 0] == 0:
                i_end -= 1
            if i_end == (i - 1):
                i = 0
                while self.gait[i, 1+j] == 1:
                    t_stance += self.gait[i, 0]
                    i += 1
            t_stance *= self.dt"""

        # Order of feet: FL, FR, HL, HR

        # self.next_footstep = np.zeros((3, 4))
        #print("Py computing ", j)

        # Add symmetry term
        self.next_footstep[0:2, :] = t_stance * 0.5 * self.b_v_cur[0:2, 0:1]  # + q_cur[0:2, 0:1]

        #print(self.next_footstep[0:2, j:(j+1)].ravel())

        # Add feedback term
        self.next_footstep[0:2, :] += self.k_feedback * (self.b_v_cur[0:2, 0:1] - self.b_v_ref[0:2, 0:1])

        #print(self.next_footstep[0:2, j:(j+1)].ravel())

        # Add centrifugal term
        #print("b_v_cur: ", self.b_v_cur[0:3, 0].ravel())
        #print("v_ref: ", v_ref[3:6, 0])
        cross = self.cross3(np.array(self.b_v_cur[0:3, 0]), v_ref[3:6, 0])
        # cross = np.cross(v_cur[0:3, 0:1], v_ref[3:6, 0:1], 0, 0).T

        self.next_footstep[0:2, :] += 0.5 * math.sqrt(self.h_ref/self.g) * cross[0:2, 0:1]

        #print(self.next_footstep[0:2, j:(j+1)].ravel())

        # Legs have a limited length so the deviation has to be limited
        (self.next_footstep[0:2, :])[(self.next_footstep[0:2, :]) > self.L] = self.L
        (self.next_footstep[0:2, :])[(self.next_footstep[0:2, :]) < (-self.L)] = -self.L

        # solo8: no degree of freedom along Y for footsteps
        if self.on_solo8:
            # self.next_footstep[1, :] = 0.0
            # TODO: Adapt behaviour for world frame
            pass

        # Add shoulders
        self.next_footstep[0:2, :] += self.shoulders[0:2, :]

        #print(self.next_footstep[0:2, j:(j+1)].ravel())

        return 0

    def run_planner(self, k, k_mpc, q, v, b_vref, h_estim, z_average, joystick=None):

        # Get the reference velocity in world frame (given in base frame)
        self.RPY = utils_mpc.quaternionToRPY(q[3:7, 0])
        c, s = math.cos(self.RPY[2, 0]), math.sin(self.RPY[2, 0])
        vref = b_vref.copy()
        vref[0:2, 0:1] = np.array([[c, -s], [s, c]]) @ b_vref[0:2, 0:1]

        """if k == 0:
            self.new_desired_gait = self.static_gait()
            self.is_static = True
            self.q_static[0:7, 0:1] = q.copy()"""
        """elif k == 2000:
            self.new_desired_gait = self.one_swing_gait()
            self.is_static = False"""

        joystick_code = 0
        if joystick is not None:
            if joystick.northButton:
                joystick_code = 1
                self.new_desired_gait = self.pacing_gait()
                self.is_static = False
                joystick.northButton = False
            elif joystick.eastButton:
                joystick_code = 2
                self.new_desired_gait = self.bounding_gait()
                self.is_static = False
                joystick.eastButton = False
            elif joystick.southButton:
                joystick_code = 3
                self.new_desired_gait = self.trot_gait()
                self.is_static = False
                joystick.southButton = False
            elif joystick.westButton:
                joystick_code = 4
                self.new_desired_gait = self.static_gait()
                self.is_static = True
                self.q_static[0:7, 0:1] = q.copy()
                joystick.westButton = False

        """if (k == 2000):
            self.new_desired_gait = self.static_gait()
        """
        """if (k == 1000):
            self.new_desired_gait = self.pacing_gait()"""

        """if (k == 5000):
            self.new_desired_gait = self.pacing_gait()"""

        # if ((k % k_mpc) == 0):
        # Move one step further in the gait
        # self.roll_experimental(k, k_mpc)

        # Get current and reference velocities in base frame
        # R = np.array([[c, -s, 0], [s, c, 0], [0.0, 0.0, 0.0]])
        # self.b_v_cur = R.transpose() @ v[0:3, 0:1]
        # self.b_v_ref = R.transpose() @ vref[0:3, 0:1]

        # Compute the desired location of footsteps over the prediction horizon
        # self.compute_footsteps(q, v, vref, joystick.reduced)

        # Get the reference trajectory for the MPC
        # self.getRefStates(q, v, vref, z_average)

        # Update desired location of footsteps on the ground
        # self.update_target_footsteps()

        self.Cplanner.run_planner(k, q, v, b_vref, np.double(h_estim), np.double(z_average), joystick_code)

        # Update trajectory generator (3D pos, vel, acc)
        # self.update_trajectory_generator(k, h_estim, q)

        self.xref = self.Cplanner.get_xref()
        self.fsteps = self.Cplanner.get_fsteps()
        self.gait = self.Cplanner.get_gait()
        self.goals = self.Cplanner.get_goals()
        self.vgoals = self.Cplanner.get_vgoals()
        self.agoals = self.Cplanner.get_agoals()

        """if (k % 10) == 0:
            print('- xref:')
            print(self.xref[:, 0:4])
            print('- fsteps:')
            print(self.fsteps[0:6, :])
            print('- gait:')
            print(self.gait[0:6, :])
            print('- goals:')
            print(self.goals)
            from IPython import embed
            embed()"""

        """self.log_debug1[k, :] = self.goals[:, 1]
        self.log_debug2[k, :] = (self.Cplanner.get_goals())[:, 1]
        if (k == 0):
            for m in range(self.log_debug1.shape[1]):
                self.log_debug1[m, :] = self.log_debug1[0, :]
                self.log_debug2[m, :] = self.log_debug1[0, :]"""

        """print("Pytarget")
        print(self.footsteps_target)
        print("Pygoals")
        print(self.goals)
        print("Cgoals")
        print(self.Cplanner.get_goals())"""

        """if (k >= 4000):
            from matplotlib import pyplot as plt

            plt.figure()
            for i in range(3):
                if i == 0:
                    ax0 = plt.subplot(3, 1, 1+i)
                else:
                    plt.subplot(3, 1, 1+i, sharex=ax0)

                h1, = plt.plot(self.log_debug1[:, i], "r", linewidth=3)
                h1, = plt.plot(self.log_debug2[:, i], "b", linewidth=3)

            plt.show(block=True)

            if not np.allclose(self.gait, self.Cplanner.get_gait()):
                print("gait not equal")
            else:
                print("Gait OK")
            if not np.allclose(self.xref, self.Cplanner.get_xref()):
                print("xref not equal")
            else:
                print("xref OK")
            tmp = self.fsteps.copy()
            tmp[np.isnan(tmp)] = 0.0
            if not np.allclose(tmp, self.Cplanner.get_fsteps()):
                print("fsteps not equal")
            else:
                print("fsteps OK")
            if not np.allclose(self.goals, self.Cplanner.get_goals()):
                print("goals not equal")
                print(self.goals)
                print(self.Cplanner.get_goals())
            else:
                print("goals OK")
            if not np.allclose(self.vgoals, self.Cplanner.get_vgoals()):
                print("vgoals not equal")
            else:
                print("vgoals OK")
            if not np.allclose(self.agoals, self.Cplanner.get_agoals()):
                print("agoals not equal")
            else:
                print("agoals OK")
        
            from IPython import embed
            embed()"""

        """print("###")
        print(self.t_stance)
        print(self.t_swing)"""

        return 0

    def getRefStates(self, q, v, vref, z_average):
        """Compute the reference trajectory of the CoM for each time step of the
        predition horizon. The ouput is a matrix of size 12 by (N+1) with N the number
        of time steps in the gait cycle (T_gait/dt) and 12 the position, orientation,
        linear velocity and angular velocity vertically stacked. The first column contains
        the current state while the remaining N columns contains the desired future states.

        Args:
            T_gait (float): duration of one period of gait
            q (7x1 array): current position vector of the flying base in world frame (linear and angular stacked)
            v (6x1 array): current velocity vector of the flying base in world frame (linear and angular stacked)
            vref (6x1 array): desired velocity vector of the flying base in world frame (linear and angular stacked)
        """

        # Update x and y velocities taking into account the rotation of the base over the prediction horizon
        # dt_vector = np.linspace(self.dt, self.T_gait, self.n_steps)
        # yaw = np.linspace(0, self.T_gait-self.dt, self.n_steps) * vref[5, 0]

        # Update yaw and yaw velocity
        # dt_vector = np.linspace(self.dt, T_gait, self.n_steps)
        self.xref[5, 1:] = vref[5, 0] * self.dt_vector
        self.xref[11, 1:] = vref[5, 0]

        # Update x and y velocities taking into account the rotation of the base over the prediction horizon
        # yaw = np.linspace(0, T_gait-self.dt, self.n_steps) * v_ref[5, 0]
        self.xref[6, 1:] = vref[0, 0] * np.cos(self.xref[5, 1:]) - vref[1, 0] * np.sin(self.xref[5, 1:])
        self.xref[7, 1:] = vref[0, 0] * np.sin(self.xref[5, 1:]) + vref[1, 0] * np.cos(self.xref[5, 1:])

        # Update x and y depending on x and y velocities (cumulative sum)
        if vref[5, 0] != 0:
            self.xref[0, 1:] = (vref[0, 0] * np.sin(vref[5, 0] * self.dt_vector[:]) +
                                vref[1, 0] * (np.cos(vref[5, 0] * self.dt_vector[:]) - 1)) / vref[5, 0]
            self.xref[1, 1:] = (vref[1, 0] * np.sin(vref[5, 0] * self.dt_vector[:]) -
                                vref[0, 0] * (np.cos(vref[5, 0] * self.dt_vector[:]) - 1)) / vref[5, 0]
        else:
            self.xref[0, 1:] = vref[0, 0] * self.dt_vector[:]
            self.xref[1, 1:] = vref[1, 0] * self.dt_vector[:]
        # self.xref[0, 1:] = self.dx  # dt_vector * self.xref[6, 1:]
        # self.xref[1, 1:] = self.dy  # dt_vector * self.xref[7, 1:]

        # Start from position of the CoM in local frame
        #self.xref[0, 1:] += q[0, 0]
        #self.xref[1, 1:] += q[1, 0]

        self.xref[5, 1:] += self.RPY[2, 0]

        # Desired height is supposed constant
        self.xref[2, 1:] = self.h_ref + z_average
        self.xref[8, 1:] = 0.0

        # No need to update Z velocity as the reference is always 0
        # No need to update roll and roll velocity as the reference is always 0 for those
        # No need to update pitch and pitch velocity as the reference is always 0 for those

        # Update the current state
        self.xref[0:3, 0:1] = q[0:3, 0:1]
        self.xref[3:6, 0:1] = self.RPY
        self.xref[6:9, 0:1] = v[0:3, 0:1]
        self.xref[9:12, 0:1] = v[3:6, 0:1]

        # Time steps [0, dt, 2*dt, ...]
        # to = np.linspace(0, self.T_gait-self.dt, self.n_steps)

        # Threshold for gamepad command (since even if you do not touch the joystick it's not 0.0)
        step = 0.05

        # Detect if command is above threshold
        if (np.abs(vref[2, 0]) > step) and (self.flag_rotation_command != 1):
            self.flag_rotation_command = 1

        """if True:  # If using joystick
            # State machine
            if (np.abs(vref[2, 0]) > step) and (self.flag_rotation_command == 1):  # Command with joystick
                self.h_rotation_command += vref[2, 0] * self.dt
                self.xref[2, 1:] = self.h_rotation_command
                self.xref[8, 1:] = vref[2, 0]

                self.flag_rotation_command = 1
            elif (np.abs(vref[2, 0]) < step) and (self.flag_rotation_command == 1):  # No command with joystick
                self.xref[8, 1:] = 0.0
                self.xref[9, 1:] = 0.0
                self.xref[10, 1:] = 0.0
                self.flag_rotation_command = 2
            elif self.flag_rotation_command == 0:  # Starting state of state machine
                self.xref[2, 1:] = self.h_ref
                self.xref[8, 1:] = 0.0

            if self.flag_rotation_command != 0:
                # Applying command to pitch and roll components
                self.xref[3, 1:] = self.xref[3, 0].copy() + vref[3, 0].copy() * to
                self.xref[4, 1:] = self.xref[4, 0].copy() + vref[4, 0].copy() * to
                self.xref[9, 1:] = vref[3, 0].copy()
                self.xref[10, 1:] = vref[4, 0].copy()
        else:
            self.xref[2, 1:] = self.h_ref
            self.xref[8, 1:] = 0.0"""

        # Current state vector of the robot
        # self.x0 = self.xref[:, 0:1]

        """self.xref[0:3, 1:2] = self.xref[0:3, 0:1] + self.xref[6:9, 0:1] * self.dt
        self.xref[3:6, 1:2] = self.xref[3:6, 0:1] + self.xref[9:12, 0:1] * self.dt"""

        self.xref[0, 1:] += self.xref[0, 0]
        self.xref[1, 1:] += self.xref[1, 0]

        if self.is_static:
            self.xref[0:3, 1:] = self.q_static[0:3, 0:1]
            self.xref[3:6, 1:] = (utils_mpc.quaternionToRPY(self.q_static[3:7, 0])).reshape((3, 1))

        """if v[0, 0] > 0.02:
            from IPython import embed
            embed()"""

        return 0

    def update_target_footsteps(self):
        """ Update desired location of footsteps using information coming from the footsteps planner
        """

        self.footsteps_target = np.zeros((2, 4))

        for i in range(4):
            index = next((idx for idx, val in np.ndenumerate(
                self.fsteps[:, 3*i+1]) if ((not (val == 0)) and (not np.isnan(val)))), [-1])[0]
            self.footsteps_target[:, i] = self.fsteps[index, (1+i*3):(3+i*3)]

        return 0

    def update_trajectory_generator(self, k, h_estim, q):
        """Update the 3D desired position for feet in swing phase by using a 5-th order polynomial that lead them
           to the desired position on the ground (computed by the footstep planner)

        Args:
            k (int): number of time steps since the start of the simulation
        """

        looping = int(self.T_gait/self.dt_tsid)  # Number of TSID iterations in one gait cycle
        k_loop = (k - 0) % looping  # Current number of iterations since the start of the current gait cycle

        if ((k_loop % self.k_mpc) == 0):

            # Indexes of feet in swing phase
            self.feet = np.where(self.gait[0, 1:] == 0)[0]
            if len(self.feet) == 0:  # If no foot in swing phase
                return 0

            # i_end_gait should point to the latest non-zero line
            if (self.gait[self.i_end_gait, 0] == 0):
                self.i_end_gait -= 1
                while (self.gait[self.i_end_gait, 0] == 0):
                    self.i_end_gait -= 1
            else:
                while (self.gait[self.i_end_gait+1, 0] != 0):
                    self.i_end_gait += 1

            self.t0s = []
            for i in self.feet:  # For each foot in swing phase get remaining duration of the swing phase
                # Index of the line containing the next stance phase
                # index = next((idx for idx, val in np.ndenumerate(gait[:, 1+i]) if (((val == 1)))), [-1])[0]
                # remaining_iterations = np.cumsum(gait[:index, 0])[-1] * self.k_mpc - ((k_loop+1) % self.k_mpc)

                # Compute total duration of current swing phase
                i_iter = 1
                self.t_swing[i] = self.gait[0, 0]
                while self.gait[i_iter, 1+i] == 0:
                    self.t_swing[i] += self.gait[i_iter, 0]
                    i_iter += 1

                remaining_iterations = self.t_swing[i] * self.k_mpc - ((k_loop+1) % self.k_mpc)

                i_iter = self.i_end_gait
                while self.gait[i_iter, 1+i] == 0:
                    self.t_swing[i] += self.gait[i_iter, 0]
                    i_iter -= 1
                self.t_swing[i] *= self.dt_tsid * self.k_mpc

                # TODO: Fix that. We need to assess properly the duration of the swing phase even during the transition
                # between two gaits (need to take into account past information)
                # print(i, " ", self.T_gait, " ", self.t_stance[i])
                self.t_swing[i] = self.T_gait * 0.5 # self.T_gait - self.t_stance[i]  # self.T_gait * 0.5  # - 0.02

                self.t0s.append(
                    np.round(np.max((self.t_swing[i] - remaining_iterations * self.dt_tsid - self.dt_tsid, 0.0)), decimals=3))

            # self.footsteps contains the target (x, y) positions for both feet in swing phase

        else:
            if len(self.feet) == 0:  # If no foot in swing phase
                return 0

            for i in range(len(self.feet)):
                self.t0s[i] = np.round(np.max((self.t0s[i] + self.dt_tsid, 0.0)), decimals=3)

        # Get position, velocity and acceleration commands for feet in swing phase
        for i in range(len(self.feet)):
            i_foot = self.feet[i]

            # Get desired 3D position, velocity and acceleration
            if (self.t0s[i] == 0.000) or (k == 0):
                [x0, dx0, ddx0,  y0, dy0, ddy0,  z0, dz0, ddz0, gx1, gy1] = (self.ftgs[i_foot]).get_next_foot(
                    self.mgoals[0, i_foot], 0.0, 0.0,
                    self.mgoals[3, i_foot], 0.0, 0.0,
                    self.footsteps_target[0, i_foot], self.footsteps_target[1, i_foot], self.t0s[i],  self.t_swing[i_foot], self.dt_tsid)
                self.mgoals[:, i_foot] = np.array([x0, dx0, ddx0, y0, dy0, ddy0])
            else:
                [x0, dx0, ddx0,  y0, dy0, ddy0,  z0, dz0, ddz0, gx1, gy1] = (self.ftgs[i_foot]).get_next_foot(
                    self.mgoals[0, i_foot], self.mgoals[1, i_foot], self.mgoals[2, i_foot],
                    self.mgoals[3, i_foot], self.mgoals[4, i_foot], self.mgoals[5, i_foot],
                    self.footsteps_target[0, i_foot], self.footsteps_target[1, i_foot], self.t0s[i],  self.t_swing[i_foot], self.dt_tsid)
                self.mgoals[:, i_foot] = np.array([x0, dx0, ddx0, y0, dy0, ddy0])

            # Store desired position, velocity and acceleration for later call to this function
            self.goals[:, i_foot] = np.array([x0, y0, z0])  # + np.array([0.0, 0.0, q[2, 0] - self.h_ref])
            self.vgoals[:, i_foot] = np.array([dx0, dy0, dz0])
            self.agoals[:, i_foot] = np.array([ddx0, ddy0, ddz0])
            if k % 10 == 0:
                test = 1

    def cross3(self, left, right):
        """Numpy is inefficient for this"""
        return np.array([[left[1] * right[2] - left[2] * right[1]],
                         [left[2] * right[0] - left[0] * right[2]],
                         [left[0] * right[1] - left[1] * right[0]]])


def test_planner():

    # Set the paths where the urdf and srdf file of the robot are registered
    modelPath = "/opt/openrobots/share/example-robot-data/robots"
    urdf = modelPath + "/solo_description/robots/solo12.urdf"
    srdf = modelPath + "/solo_description/srdf/solo.srdf"
    vector = pin.StdVec_StdString()
    vector.extend(item for item in modelPath)

    # Create the robot wrapper from the urdf model (which has no free flyer) and add a free flyer
    robot = tsid.RobotWrapper(urdf, vector, pin.JointModelFreeFlyer(), False)
    model = robot.model()

    dt = 0.001
    dt_mpc = 0.02
    T_gait = 0.64
    on_solo8 = False
    k = 0
    N = 10000
    k_mpc = 20

    # Logging variables
    ground_pos_target = np.zeros((3, 4, N))
    feet_pos_target = np.zeros((3, 4, N))
    feet_vel_target = np.zeros((3, 4, N))
    feet_acc_target = np.zeros((3, 4, N))
    mpc_traj = np.zeros((12, N))

    # Initialisation
    q = np.zeros((19, 1))
    q[0:7, 0] = np.array([0.0, 0.0, 0.22294615, 0.0, 0.0, 0.0, 1.0])
    v = np.zeros((18, 1))
    b_v = np.zeros((18, 1))

    joystick = Joystick.Joystick(False)
    planner = Planner(dt_mpc, dt, T_gait, k_mpc, on_solo8, q[2, 0])

    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    while k < N:

        t_start = time.time()

        RPY = utils_mpc.quaternionToRPY(q[3:7, 0])
        c = math.cos(RPY[2, 0])
        s = math.sin(RPY[2, 0])

        if (k % k_mpc) == 0:
            joystick.update_v_ref(k, 0)
            joystick.v_ref[0, 0] = 0.3
            joystick.v_ref[1, 0] = 0.0
            joystick.v_ref[5, 0] = 0.0
            # b_v[0:2, 0:1] = joystick.v_ref[0:2, 0:1]
            # b_v[5, 0] = joystick.v_ref[5, 0]

        v[0:2, 0:1] = np.array([[c, -s], [s, c]]) @ joystick.v_ref[0:2, 0:1]
        v[5, 0] = joystick.v_ref[5, 0]

        planner.run_planner(k, k_mpc, q[0:7, 0:1], v[0:6, 0:1], joystick.v_ref)

        # Logging output of foot trajectory generator
        ground_pos_target[0:2, :, k] = planner.footsteps_target.copy()
        feet_pos_target[:, :, k] = planner.goals.copy()
        feet_vel_target[:, :, k] = planner.vgoals.copy()
        feet_acc_target[:, :, k] = planner.agoals.copy()

        # Logging output of MPC trajectory generator
        mpc_traj[:, k] = planner.xref[:, 1]

        # print(RPY.ravel())
        if k % 10 == 0:
            test = 1

        """if k == 0:
            foot_FL = ax.scatter(planner.fsteps[:, 1], planner.fsteps[:, 2], marker="o", s=100)
            foot_FR = ax.scatter(planner.fsteps[:, 4], planner.fsteps[:, 5], marker="o", s=100)
            foot_HL = ax.scatter(planner.fsteps[:, 7], planner.fsteps[:, 8], marker="o", s=100)
            foot_HR = ax.scatter(planner.fsteps[:, 10], planner.fsteps[:, 11], marker="o", s=100)
            pos = ax.scatter([q[0, 0]], [q[1, 0]], marker="x", s=200)
            orientation, = ax.plot(np.array([q[0, 0], q[0, 0]+0.2*c]), np.array([q[1, 0], q[1, 0]+0.2*s]), linestyle="-", linewidth=3)
            velocity, = ax.plot(np.array([q[0, 0], q[0, 0]+v[0,0]]), np.array([q[1, 0], q[1, 0]+v[1,0]]), linestyle="-", linewidth=3)
            trajectory, = ax.plot(planner.xref[0, :], planner.xref[1, :], linestyle="-", linewidth=3)
        else:
            foot_FL.set_offsets(planner.fsteps[:, 1:3])
            foot_FR.set_offsets(planner.fsteps[:, 4:6])
            foot_HL.set_offsets(planner.fsteps[:, 7:9])
            foot_HR.set_offsets(planner.fsteps[:, 10:12])
            pos.set_offsets(np.array([q[0, 0], q[1, 0]]))
            orientation.set_xdata([q[0, 0], q[0, 0]+0.2*c])
            orientation.set_ydata([q[1, 0], q[1, 0]+0.2*s])
            velocity.set_xdata([q[0, 0], q[0, 0]+v[0,0]])
            velocity.set_ydata([q[1, 0], q[1, 0]+v[1,0]])
            trajectory.set_xdata(planner.xref[0, :])
            trajectory.set_ydata(planner.xref[1, :])

        if k % 20 == 0:
            fig.canvas.draw()
            fig.canvas.flush_events()"""

        # Following the mpc reference trajectory perfectly
        q[0:3, 0] = planner.xref[0:3, 1].copy()  # np.array(pin.integrate(model, q, b_v * dt))
        q[3:7, 0] = utils_mpc.EulerToQuaternion(planner.xref[3:6, 1])
        v[0:3, 0] = planner.xref[6:9, 1].copy()
        v[3:6, 0] = planner.xref[9:12, 1].copy()
        k += 1

        while (time.time() - t_start) < 0.002:
            pass

    plt.close("all")

    t_range = np.array([k*dt for k in range(N)])
    plt.figure()
    index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
    lgd_X = ["FL", "FR", "HL", "HR"]
    lgd_Y = ["Pos X", "Pos Y", "Pos Z"]
    plt.figure()
    for i in range(12):
        plt.subplot(3, 4, index[i])
        plt.plot(t_range, feet_pos_target[i % 3, np.int(i/3), :], color='r', linewidth=3, marker='')
        plt.plot(t_range, ground_pos_target[i % 3, np.int(i/3), :], color='b', linewidth=3, marker='')
        plt.legend([lgd_Y[i % 3] + " " + lgd_X[np.int(i/3)]+" Ref", lgd_Y[i % 3] + " " + lgd_X[np.int(i/3)]+" Target"])
    plt.suptitle("Reference positions of feet (world frame)")

    index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
    lgd_X = ["FL", "FR", "HL", "HR"]
    lgd_Y = ["Vel X", "Vel Y", "Vel Z"]
    plt.figure()
    for i in range(12):
        plt.subplot(3, 4, index[i])
        plt.plot(t_range, feet_vel_target[i % 3, np.int(i/3), :], color='r', linewidth=3, marker='')
        plt.legend([lgd_Y[i % 3] + " " + lgd_X[np.int(i/3)]+" Ref"])
    plt.suptitle("Current and reference velocities of feet (world frame)")

    index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
    lgd_X = ["FL", "FR", "HL", "HR"]
    lgd_Y = ["Acc X", "Acc Y", "Acc Z"]
    plt.figure()
    for i in range(12):
        plt.subplot(3, 4, index[i])
        plt.plot(t_range, feet_acc_target[i % 3, np.int(i/3), :], color='r', linewidth=3, marker='')
        plt.legend([lgd_Y[i % 3] + " " + lgd_X[np.int(i/3)]+" Ref"])
    plt.suptitle("Current and reference accelerations of feet (world frame)")

    index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]

    lgd = ["Position X", "Position Y", "Position Z", "Position Roll", "Position Pitch", "Position Yaw", "Linear vel X", "Linear vel Y", "Linear vel Z",
           "Angular vel Roll", "Angular vel Pitch", "Angular vel Yaw"]
    plt.figure()
    for i in range(12):
        plt.subplot(3, 4, index[i])
        plt.plot(t_range[::10], mpc_traj[i, ::10], "r", linewidth=2)
        plt.ylabel(lgd[i])
    plt.suptitle("Predicted trajectories (world frame)")

    plt.show(block=True)


def test_planner_mpc():

    import MPC_Wrapper

    # Set the paths where the urdf and srdf file of the robot are registered
    modelPath = "/opt/openrobots/share/example-robot-data/robots"
    urdf = modelPath + "/solo_description/robots/solo12.urdf"
    srdf = modelPath + "/solo_description/srdf/solo.srdf"
    vector = pin.StdVec_StdString()
    vector.extend(item for item in modelPath)

    # Create the robot wrapper from the urdf model (which has no free flyer) and add a free flyer
    robot = tsid.RobotWrapper(urdf, vector, pin.JointModelFreeFlyer(), False)
    model = robot.model()

    dt = 0.0020
    dt_mpc = 0.02
    T_gait = 0.32
    on_solo8 = False
    k = 0
    N = 20000
    k_mpc = 10

    # Logging variables
    ground_pos_target = np.zeros((3, 4, N))
    feet_pos_target = np.zeros((3, 4, N))
    feet_vel_target = np.zeros((3, 4, N))
    feet_acc_target = np.zeros((3, 4, N))
    planner_traj = np.zeros((12, N))
    mpc_traj = np.zeros((12, N))
    mpc_fc = np.zeros((12, N))

    # Initialisation
    q = np.zeros((19, 1))
    q[0:7, 0] = np.array([0.0, 0.0, 0.22294615, 0.0, 0.0, 0.0, 1.0])
    v = np.zeros((18, 1))
    b_v = np.zeros((18, 1))

    joystick = Joystick.Joystick(False)
    planner = Planner(dt_mpc, dt, T_gait, k_mpc, on_solo8, q[2, 0])
    mpc_wrapper = MPC_Wrapper.MPC_Wrapper(True, dt_mpc, planner.n_steps, k_mpc, planner.T_gait, q, True)
    solo = utils_mpc.init_viewer(True)

    while k < N:

        t_start = time.time()

        RPY = utils_mpc.quaternionToRPY(q[3:7, 0])
        c = math.cos(RPY[2, 0])
        s = math.sin(RPY[2, 0])

        # if (k % k_mpc) == 0:
        joystick.update_v_ref(k, 0)
        joystick.v_ref[0, 0] = np.min((0.4, k * 0.4 / 3000))
        """joystick.v_ref[1, 0] = 0.0
        joystick.v_ref[5, 0] = 0.4"""

        # v[0:2, 0:1] = np.array([[c, -s], [s, c]]) @ joystick.v_ref[0:2, 0:1]
        # v[5, 0] = joystick.v_ref[5, 0]

        """if k == 101:
            mpc_wrapper.last_available_result[3] = 15 * 3.1415/180
            #x_f_mpc[3] = 15 * 3.1415/180
            print("Pass")"""

        # Run planner
        """if k == 100:
            q[3:7, 0] = utils_mpc.EulerToQuaternion([0.2, 0.0, 0.0])"""
        planner.run_planner(k, k_mpc, q[0:7, 0:1], v[0:6, 0:1], joystick.v_ref)

        # Logging output of foot trajectory generator
        ground_pos_target[0:2, :, k] = planner.footsteps_target.copy()
        feet_pos_target[:, :, k] = planner.goals.copy()
        feet_vel_target[:, :, k] = planner.vgoals.copy()
        feet_acc_target[:, :, k] = planner.agoals.copy()

        # Logging output of MPC trajectory generator
        planner_traj[:, k] = planner.xref[:, 1]

        # Send data to MPC parallel process
        if (k % k_mpc) == 0:
            try:
                mpc_wrapper.solve(k, planner)
            except ValueError:
                print("MPC Problem")

        # Check if the MPC has outputted a new result
        x_f_mpc = mpc_wrapper.get_latest_result()

        # print("x_f_mpc: ", x_f_mpc)

        # Logging output of MPC trajectory generator
        mpc_traj[:, k] = x_f_mpc[:12]

        # Contact forces desired by MPC (transformed into world frame)
        mpc_fc[:, k] = x_f_mpc[12:]

        # Following the mpc reference trajectory perfectly
        """q[0:3, 0] = planner.xref[0:3, 1].copy()  # np.array(pin.integrate(model, q, b_v * dt))
        q[3:7, 0] = utils_mpc.EulerToQuaternion(planner.xref[3:6, 1])
        v[0:3, 0] = planner.xref[6:9, 1].copy()
        v[3:6, 0] = planner.xref[9:12, 1].copy()"""

        q[0:3, 0] = x_f_mpc[0:3]
        q[3:7, 0] = utils_mpc.EulerToQuaternion(x_f_mpc[3:6])
        v[0:3, 0] = x_f_mpc[6:9]
        v[3:6, 0] = x_f_mpc[9:12]
        k += 1

        solo.display(q)

        while (time.time() - t_start) < 0.002:
            pass

    mpc_wrapper.stop_parallel_loop()

    # np.savez("mpc_traj_4.npz", mpc_traj_1=mpc_traj)
    # data = np.load("mpc_traj_4.npz")
    # mpc_traj_1 = data['mpc_traj_1']  # Position

    t_range = np.array([k*dt for k in range(N)])

    plt.figure()
    index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
    lgd_X = ["FL", "FR", "HL", "HR"]
    lgd_Y = ["Pos X", "Pos Y", "Pos Z"]
    plt.figure()
    for i in range(12):
        plt.subplot(3, 4, index[i])
        plt.plot(t_range, feet_pos_target[i % 3, np.int(i/3), :], color='r', linewidth=3, marker='')
        plt.plot(t_range, ground_pos_target[i % 3, np.int(i/3), :], color='b', linewidth=3, marker='')
        plt.legend([lgd_Y[i % 3] + " " + lgd_X[np.int(i/3)]+" Ref", lgd_Y[i % 3] + " " + lgd_X[np.int(i/3)]+" Target"])
    plt.suptitle("Reference positions of feet (world frame)")

    index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
    lgd_X = ["FL", "FR", "HL", "HR"]
    lgd_Y = ["Vel X", "Vel Y", "Vel Z"]
    plt.figure()
    for i in range(12):
        plt.subplot(3, 4, index[i])
        plt.plot(t_range, feet_vel_target[i % 3, np.int(i/3), :], color='r', linewidth=3, marker='')
        plt.legend([lgd_Y[i % 3] + " " + lgd_X[np.int(i/3)]+" Ref"])
    plt.suptitle("Current and reference velocities of feet (world frame)")

    index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
    lgd_X = ["FL", "FR", "HL", "HR"]
    lgd_Y = ["Acc X", "Acc Y", "Acc Z"]
    plt.figure()
    for i in range(12):
        plt.subplot(3, 4, index[i])
        plt.plot(t_range, feet_acc_target[i % 3, np.int(i/3), :], color='r', linewidth=3, marker='')
        plt.legend([lgd_Y[i % 3] + " " + lgd_X[np.int(i/3)]+" Ref"])
    plt.suptitle("Current and reference accelerations of feet (world frame)")

    index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]

    lgd = ["Position X", "Position Y", "Position Z", "Position Roll", "Position Pitch", "Position Yaw", "Linear vel X", "Linear vel Y", "Linear vel Z",
           "Angular vel Roll", "Angular vel Pitch", "Angular vel Yaw"]
    plt.figure()
    for i in range(12):
        plt.subplot(3, 4, index[i])
        plt.plot(t_range[::10], planner_traj[i, ::10], "r", linewidth=2)
        plt.plot(t_range[::10], mpc_traj[i, ::10], "b", linewidth=2)
        # plt.plot(t_range[::10], mpc_traj_1[i, ::10], "g", linewidth=2)
        plt.ylabel(lgd[i])
    plt.suptitle("Planner trajectory VS Predicted trajectory (world frame)")

    index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
    lgd1 = ["Ctct force X", "Ctct force Y", "Ctct force Z"]
    lgd2 = ["FL", "FR", "HL", "HR"]
    plt.figure()
    for i in range(12):
        if i == 0:
            ax0 = plt.subplot(3, 4, index[i])
        else:
            plt.subplot(3, 4, index[i], sharex=ax0)

        h1, = plt.plot(t_range, mpc_fc[i, :], "r", linewidth=5)

        plt.xlabel("Time [s]")
        plt.ylabel(lgd1[i % 3]+" "+lgd2[int(i/3)])

        if (i % 3) == 2:
            plt.ylim([-1.0, 15.0])
        else:
            plt.ylim([-1.5, 1.5])

    plt.suptitle("MPC contact forces (world frame)")

    plt.show(block=True)


def test_planner_mpc_tsid():

    import MPC_Wrapper
    from Controller import controller
    import robots_loader

    # Set the paths where the urdf and srdf file of the robot are registered
    modelPath = "/opt/openrobots/share/example-robot-data/robots"
    urdf = modelPath + "/solo_description/robots/solo12.urdf"
    srdf = modelPath + "/solo_description/srdf/solo.srdf"
    vector = pin.StdVec_StdString()
    vector.extend(item for item in modelPath)

    # Create the robot wrapper from the urdf model (which has no free flyer) and add a free flyer
    robot = tsid.RobotWrapper(urdf, vector, pin.JointModelFreeFlyer(), False)
    model = robot.model()

    dt = 0.002
    dt_mpc = 0.02
    T_gait = 0.64
    on_solo8 = False
    k = 0
    N = 110000
    k_mpc = 10

    # Logging variables
    ground_pos_target = np.zeros((3, 4, N))
    feet_pos_target = np.zeros((3, 4, N))
    feet_vel_target = np.zeros((3, 4, N))
    feet_acc_target = np.zeros((3, 4, N))
    planner_traj = np.zeros((12, N))
    mpc_traj = np.zeros((12, N))
    mpc_fc = np.zeros((12, N))
    tsid_traj = np.zeros((12, N))
    tsid_fc = np.zeros((12, N))
    log_xfmpc = np.zeros((24, N))

    # Initialisation
    q = np.zeros((19, 1))
    q[0:7, 0] = np.array([0.0, 0.0, 0.22294615, 0.0, 0.0, 0.0, 1.0])
    q[7:, 0] = np.array([0.0, 0.8, -1.6, 0, 0.8, -1.6, 0, -0.8, 1.6, 0, -0.8, 1.6])
    v = np.zeros((18, 1))
    b_v = np.zeros((18, 1))

    joystick = Joystick.Joystick(False)
    planner = Planner(dt_mpc, dt, T_gait, k_mpc, on_solo8, q[2, 0])
    mpc_wrapper = MPC_Wrapper.MPC_Wrapper(True, dt_mpc, planner.n_steps, k_mpc, planner.T_gait, q, True)
    myController = controller(q[7:, 0], int(N), dt, k_mpc, T_gait, on_solo8)
    solo = utils_mpc.init_viewer(True)

    while k < N:

        t_start = time.time()

        # RPY = utils_mpc.quaternionToRPY(q[3:7, 0])
        # c = math.cos(RPY[2, 0])
        # s = math.sin(RPY[2, 0])

        if (k % k_mpc) == 0:
            joystick.update_v_ref(k, 0)
            joystick.v_ref[0, 0] = 0.0  # np.min((0.3, k * 0.3 / 1000))
            joystick.v_ref[1, 0] = 0.0
            joystick.v_ref[5, 0] = -0.3
            """if k > 2000:
                joystick.v_ref[5, 0] = +0.2
            if k > 4500:
                joystick.v_ref[5, 0] = -0.2"""

        # v[0:2, 0:1] = np.array([[c, -s], [s, c]]) @ joystick.v_ref[0:2, 0:1]
        # v[5, 0] = joystick.v_ref[5, 0]

        # Run planner
        planner.run_planner(k, k_mpc, q[0:7, 0:1], v[0:6, 0:1], joystick.v_ref, v[0:6, 0:1])

        # Logging output of foot trajectory generator
        ground_pos_target[0:2, :, k] = planner.footsteps_target.copy()
        feet_pos_target[:, :, k] = planner.goals.copy()
        feet_vel_target[:, :, k] = planner.vgoals.copy()
        feet_acc_target[:, :, k] = planner.agoals.copy()

        # Logging output of MPC trajectory generator
        planner_traj[:, k] = planner.xref[:, 1]

        """if (k % k_mpc) == 0:
            RPY_tsid = pin.rpy.matrixToRpy(pin.Quaternion(q[3:7, 0:1]).toRotationMatrix())
            oMl = pin.SE3(pin.utils.rotate('z', RPY_tsid[2]), np.array([q[0, 0], q[1, 0], 0.0]))"""

        """if (k % k_mpc) == 0:
            RPY_tsid = pin.rpy.matrixToRpy(pin.Quaternion(q[3:7, 0:1]).toRotationMatrix())
            oMl = pin.SE3(pin.utils.rotate('z', RPY_tsid[2]), np.array([q[0, 0], q[1, 0], 0.0]))

            for i in range(planner.xref.shape[1]):
                planner.xref[0:3, i] = oMl.inverse() * planner.xref[0:3, i:(i+1)]
                planner.xref[3:6, i] = pin.rpy.matrixToRpy(oMl.rotation.transpose() @ pin.Quaternion(np.array([utils_mpc.EulerToQuaternion(planner.xref[3:6, i])]).T).toRotationMatrix())
                planner.xref[6:9, i:(i+1)] = oMl.rotation.transpose() @ planner.xref[6:9, i:(i+1)]
                planner.xref[9:12, i:(i+1)] = oMl.rotation.transpose() @ planner.xref[9:12, i:(i+1)]

            index = next((idx for idx, val in np.ndenumerate(planner.gait[:, 0]) if val == 0.0), 0.0)[0]
            for i in range(index):
                for j in range(4):
                    if not np.isnan(planner.fsteps[i, 1+3*j]):
                        planner.fsteps[i, (1+3*j):(1+3*(j+1))] = (oMl * planner.fsteps[i, (1+3*j):(1+3*(j+1))].transpose()).ravel()
        
        """
        # Send data to MPC parallel process
        if (k % k_mpc) == 0:
            try:
                mpc_wrapper.solve(k, planner)
            except ValueError:
                print("MPC Problem")

        # Check if the MPC has outputted a new result

        x_f_mpc = mpc_wrapper.get_latest_result()
        """b_x_f_mpc = x_f_mpc.copy()
        b_x_f_mpc[0:3] = (oMl * np.array([x_f_mpc[0:3]]).transpose()).ravel()
        b_x_f_mpc[3:6] = pin.rpy.matrixToRpy(oMl.rotation @ pin.Quaternion(np.array([utils_mpc.EulerToQuaternion(x_f_mpc[3:6])]).T).toRotationMatrix())
        b_x_f_mpc[6:9] = (oMl.rotation @ np.array([x_f_mpc[6:9]]).transpose()).ravel()  
        b_x_f_mpc[9:12] = (oMl.rotation @  np.array([x_f_mpc[9:12]]).transpose()).ravel()
        for i in range(4):
            b_x_f_mpc[(12+3*i):(12+3*(i+1))] = (oMl.rotation @ np.array([x_f_mpc[(12+3*i):(12+3*(i+1))]]).transpose()).ravel()
        log_xfmpc[:, k] = b_x_f_mpc.copy()"""

        #print("x_f_mpc: ", x_f_mpc)
        if k == 30:
            deb = 1

        # Logging output of MPC trajectory generator
        mpc_traj[:, k] = x_f_mpc[:12]
        """if k == 2750:
            t_range = np.array([u*dt for u in range(N)])
            index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]

            lgd = ["Position X", "Position Y", "Position Z", "Position Roll", "Position Pitch", "Position Yaw", "Linear vel X", "Linear vel Y", "Linear vel Z",
                    "Angular vel Roll", "Angular vel Pitch", "Angular vel Yaw"]
            plt.figure()
            for i in range(12):
                plt.subplot(3, 4, index[i])
                plt.plot(t_range[::10], planner_traj[i, ::10], "r", linewidth=2)
                plt.plot(t_range[::10], mpc_traj[i, ::10], "b", linewidth=2)
                plt.plot(t_range[::10], tsid_traj[i, ::10], "g", linewidth=2)
                plt.ylabel(lgd[i])
            plt.show(block=True)"""

        """q[0:3, 0] = x_f_mpc[0:3]       
        q[3:7, 0] = utils_mpc.EulerToQuaternion(x_f_mpc[3:6])
        v[0:3, 0] = x_f_mpc[6:9]
        v[3:6, 0] = x_f_mpc[9:12]"""

        # Contact forces desired by MPC (transformed into world frame)
        mpc_fc[:, k] = x_f_mpc[12:]

        # Process Inverse Dynamics - If nothing wrong happened yet in TSID controller
        if (not myController.error):

            if k == 0:
                b_v = v.copy()
            """oMb = pin.SE3(pin.Quaternion(q[3:7, 0:1]), q[0:3, 0:1])
            b_v[0:3, 0:1] = oMb.rotation.transpose() @ v[0:3, 0:1]
            b_v[3:6, 0:1] = oMb.rotation.transpose() @ v[3:6, 0:1]
            b_v[6:, 0] = v[6:, 0]"""
            """print("###")
            print(v[:6, 0].ravel())
            print(b_v[:6, 0].ravel())
            if k > 0:   
                print(myController.vdes[:6, 0].ravel())
            print(x_f_mpc[:12])"""

            # Initial conditions
            """if k == 0:
                myController.qtsid = q.copy()
                myController.vtsid = b_v.copy()
                q_tsid = q.copy()
                v_tsid = b_v.copy()
            else:
                q_tsid = myController.qdes
                v_tsid = myController.vdes"""

            # pin.forwardKinematics(solo.model, solo.data, q, b_v)

            # pin.updateFramePlacements(solo.model, solo.data)

            # print("###")
            # print(utils_mpc.quaternionToRPY(q[3:7, 0:1]).ravel())

            if k == 0:
                log_q = np.zeros((18, N))
                log_v = np.zeros((18, N))
                log_x_mpc = np.zeros((12, N))
                log_f_mpc = np.zeros((12, N))
                log_f_tsid = np.zeros((12, N))
                log_fsteps = np.zeros((8, N))

            log_x_mpc[:, k] = x_f_mpc[:12]
            log_f_mpc[:, k] = x_f_mpc[12:]
            for i in range(4):
                index = next((idx for idx, val in np.ndenumerate(
                    planner.fsteps[:, 3*i+1]) if ((not (val == 0)) and (not np.isnan(val)))), [-1])[0]
                log_fsteps[(2*i):(2*i+2), k] = planner.fsteps[index, (1+i*3):(3+i*3)].copy()

            # print(utils_mpc.quaternionToRPY(q[3:7]).ravel())
            """if k == 0:
                q_tsid = q.copy()"""

            myController.control(q, b_v.copy(), k, solo,
                                 planner, x_f_mpc[:12], x_f_mpc[12:], planner.fsteps,
                                 planner.gait, True, True,
                                 q, b_v)

            for i, j in enumerate(myController.contacts_order):
                log_f_tsid[(3*j):(3*(j+1)), k:(k+1)] = np.reshape(myController.fc[(3*i):(3*(i+1))], (3, 1))
            """tsid_traj[0:3, k] = myController.qdes[0:3]
            tsid_traj[3:6, k:(k+1)] = utils_mpc.quaternionToRPY(myController.qdes[3:7])
            tsid_traj[6:9, k:(k+1)] = oMb.rotation @ myController.vdes[0:3, 0:1]
            tsid_traj[9:12, k:(k+1)] = oMb.rotation @ myController.vdes[3:6, 0:1]"""

            # print(utils_mpc.quaternionToRPY(q[3:7, 0:1]).ravel())

            """# Quantities sent to the control board
            self.result.P = 4.0 * np.ones(12)
            self.result.D = 0.2 * np.ones(12)  # * \
            # np.array([1.0, 0.3, 0.3, 1.0, 0.3, 0.3,
            # 1.0, 0.3, 0.3, 1.0, 0.3, 0.3])
            self.result.q_des[:] = self.myController.qdes[7:]
            self.result.v_des[:] = self.myController.vdes[6:, 0]
            self.result.tau_ff[:] = self.myController.tau_ff"""
        else:
            print("ERROR IN TSID")
            break

        # Following the mpc reference trajectory perfectly
        """q[0:3, 0] = planner.xref[0:3, 1].copy()  # np.array(pin.integrate(model, q, b_v * dt))
        q[3:7, 0] = utils_mpc.EulerToQuaternion(planner.xref[3:6, 1])
        v[0:3, 0] = planner.xref[6:9, 1].copy()
        v[3:6, 0] = planner.xref[9:12, 1].copy()"""
        """q[0:3, 0] = x_f_mpc[0:3]
        q[3:7, 0] = utils_mpc.EulerToQuaternion(x_f_mpc[3:6])
        v[0:3, 0] = x_f_mpc[6:9]
        v[3:6, 0] = x_f_mpc[9:12]"""

        q[:, 0] = myController.qdes.copy()
        oMb = pin.SE3(pin.Quaternion(q[3:7, 0:1]), q[0:3, 0:1])
        v[0:3, 0:1] = oMb.rotation @ myController.vdes[0:3, 0:1].copy()
        v[3:6, 0:1] = oMb.rotation @ myController.vdes[3:6, 0:1].copy()
        v[6:, 0] = myController.vdes[6:, 0].copy()

        # q[:, 0] = myController.qdes.copy()
        b_v[:, 0:1] = myController.vdes.copy()

        q_tsid = q.copy()
        log_q[:3, k] = q_tsid[:3, 0].copy()
        log_q[3:6, k:(k+1)] = utils_mpc.quaternionToRPY(q_tsid[3:7, 0])
        oMb = pin.SE3(pin.Quaternion(q_tsid[3:7, 0:1]), q_tsid[0:3, 0:1])
        log_q[6:, k] = q_tsid[7:, 0].copy()
        log_v[0:3, k:(k+1)] = oMb.rotation @ myController.vdes[0:3, 0:1].copy()
        log_v[3:6, k:(k+1)] = oMb.rotation @ myController.vdes[3:6, 0:1].copy()
        log_v[6:, k:(k+1)] = b_v[6:]

        """for i, j in enumerate(myController.contacts_order):
            tsid_fc[(3*j):(3*(j+1)), k:(k+1)] = np.reshape(myController.fc[(3*i):(3*(i+1))], (3, 1))"""

        k += 1

        # if k == 1000:
        #    oMb = pin.SE3(pin.utils.rotate('z', 3.1415/2), np.array([0.0, 0.0, 0.0]))
        #    q[3:7, 0] = np.array([0, 0, 0.7009093, 0.7132504]) #pin.Quaternion(utils_mpc.EulerToQuaternion(pin.rpy.matrixToRpy(oMb.rotation @ pin.Quaternion(q[3:7, 0:1]).toRotationMatrix())))

        while (time.time() - t_start) < 0.002:
            pass

    mpc_wrapper.stop_parallel_loop()

    # np.savez("mpc_traj_4.npz", mpc_traj_1=mpc_traj)
    # data = np.load("mpc_traj_4.npz")
    # mpc_traj_1 = data['mpc_traj_1']  # Position

    t_range = np.array([k*dt for k in range(N)])

    index6 = [1, 3, 5, 2, 4, 6]
    index12 = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]

    # LOG_Q
    lgd = ["Position X", "Position Y", "Position Z", "Position Roll", "Position Pitch", "Position Yaw"]
    plt.figure()
    for i in range(6):
        if i == 0:
            ax0 = plt.subplot(3, 2, index6[i])
        else:
            plt.subplot(3, 2, index6[i], sharex=ax0)
        plt.plot(t_range, log_x_mpc[i, :], "b", linewidth=2)
        plt.plot(t_range, log_q[i, :], "r", linewidth=2)
        plt.ylabel(lgd[i])

    # LOG_V
    lgd = ["Linear vel X", "Linear vel Y", "Linear vel Z", "Angular vel Roll", "Angular vel Pitch", "Angular vel Yaw"]
    plt.figure()
    for i in range(6):
        if i == 0:
            ax0 = plt.subplot(3, 2, index6[i])
        else:
            plt.subplot(3, 2, index6[i], sharex=ax0)
        plt.plot(t_range, log_x_mpc[i+6, :], "b", linewidth=2)
        plt.plot(t_range, log_v[i, :], "r", linewidth=2)
        plt.ylabel(lgd[i])

    # Forces
    lgd1 = ["Ctct force X", "Ctct force Y", "Ctct force Z"]
    lgd2 = ["FL", "FR", "HL", "HR"]
    plt.figure()
    for i in range(12):
        if i == 0:
            ax0 = plt.subplot(3, 4, index12[i])
        else:
            plt.subplot(3, 4, index12[i], sharex=ax0)

        h1, = plt.plot(t_range, log_f_mpc[i, :], "b", linewidth=5)
        h2, = plt.plot(t_range, log_f_tsid[i, :], "r", linewidth=5)

        plt.xlabel("Time [s]")
        plt.ylabel(lgd1[i % 3]+" "+lgd2[int(i/3)])

        if (i % 3) == 2:
            plt.ylim([-1.0, 15.0])
        else:
            plt.ylim([-1.5, 1.5])

    plt.suptitle("MPC contact forces (world frame)")

    plt.show(block=True)
    ####

    plt.figure()
    index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
    lgd_X = ["FL", "FR", "HL", "HR"]
    lgd_Y = ["Pos X", "Pos Y", "Pos Z"]
    plt.figure()
    for i in range(12):
        plt.subplot(3, 4, index[i])
        plt.plot(t_range, feet_pos_target[i % 3, np.int(i/3), :], color='r', linewidth=3, marker='')
        plt.plot(t_range, ground_pos_target[i % 3, np.int(i/3), :], color='b', linewidth=3, marker='')
        plt.legend([lgd_Y[i % 3] + " " + lgd_X[np.int(i/3)]+" Ref", lgd_Y[i % 3] + " " + lgd_X[np.int(i/3)]+" Target"])
    plt.suptitle("Reference positions of feet (world frame)")

    index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
    lgd_X = ["FL", "FR", "HL", "HR"]
    lgd_Y = ["Vel X", "Vel Y", "Vel Z"]
    plt.figure()
    for i in range(12):
        plt.subplot(3, 4, index[i])
        plt.plot(t_range, feet_vel_target[i % 3, np.int(i/3), :], color='r', linewidth=3, marker='')
        plt.legend([lgd_Y[i % 3] + " " + lgd_X[np.int(i/3)]+" Ref"])
    plt.suptitle("Current and reference velocities of feet (world frame)")

    index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
    lgd_X = ["FL", "FR", "HL", "HR"]
    lgd_Y = ["Acc X", "Acc Y", "Acc Z"]
    plt.figure()
    for i in range(12):
        plt.subplot(3, 4, index[i])
        plt.plot(t_range, feet_acc_target[i % 3, np.int(i/3), :], color='r', linewidth=3, marker='')
        plt.legend([lgd_Y[i % 3] + " " + lgd_X[np.int(i/3)]+" Ref"])
    plt.suptitle("Current and reference accelerations of feet (world frame)")

    index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]

    lgd = ["Position X", "Position Y", "Position Z", "Position Roll", "Position Pitch", "Position Yaw", "Linear vel X", "Linear vel Y", "Linear vel Z",
           "Angular vel Roll", "Angular vel Pitch", "Angular vel Yaw"]
    plt.figure()
    for i in range(12):
        plt.subplot(3, 4, index[i])
        plt.plot(t_range[::10], planner_traj[i, ::10], "r", linewidth=2)
        plt.plot(t_range[::10], mpc_traj[i, ::10], "b", linewidth=2)
        plt.plot(t_range[::10], tsid_traj[i, ::10], "g", linewidth=2)
        plt.ylabel(lgd[i])
    plt.suptitle("Planner trajectory VS Predicted trajectory (world frame)")

    index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
    lgd1 = ["Ctct force X", "Ctct force Y", "Ctct force Z"]
    lgd2 = ["FL", "FR", "HL", "HR"]
    plt.figure()
    for i in range(12):
        if i == 0:
            ax0 = plt.subplot(3, 4, index[i])
        else:
            plt.subplot(3, 4, index[i], sharex=ax0)

        h1, = plt.plot(t_range, mpc_fc[i, :], "b", linewidth=5)
        h2, = plt.plot(t_range, tsid_fc[i, :], "g", linewidth=5)

        plt.xlabel("Time [s]")
        plt.ylabel(lgd1[i % 3]+" "+lgd2[int(i/3)])

        if (i % 3) == 2:
            plt.ylim([-1.0, 15.0])
        else:
            plt.ylim([-1.5, 1.5])

    plt.suptitle("MPC contact forces (world frame)")

    plt.figure()
    for i in range(12):
        if i == 0:
            ax0 = plt.subplot(3, 4, index[i])
        else:
            plt.subplot(3, 4, index[i], sharex=ax0)
        plt.plot(t_range, log_xfmpc[i, :], "b", linewidth=2)
        plt.ylabel(lgd[i])
    plt.suptitle("b_xfmpc")

    plt.figure()
    for i in range(12):
        if i == 0:
            ax0 = plt.subplot(3, 4, index[i])
        else:
            plt.subplot(3, 4, index[i], sharex=ax0)

        h1, = plt.plot(t_range, log_xfmpc[12+i, :], "b", linewidth=5)

        plt.xlabel("Time [s]")
        plt.ylabel(lgd1[i % 3]+" "+lgd2[int(i/3)])

        if (i % 3) == 2:
            plt.ylim([-1.0, 15.0])
        else:
            plt.ylim([-1.5, 1.5])

    plt.suptitle("b_xfmpc forces")

    plt.show(block=True)


if __name__ == "__main__":
    print("START")
    test_planner_mpc()
    print("END")
