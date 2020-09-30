# coding: utf8

import numpy as np
import math


class FootstepPlanner:
    """A footstep planner that handles the choice of future
    footsteps location depending on the current and reference
    velocities of the quadruped.

    Args:
        dt (float): Duration of one time step of the MPC
        n_periods (int): Number of gait periods in one gait cycle
        T_gait (float): Duration of one gait period
        on_solo8 (bool): if we are working on solo8 (True) or solo12 (False)
    """

    def __init__(self, dt, n_periods, T_gait, on_solo8):

        # Feedback gain for the feedback term of the planner
        self.k_feedback = 0.03

        # Position of shoulders in local frame
        self.shoulders = np.array(
            [[0.19, 0.19, -0.19, -0.19], [0.15005, -0.15005, 0.15005, -0.15005]])

        # Time step of the contact sequence
        self.dt = dt

        # Value of the gravity acceleartion
        self.g = 9.81

        # Value of the maximum allowed deviation due to leg length
        self.L = 0.155

        # Whether we are working on solo8 or not
        self.on_solo8 = on_solo8

        # The desired (x,y) position of footsteps
        # If a foot is in swing phase it is where it should land
        # If a foot is in stance phase is is where it should land at the end of its next swing phase
        self.footsteps = self.shoulders.copy()

        # Previous variable but in world frame for visualisation purpose
        self.footsteps_world = self.footsteps.copy()

        # To store the result of the get_prediction function
        self.footsteps_prediction = np.zeros((3, 4))

        # To store the result of the update_footsteps_tsid function
        self.footsteps_tsid = np.zeros((3, 4))
        self.t_remaining_tsid = np.zeros((1, 4))

        # To store the result of the compute_next_footstep function
        self.next_footstep = np.zeros((3, 4))

        # To store the height of contacts
        self.z_contacts = np.zeros((1, 4))

        # Gait duration
        self.n_periods = n_periods
        self.T_gait = T_gait

        # Number of time steps in the prediction horizon
        self.n_steps = np.int(n_periods*self.T_gait/self.dt)

        # Reference trajectory matrix of size 12 by (1 + N)  with the current state of
        # the robot in column 0 and the N steps of the prediction horizon in the others
        self.xref = np.zeros((12, 1 + self.n_steps))

        # Gait matrix
        self.gait = np.zeros((20, 5))
        self.fsteps = np.full((self.gait.shape[0], 13), np.nan)
        self.gait_invdyn = self.gait.copy()
        self.fsteps_invdyn = self.fsteps.copy()

        self.flag_rotation_command = int(0)
        self.h_rotation_command = 0.20

        # Create gait matrix
        self.create_walking_trot()
        # self.create_bounding()
        # self.create_side_walking()
        # self.create_static()

        self.desired_gait = self.gait.copy()
        self.new_desired_gait = self.gait.copy()

        # Predefined matrices for compute_footstep function
        # rpt_gait = np.zeros((self.gait.shape[0], 12), dtype='int64')
        self.R = np.zeros((3, 3, self.gait.shape[0]))
        self.R[2, 2, :] = 1.0
        """
        Predefining sizes slows down the program for some reason...
        self.dt_cum = np.zeros((self.gait.shape[0], ))
        self.c = np.zeros((self.gait.shape[0], ))
        self.s = np.zeros((self.gait.shape[0], ))
        self.angle = np.zeros((self.gait.shape[0], ))
        self.dx = np.zeros((self.gait.shape[0], ))
        self.dy = np.zeros((self.gait.shape[0], ))
        self.next_ft = np.zeros((12,))"""

    def getRefStates(self, k, T_gait, lC, abg, lV, lW, v_ref, h_ref=0.2027682, predefined=True):
        """Compute the reference trajectory of the CoM for each time step of the
        predition horizon. The ouput is a matrix of size 12 by (N+1) with N the number
        of time steps in the gait cycle (T_gait/dt) and 12 the position, orientation,
        linear velocity and angular velocity vertically stacked. The first column contains
        the current state while the remaining N columns contains the desired future states.

        Args:
            k (int): the number of MPC iterations since the start of the simulation
            T_gait (float): duration of one period of gait
            lC (3x0 array): position of the center of mass in local frame
            abg (3x0 array): orientation of the trunk in local frame
            lV (3x0 array): linear velocity of the CoM in local frame
            lW (3x0 array): angular velocity of the trunk in local frame
            v_ref (6x1 array): desired velocity vector of the flying base in local frame (linear and angular stacked)
            h_ref (float): reference height for the trunk
            predefined (bool): if we are using a predefined reference velocity (True) or a joystick (False)
        """

        # Update x and y velocities taking into account the rotation of the base over the prediction horizon
        yaw = np.linspace(0, T_gait-self.dt, self.n_steps) * v_ref[5, 0]
        self.xref[6, 1:] = v_ref[0, 0] * np.cos(yaw) - v_ref[1, 0] * np.sin(yaw)
        self.xref[7, 1:] = v_ref[0, 0] * np.sin(yaw) + v_ref[1, 0] * np.cos(yaw)

        # Update x and y depending on x and y velocities (cumulative sum)
        self.xref[0, 1:] = self.dt * np.cumsum(self.xref[6, 1:])
        self.xref[1, 1:] = self.dt * np.cumsum(self.xref[7, 1:])

        # Start from position of the CoM in local frame
        self.xref[0, 1:] += lC[0, 0]
        self.xref[1, 1:] += lC[1, 0]

        # Desired height is supposed constant so we only need to set it once
        if k == 0:
            self.xref[2, 1:] = h_ref

        # No need to update Z velocity as the reference is always 0
        # No need to update roll and roll velocity as the reference is always 0 for those
        # No need to update pitch and pitch velocity as the reference is always 0 for those
        # Update yaw and yaw velocity
        dt_vector = np.linspace(self.dt, T_gait, self.n_steps)
        self.xref[5, 1:] = v_ref[5, 0] * dt_vector
        self.xref[11, 1:] = v_ref[5, 0]

        # Update the current state
        self.xref[0:3, 0:1] = lC
        self.xref[3:6, 0:1] = abg
        self.xref[6:9, 0:1] = lV
        self.xref[9:12, 0:1] = lW

        # Time steps [0, dt, 2*dt, ...]
        to = np.linspace(0, T_gait-self.dt, self.n_steps)

        # Threshold for gamepad command (since even if you do not touch the joystick it's not 0.0)
        step = 0.05

        # Detect if command is above threshold
        if (np.abs(v_ref[2, 0]) > step) and (self.flag_rotation_command != 1):
            self.flag_rotation_command = 1

        if not predefined:  # If using joystick
            # State machine
            if (np.abs(v_ref[2, 0]) > step) and (self.flag_rotation_command == 1):  # Command with joystick
                self.h_rotation_command += v_ref[2, 0] * self.dt
                self.xref[2, 1:] = self.h_rotation_command
                self.xref[8, 1:] = v_ref[2, 0]

                self.flag_rotation_command = 1
            elif (np.abs(v_ref[2, 0]) < step) and (self.flag_rotation_command == 1):  # No command with joystick
                self.xref[8, 1:] = 0.0
                self.xref[9, 1:] = 0.0
                self.xref[10, 1:] = 0.0
                self.flag_rotation_command = 2
            elif self.flag_rotation_command == 0:  # Starting state of state machine
                self.xref[2, 1:] = h_ref
                self.xref[8, 1:] = 0.0

            if self.flag_rotation_command != 0:
                # Applying command to pitch and roll components
                self.xref[3, 1:] = self.xref[3, 0].copy() + v_ref[3, 0].copy() * to
                self.xref[4, 1:] = self.xref[4, 0].copy() + v_ref[4, 0].copy() * to
                self.xref[9, 1:] = v_ref[3, 0].copy()
                self.xref[10, 1:] = v_ref[4, 0].copy()
        else:
            self.xref[2, 1:] = h_ref
            self.xref[8, 1:] = 0.0

        # Current state vector of the robot
        self.x0 = self.xref[:, 0:1]

        return 0

    def update_viewer(self, viewer, initialisation):
        """Update display for visualization purpose

        Create sphere objects during the first iteration of the main loop then only
        update their location

        Args:
            viewer (gepetto-viewer): A gepetto viewer object
            initialisation (bool): true if it is the first iteration of the main loop
        """

        # Display non-locked target footholds with green spheres (gepetto gui)
        rgbt = [0.0, 1.0, 0.0, 0.5]
        for i in range(4):
            if initialisation:
                viewer.gui.addSphere("world/sphere"+str(i)+"_nolock", .02, rgbt)  # .1 is the radius
            viewer.gui.applyConfiguration(
                "world/sphere"+str(i)+"_nolock", (self.footsteps_world[0, i],
                                                  self.footsteps_world[1, i], 0.0, 1., 0., 0., 0.))

        return 0

    def create_static(self):
        """Create the matrices used to handle the gait and initialize them to keep the 4 feet in contact

        self.gait and self.fsteps matrices contains information about the gait
        """

        # Number of timesteps in a half period of gait
        N = np.int(0.5 * self.T_gait/self.dt)

        # Starting status of the gait
        # 4-stance phase, 2-stance phase, 4-stance phase, 2-stance phase
        self.gait = np.zeros((6, 5))
        self.gait[0:4, 0] = np.array([2*N, 0, 0, 0])
        self.fsteps[0:4, 0] = self.gait[0:4, 0]

        # Set stance and swing phases
        # Coefficient (i, j) is equal to 0.0 if the j-th feet is in swing phase during the i-th phase
        # Coefficient (i, j) is equal to 1.0 if the j-th feet is in stance phase during the i-th phase
        self.gait[0, 1:] = np.ones((4,))

        return 0

    def create_walking_trot(self):
        """Create the matrices used to handle the gait and initialize them to perform a walking trot

        self.gait and self.fsteps matrices contains information about the walking trot
        """

        # Number of timesteps in a half period of gait
        N = np.int(0.5 * self.T_gait/self.dt)

        # Starting status of the gait
        # 4-stance phase, 2-stance phase, 4-stance phase, 2-stance phase
        self.gait = np.zeros((self.fsteps.shape[0], 5))
        for i in range(self.n_periods):
            self.gait[(4*i):(4*(i+1)), 0] = np.array([1, N-1, 1, N-1])
            self.fsteps[(4*i):(4*(i+1)), 0] = self.gait[(4*i):(4*(i+1)), 0]

            # Set stance and swing phases
            # Coefficient (i, j) is equal to 0.0 if the j-th feet is in swing phase during the i-th phase
            # Coefficient (i, j) is equal to 1.0 if the j-th feet is in stance phase during the i-th phase
            self.gait[4*i+0, 1:] = np.ones((4,))
            self.gait[4*i+1, [1, 4]] = np.ones((2,))
            self.gait[4*i+2, 1:] = np.ones((4,))
            self.gait[4*i+3, [2, 3]] = np.ones((2,))

        return 0

    def create_bounding(self):
        """Create the matrices used to handle the gait and initialize them to perform a bounding gait

        self.gait and self.fsteps matrices contains information about the gait
        """

        # Number of timesteps in a half period of gait
        N = np.int(0.5 * self.T_gait/self.dt)

        # Starting status of the gait
        # 4-stance phase, 2-stance phase, 4-stance phase, 2-stance phase
        self.gait = np.zeros((self.fsteps.shape[0], 5))
        self.gait[0:4, 0] = np.array([1, N-1, 1, N-1])
        self.fsteps[0:4, 0] = self.gait[0:4, 0]

        # Set stance and swing phases
        # Coefficient (i, j) is equal to 0.0 if the j-th feet is in swing phase during the i-th phase
        # Coefficient (i, j) is equal to 1.0 if the j-th feet is in stance phase during the i-th phase
        self.gait[0, 1:] = np.ones((4,))
        self.gait[1, [1, 2]] = np.ones((2,))
        self.gait[2, 1:] = np.ones((4,))
        self.gait[3, [3, 4]] = np.ones((2,))

        return 0

    def create_side_walking(self):
        """Create the matrices used to handle the gait and initialize them to perform a walking gait
        with feet on the same side in contact

        self.gait and self.fsteps matrices contains information about the gait
        """

        # Number of timesteps in a half period of gait
        N = np.int(0.5 * self.T_gait/self.dt)

        # Starting status of the gait
        # 4-stance phase, 2-stance phase, 4-stance phase, 2-stance phase
        self.gait = np.zeros((self.fsteps.shape[0], 5))
        self.gait[0:4, 0] = np.array([1, N-1, 1, N-1])
        self.fsteps[0:4, 0] = self.gait[0:4, 0]

        # Set stance and swing phases
        # Coefficient (i, j) is equal to 0.0 if the j-th feet is in swing phase during the i-th phase
        # Coefficient (i, j) is equal to 1.0 if the j-th feet is in stance phase during the i-th phase
        self.gait[0, 1:] = np.ones((4,))
        self.gait[1, [1, 3]] = np.ones((2,))
        self.gait[2, 1:] = np.ones((4,))
        self.gait[3, [2, 4]] = np.ones((2,))

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

    def walking_trot_gait(self):
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
        new_desired_gait[0:4, 0] = np.array([1, N-1, 1, N-1])
        new_desired_gait[0, 1:] = np.ones((4,))
        new_desired_gait[1, [1, 4]] = np.ones((2,))
        new_desired_gait[2, 1:] = np.ones((4,))
        new_desired_gait[3, [2, 3]] = np.ones((2,))

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

    def compute_footsteps(self, l_feet, v_cur, v_ref, h, reduced):
        """Compute the desired location of footsteps over the prediction horizon

        Compute a X by 13 matrix containing the remaining number of steps of each phase of the gait (first column)
        and the [x, y, z]^T desired position of each foot for each phase of the gait (12 other columns).
        For feet currently touching the ground the desired position is where they currently are.

        Args:
            l_feet (3x4 array): current position of feet in local frame
            v_cur (6x1 array): current velocity vector of the flying base in local frame (linear and angular stacked)
            v_ref (6x1 array): desired velocity vector of the flying base in local frame (linear and angular stacked)
            h (float): desired height for the trunk of the robot
            reduced (bool): if the size of the support polygon is reduced or not
        """

        self.fsteps[:, 0] = self.gait[:, 0]
        self.fsteps[:, 1:] = np.nan

        i = 1

        rpt_gait = np.repeat(self.gait[:, 1:] == 1, 3, axis=1)

        # Set current position of feet for feet in stance phase
        (self.fsteps[0, 1:])[rpt_gait[0, :]] = (l_feet.ravel(order='F'))[rpt_gait[0, :]]

        # Get future desired position of footsteps
        self.compute_next_footstep(v_cur, v_ref, h)

        if reduced:  # Reduce size of support polygon
            self.next_footstep[0:2, :] -= np.array([[0.12, 0.12, -0.12, -0.12],
                                                    [0.10, -0.10, 0.10, -0.10]])

        self.next_footstep[2, :] = self.z_contacts[0, :].copy()

        # Cumulative time by adding the terms in the first column (remaining number of timesteps)
        dt_cum = np.cumsum(self.gait[:, 0]) * self.dt

        # Get future yaw angle compared to current position
        angle = v_ref[5, 0] * dt_cum
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
            if np.any(A):

                # Get desired position of footstep compared to current position
                next_ft = (np.dot(self.R[:, :, i-1], self.next_footstep) +
                           np.array([[dx[i-1]], [dy[i-1]], [0.0]])).ravel(order='F')
                # next_ft = (self.next_footstep).ravel(order='F')

                # Assignement only to feet that have been in swing phase
                (self.fsteps[i, 1:])[A] = next_ft[A]

            i += 1

        # print(self.fsteps[0:2, 2::3])
        return 0

    def compute_next_footstep(self, v_cur, v_ref, h):
        """Compute the desired location of footsteps for a given pair of current/reference velocities

        Compute a 3 by 4 matrix containing the desired location of each feet considering the current velocity of the
        robot and the reference velocity

        Args:
            v_cur (6x1 array): current velocity vector of the flying base in local frame (linear and angular stacked)
            v_ref (6x1 array): desired velocity vector of the flying base in local frame (linear and angular stacked)
            h (float): desired height for the trunk of the robot
        """

        # TODO: Automatic detection of t_stance to handle arbitrary gaits
        t_stance = self.T_gait * 0.5

        # Order of feet: FL, FR, HL, HR

        # self.next_footstep = np.zeros((3, 4))

        # Add symmetry term
        self.next_footstep[0:2, :] = t_stance * 0.5 * v_cur[0:2, 0:1]

        # Add feedback term
        self.next_footstep[0:2, :] += self.k_feedback * (v_cur[0:2, 0:1] - v_ref[0:2, 0:1])

        # Add centrifugal term
        cross = cross3(np.array(v_cur[0:3, 0]), v_ref[3:6, 0])
        # cross = np.cross(v_cur[0:3, 0:1], v_ref[3:6, 0:1], 0, 0).T
        self.next_footstep[0:2, :] += 0.5 * math.sqrt(h/self.g) * cross[0:2, 0:1]

        # Legs have a limited length so the deviation has to be limited
        (self.next_footstep[0:2, :])[(self.next_footstep[0:2, :]) > self.L] = self.L
        (self.next_footstep[0:2, :])[(self.next_footstep[0:2, :]) < (-self.L)] = -self.L

        # solo8: no degree of freedom along Y for footsteps
        if self.on_solo8:
            self.next_footstep[1, :] = 0.0

        # Add shoulders
        self.next_footstep[0:2, :] += self.shoulders

        return 0

    def roll(self):
        """Move one step further in the gait cycle

        Decrease by 1 the number of remaining step for the current phase of the gait and increase
        by 1 the number of remaining step for the last phase of the gait (periodic motion)
        """

        # Index of the first empty line
        index = next((idx for idx, val in np.ndenumerate(self.gait[:, 0]) if val == 0.0), 0.0)[0]

        # Create a new phase if needed or increase the last one by 1 step
        if np.array_equal(self.gait[0, 1:], self.gait[index-1, 1:]):
            self.gait[index-1, 0] += 1.0
        else:
            self.gait[index, 1:] = self.gait[0, 1:]
            self.gait[index, 0] = 1.0

        # Decrease the current phase by 1 step and delete it if it has ended
        if self.gait[0, 0] > 1.0:
            self.gait[0, 0] -= 1.0
        else:
            self.gait = np.roll(self.gait, -1, axis=0)
            self.gait[-1, :] = np.zeros((5, ))

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

        """print(k, ": ")
        print("pt_line: ", self.pt_line)
        print("pt_sum: ", self.pt_sum)
        print("pt: ", pt)
        print(self.gait[0:6, :])
        print("###")"""

        return 0

    def update_fsteps(self, k, k_mpc, l_feet, v_cur, v_ref, h, oMl, ftps_Ids, reduced):
        """Update the gait cycle and compute the desired location of footsteps for a given pair of current/reference velocities

        Args:
            k (int): number of MPC iterations since the start of the simulation
            k_mpc (int): Number of inv dynamics iterations for one iteration of the MPC
            l_feet (3x4 array): current position of feet in local frame
            v_cur (6x1 array): current velocity vector of the flying base in local frame (linear and angular stacked)
            v_ref (6x1 array): desired velocity vector of the flying base in local frame (linear and angular stacked)
            h (float): desired height for the trunk of the robot
            oMl (SE3): SE3 object that contains the translation and rotation to go from local frame to world frame
            ftps_Ids (4xX array): IDs of PyBullet objects to visualize desired footsteps location with spheres
        """

        """self.gait = self.desired_gait.copy()
        print("== DEBUG ==")
        print(self.gait)
        for i in range(640):
            if i == 240:
                # Number of timesteps in a half period of gait
                N = np.int(0.5 * self.T_gait/self.dt)

                # Starting status of the gait
                # 4-stance phase, 2-stance phase, 4-stance phase, 2-stance phase
                self.new_desired_gait = np.zeros((self.fsteps.shape[0], 5))
                self.new_desired_gait[0:4, 0] = np.array([1, N-1, 1, N-1])

                # Set stance and swing phases
                # Coefficient (i, j) is equal to 0.0 if the j-th feet is in swing phase during the i-th phase
                # Coefficient (i, j) is equal to 1.0 if the j-th feet is in stance phase during the i-th phase
                self.new_desired_gait[0, 1:] = np.ones((4,))
                self.new_desired_gait[1, [1, 3]] = np.ones((2,))
                self.new_desired_gait[2, 1:] = np.ones((4,))
                self.new_desired_gait[3, [2, 4]] = np.ones((2,))

                N = np.int(0.5 * self.T_gait/self.dt)
                self.new_desired_gait = np.zeros((self.fsteps.shape[0], 5))
                self.new_desired_gait[0:8, 0] = np.array([2, 2, 2, 2, 2, 2, 2, 2])
                self.new_desired_gait[0, 1:] = np.array([1, 0, 0, 0])
                self.new_desired_gait[1, 1:] = np.array([1, 0, 0, 1])
                self.new_desired_gait[2, 1:] = np.array([0, 0, 0, 1])
                self.new_desired_gait[3, 1:] = np.array([0, 0, 0, 0])
                self.new_desired_gait[4, 1:] = np.array([0, 0, 1, 0])
                self.new_desired_gait[5, 1:] = np.array([0, 1, 1, 0])
                self.new_desired_gait[6, 1:] = np.array([0, 1, 0, 0])
                self.new_desired_gait[7, 1:] = np.array([0, 0, 0, 0])

            if (i != -1) and ((i % k_mpc) == 0):
                self.roll_experimental(i, k_mpc)"""

        """if (k == 1500) or (k == 3500) or (k == 5500) or (k == 7500):
            self.new_desired_gait = self.static_gait()
        elif (k == 2000):
            self.new_desired_gait = self.pacing_gait()
        elif (k == 4000):
            self.new_desired_gait = self.bounding_gait()
        elif (k == 6000):
            self.new_desired_gait = self.pronking_gait()
        elif (k == 8000):
            self.new_desired_gait = self.walking_trot_gait()"""

        if (k != -1) and ((k % k_mpc) == 0):
            # Move one step further in the gait
            self.roll_experimental(k, k_mpc)

        for i in range(4):
            if self.gait[0, i+1]:
                self.z_contacts[0, i] = l_feet[2, i]

        # Compute the desired location of footsteps over the prediction horizon
        self.compute_footsteps(l_feet, v_cur, v_ref, h, reduced)

        # Display spheres for footsteps visualization
        """
        import pybullet as pyb
        i = 0
        up = np.isnan(self.gait[:, 1:])
        while (self.gait[i, 0] != 0 and i < 2):
            for j in range(4):
                if not up[i, j]:
                    pos_tmp = np.array(oMl * np.array([self.fsteps[i, (1+j*3):(4+j*3)]]).transpose())
                    pyb.resetBasePositionAndOrientation(ftps_Ids[j, i],
                                                        posObj=pos_tmp,
                                                        ornObj=np.array([0.0, 0.0, 0.0, 1.0]))
            i += 1
        """
        """for j in range(4):
            pos_tmp = np.array(oMl * np.array([l_feet[:, j]]).transpose())
            pyb.resetBasePositionAndOrientation(ftps_Ids[j, 0],
                                                posObj=pos_tmp,
                                                ornObj=np.array([0.0, 0.0, 0.0, 1.0]))"""

        return 0


def cross3(left, right):
    """Numpy is inefficient for this"""
    return np.array([[left[1] * right[2] - left[2] * right[1]],
                     [left[2] * right[0] - left[0] * right[2]],
                     [left[0] * right[1] - left[1] * right[0]]])
