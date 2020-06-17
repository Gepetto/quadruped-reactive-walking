# coding: utf8


########################################################################
#                                                                      #
#          Control law : tau = P(q*-q^) + D(v*-v^) + tau_ff            #
#                                                                      #
########################################################################

from matplotlib import pyplot as plt
import pinocchio as pin
import numpy as np
import numpy.matlib as matlib
import tsid
import FootTrajectoryGenerator as ftg
import pybullet as pyb
import utils
import time
pin.switchToNumpyMatrix()


########################################################################
#            Class for a PD with feed-forward Controller               #
########################################################################

class controller:
    """ Inverse Dynamics controller that take into account the dynamics of the quadruped to generate
        actuator torques to apply on the ground the contact forces computed by the MPC (for feet in stance
        phase) and to perform the desired footsteps (for feet in swing phase)

        Args:
            N_similation (int): maximum number of Inverse Dynamics iterations for the simulation
            k_mpc (int): number of tsid iterations for one iteration of the mpc
            n_periods (int): number of gait periods in the prediction horizon
            T_gait (float): duration of one gait period
    """

    def __init__(self, N_simulation, k_mpc, n_periods, T_gait):

        self.q_ref = np.array([[0.0, 0.0, 0.2027682, 0.0, 0.0, 0.0, 1.0,
                                0.0, 0.8, -1.6, 0, 0.8, -1.6,
                                0, -0.8, 1.6, 0, -0.8, 1.6]]).transpose()

        self.qtsid = self.q_ref.copy()
        self.vtsid = np.zeros((18, 1))
        self.ades = np.zeros((18, 1))

        self.error = False
        self.verbose = True

        # List with the names of all feet frames
        self.foot_frames = ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT']

        # Constraining the contacts
        mu = 0.9 				# friction coefficient
        fMin = 1.0				# minimum normal force
        fMax = 25.0  			# maximum normal force
        contactNormal = np.matrix([0., 0., 1.]).T  # direction of the normal to the contact surface

        # Coefficients of the posture task
        kp_posture = 10.0		# proportionnal gain of the posture task
        w_posture = 1.0         # weight of the posture task

        # Coefficients of the contact tasks
        kp_contact = 100.0         # proportionnal gain for the contacts
        self.w_forceRef = 500.0  # weight of the forces regularization
        self.w_reg_f = 50.0

        # Coefficients of the foot tracking task
        kp_foot = 5000.0               # proportionnal gain for the tracking task
        self.w_foot = 500.0       # weight of the tracking task

        # Coefficients of the trunk task
        kp_trunk = 100
        w_trunk = 10

        # Arrays to store logs
        k_max_loop = N_simulation
        self.f_pos = np.zeros((4, k_max_loop, 3))
        self.f_vel = np.zeros((4, k_max_loop, 3))
        self.f_acc = np.zeros((4, k_max_loop, 3))
        self.f_pos_ref = np.zeros((4, k_max_loop, 3))
        self.f_vel_ref = np.zeros((4, k_max_loop, 3))
        self.f_acc_ref = np.zeros((4, k_max_loop, 3))
        self.b_pos = np.zeros((k_max_loop, 6))
        self.b_vel = np.zeros((k_max_loop, 6))
        self.com_pos = np.zeros((k_max_loop, 3))
        self.com_pos_ref = np.zeros((k_max_loop, 3))
        self.c_forces = np.zeros((4, k_max_loop, 3))
        self.h_ref_feet = np.zeros((k_max_loop, ))
        self.goals = np.zeros((3, 4))
        self.vgoals = np.zeros((3, 4))
        self.agoals = np.zeros((3, 4))
        self.mgoals = np.zeros((6, 4))

        # Position of the shoulders in local frame
        self.shoulders = np.array([[0.19, 0.19, -0.19, -0.19], [0.15005, -0.15005, 0.15005, -0.15005]])
        self.footsteps = self.shoulders.copy()
        self.memory_contacts = self.shoulders.copy()

        # Foot trajectory generator
        max_height_feet = 0.05
        t_lock_before_touchdown = 0.05
        self.ftgs = [ftg.Foot_trajectory_generator(max_height_feet, t_lock_before_touchdown) for i in range(4)]

        # Which pair of feet is active (0 for [1, 2] and 1 for [0, 3])
        self.pair = -1

        # Number of TSID steps for 1 step of the MPC
        self.k_mpc = k_mpc

        # For update_feet_tasks function
        self.dt = 0.001  #  [s], time step
        self.t1 = T_gait * 0.5 - 0.02  # [s], duration of swing phase

        # Rotation matrix
        self.R = np.eye(3)

        # Feedforward torques
        self.tau_ff = np.zeros((12, 1))

        # Torques sent to the robot
        self.torques12 = np.zeros((12, 1))
        self.tau = np.zeros((12, ))

        self.ID_base = None  # ID of base link
        self.ID_feet = [None] * 4  # ID of feet links

        # Footstep planner object
        # self.fstep_planner = FootstepPlanner.FootstepPlanner(0.001, 32)
        self.vu_m = np.zeros((6, 1))
        self.t_stance = T_gait * 0.5
        self.T_gait = T_gait
        self.n_periods = n_periods
        self.h_ref = 0.235 - 0.01205385
        self.t_swing = np.zeros((4, ))  # Total duration of current swing phase for each foot

        self.contacts_order = [0, 1, 2, 3]

        # Parameter to enable/disable hybrid control
        self.enable_hybrid_control = False

        # Time since the start of the simulation
        self.t = 0.0

        ########################################################################
        #             Definition of the Model and TSID problem                 #
        ########################################################################

        # Set the paths where the urdf and srdf file of the robot are registered
        modelPath = "/opt/openrobots/share/example-robot-data/robots"
        urdf = modelPath + "/solo_description/robots/solo12.urdf"
        srdf = modelPath + "/solo_description/srdf/solo.srdf"
        vector = pin.StdVec_StdString()
        vector.extend(item for item in modelPath)

        # Create the robot wrapper from the urdf model (which has no free flyer) and add a free flyer
        self.robot = tsid.RobotWrapper(urdf, vector, pin.JointModelFreeFlyer(), False)
        self.model = self.robot.model()

        # Creation of the Invverse Dynamics HQP problem using the robot
        # accelerations (base + joints) and the contact forces
        self.invdyn = tsid.InverseDynamicsFormulationAccForce("tsid", self.robot, False)

        # Compute the problem data with a solver based on EiQuadProg
        t = 0.0
        self.invdyn.computeProblemData(t, self.qtsid, self.vtsid)

        # Saving IDs for later
        self.ID_base = self.model.getFrameId("base_link")
        for i, name in enumerate(self.foot_frames):
            self.ID_feet[i] = self.model.getFrameId(name)

        # Store a frame object to avoid creating one each time
        self.pos_foot = self.robot.framePosition(self.invdyn.data(), self.ID_feet[0])

        #####################
        # LEGS POSTURE TASK #
        #####################

        # Task definition (creating the task object)
        self.postureTask = tsid.TaskJointPosture("task-posture", self.robot)
        self.postureTask.setKp(kp_posture * matlib.ones(self.robot.nv-6).T)  # Proportional gain
        self.postureTask.setKd(2.0 * np.sqrt(kp_posture) * matlib.ones(self.robot.nv-6).T)  # Derivative gain

        # Add the task to the HQP with weight = w_posture, priority level = 1 (not real constraint)
        # and a transition duration = 0.0
        self.invdyn.addMotionTask(self.postureTask, w_posture, 1, 0.0)

        # TSID Trajectory (creating the trajectory object and linking it to the task)
        pin.loadReferenceConfigurations(self.model, srdf, False)
        self.trajPosture = tsid.TrajectoryEuclidianConstant("traj_joint", self.q_ref[7:])
        self.samplePosture = self.trajPosture.computeNext()
        self.postureTask.setReference(self.samplePosture)

        ############
        # CONTACTS #
        ############

        self.contacts = 4*[None]  # List to store the rigid contact tasks

        for i, name in enumerate(self.foot_frames):

            # Contact definition (creating the contact object)
            self.contacts[i] = tsid.ContactPoint(name, self.robot, name, contactNormal, mu, fMin, fMax)
            self.contacts[i].setKp((kp_contact * matlib.ones(3).T))
            self.contacts[i].setKd((2.0 * np.sqrt(kp_contact) * matlib.ones(3).T))
            self.contacts[i].useLocalFrame(False)

            # Set the contact reference position
            H_ref = self.robot.framePosition(self.invdyn.data(), self.ID_feet[i])
            H_ref.translation = np.matrix(
                [H_ref.translation[0, 0],
                 H_ref.translation[1, 0],
                 0.0]).T
            self.contacts[i].setReference(H_ref)

            # Regularization weight for the force tracking subtask
            self.contacts[i].setRegularizationTaskWeightVector(
                np.matrix([self.w_reg_f, self.w_reg_f, self.w_reg_f]).T)

            # Adding the rigid contact after the reference contact force has been set
            self.invdyn.addRigidContact(self.contacts[i], self.w_forceRef)

        #######################
        # FOOT TRACKING TASKS #
        #######################

        self.feetTask = 4*[None]  # List to store the foot tracking tasks
        mask = np.matrix([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]).T

        # Task definition (creating the task object)
        for i_foot in range(4):
            self.feetTask[i_foot] = tsid.TaskSE3Equality(
                "foot_track_" + str(i_foot), self.robot, self.foot_frames[i_foot])
            self.feetTask[i_foot].setKp(kp_foot * mask)
            self.feetTask[i_foot].setKd(2.0 * np.sqrt(kp_foot) * mask)
            self.feetTask[i_foot].setMask(mask)
            self.feetTask[i_foot].useLocalFrame(False)

        # The reference will be set later when the task is enabled

        ######################
        # TRUNK POSTURE TASK #
        ######################

        # Task definition (creating the task object)
        self.trunkTask = tsid.TaskSE3Equality("task-trunk", self.robot, 'base_link')
        mask = np.matrix([0.0, 0.0, 0.0, 1.0, 1.0, 0.0]).T
        self.trunkTask.setKp(np.multiply(kp_trunk, mask))
        self.trunkTask.setKd(2.0 * np.sqrt(np.multiply(kp_trunk, mask)))
        self.trunkTask.useLocalFrame(False)
        self.trunkTask.setMask(mask)

        # Add the task to the HQP with weight = w_trunk, priority level = 1 (not real constraint)
        # and a transition duration = 0.0
        if w_trunk > 0.0:
            self.invdyn.addMotionTask(self.trunkTask, w_trunk, 1, 0.0)

        # TSID Trajectory (creating the trajectory object and linking it to the task)
        self.trunk_ref = self.robot.framePosition(self.invdyn.data(), self.ID_base)
        self.trajTrunk = tsid.TrajectorySE3Constant("traj_base_link", self.trunk_ref)
        self.sampleTrunk = self.trajTrunk.computeNext()
        self.sampleTrunk.pos(np.matrix([0.0, 0.0, 0.2027682, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).T)
        self.sampleTrunk.vel(np.matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T)
        self.sampleTrunk.acc(np.matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T)
        self.trunkTask.setReference(self.sampleTrunk)

        ##########
        # SOLVER #
        ##########

        # Use EiquadprogFast solver
        self.solver = tsid.SolverHQuadProgFast("qp solver")

        # Resize the solver to fit the number of variables, equality and inequality constraints
        self.solver.resize(self.invdyn.nVar, self.invdyn.nEq, self.invdyn.nIn)

    def update_feet_tasks(self, k_loop, gait, looping, interface, ftps_Ids_deb):
        """Update the 3D desired position for feet in swing phase by using a 5-th order polynomial that lead them
           to the desired position on the ground (computed by the footstep planner)

        Args:
            k_loop (int): number of time steps since the start of the current gait cycle
            pair (int): the current pair of feet in swing phase, for a walking trot gait
            looping (int): total number of time steps in one gait cycle
            interface (Interface object): interface between the simulator and the MPC/InvDyn
            ftps_Ids_deb (list): IDs of debug spheres in PyBullet
        """

        # Indexes of feet in swing phase
        feet = np.where(gait[0, 1:] == 0)[0]
        if len(feet) == 0:  # If no foot in swing phase
            return 0

        t0s = []
        for i in feet:  # For each foot in swing phase get remaining duration of the swing phase
            # Index of the line containing the next stance phase
            index = next((idx for idx, val in np.ndenumerate(gait[:, 1+i]) if (((val==1)))), [-1])[0]
            remaining_iterations = np.cumsum(gait[:index, 0])[-1] * self.k_mpc - ((k_loop+1) % self.k_mpc)

            # Compute total duration of current swing phase
            i_iter = 1
            self.t_swing[i] = gait[0, 0]
            while gait[i_iter, 1+i] == 0:
                self.t_swing[i] += gait[i_iter, 0]
                i_iter += 1
            i_iter = -1
            while gait[i_iter, 1+i] == 0:
                self.t_swing[i] += gait[i_iter, 0]
                i_iter -= 1
            self.t_swing[i] *= self.dt * self.k_mpc

            t0s.append(np.round(self.t_swing[i] - remaining_iterations * self.dt - 0.001, decimals=3))

        # self.footsteps contains the target (x, y) positions for both feet in swing phase

        for i in range(len(feet)):
            i_foot = feet[i]

            # Get desired 3D position, velocity and acceleration
            if t0s[i] == 0.000:
                [x0, dx0, ddx0,  y0, dy0, ddy0,  z0, dz0, ddz0, gx1, gy1] = (self.ftgs[i_foot]).get_next_foot(
                    interface.o_feet[0, i_foot], interface.ov_feet[0, i_foot], interface.oa_feet[0, i_foot],
                    interface.o_feet[1, i_foot], interface.ov_feet[1, i_foot], interface.oa_feet[1, i_foot],
                    self.footsteps[0, i_foot], self.footsteps[1, i_foot], t0s[i],  self.t_swing[i_foot], self.dt)
                self.mgoals[:, i_foot] = np.array([x0, dx0, ddx0, y0, dy0, ddy0])
            else:
                [x0, dx0, ddx0,  y0, dy0, ddy0,  z0, dz0, ddz0, gx1, gy1] = (self.ftgs[i_foot]).get_next_foot(
                    self.mgoals[0, i_foot], self.mgoals[1, i_foot], self.mgoals[2, i_foot],
                    self.mgoals[3, i_foot], self.mgoals[4, i_foot], self.mgoals[5, i_foot],
                    self.footsteps[0, i_foot], self.footsteps[1, i_foot], t0s[i],  self.t_swing[i_foot], self.dt)
                self.mgoals[:, i_foot] = np.array([x0, dx0, ddx0, y0, dy0, ddy0])

            # Take into account vertical offset of Pybullet
            z0 += interface.mean_feet_z
            # z0 += self.pos_contact[i_foot][0, 2]

            # Store desired position, velocity and acceleration for later call to this function
            self.goals[:, i_foot] = np.array([x0, y0, z0])
            self.vgoals[:, i_foot] = np.array([dx0, dy0, dz0])
            self.agoals[:, i_foot] = np.array([ddx0, ddy0, ddz0])

            # Update desired pos, vel, acc
            self.sampleFeet[i_foot].pos(np.matrix([x0, y0, z0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).T)
            self.sampleFeet[i_foot].vel(np.matrix([dx0, dy0, dz0, 0.0, 0.0, 0.0]).T)
            self.sampleFeet[i_foot].acc(np.matrix([ddx0, ddy0, ddz0, 0.0, 0.0, 0.0]).T)

            # Set reference
            self.feetTask[i_foot].setReference(self.sampleFeet[i_foot])

            # Update footgoal for display purpose
            self.feetGoal[i_foot].translation = np.matrix([x0, y0, z0]).T

            # Display the goal position of the feet as green sphere in PyBullet
            pyb.resetBasePositionAndOrientation(ftps_Ids_deb[i_foot],
                                                posObj=np.array([gx1, gy1, 0.0]),
                                                ornObj=np.array([0.0, 0.0, 0.0, 1.0]))

        return 0

    ####################################################################
    #                      Torque Control method                       #
    ####################################################################

    def control(self, qtsid, vtsid, k_simu, solo, interface, f_applied, fsteps, gait,
                ftps_Ids_deb, enable_hybrid_control=False, qmes=None, vmes=None, qmpc=None, vmpc=None):
        """Update the 3D desired position for feet in swing phase by using a 5-th order polynomial that lead them
           to the desired position on the ground (computed by the footstep planner)

        Args:
            qtsid (19x1 array): the position/orientation of the trunk and angular position of actuators
            vtsid (18x1 array): the linear/angular velocity of the trunk and angular velocity of actuators
            t (float): time elapsed since the start of the simulation
            k_simu (int): number of time steps since the start of the simulation
            solo (object): Pinocchio wrapper for the quadruped
            interface (Interface object): interface between the simulator and the MPC/InvDyn
            f_applied (12 array): desired contact forces for all feet (0s for feet in swing phase)
            fsteps (Xx13 array): contains the remaining number of steps of each phase of the gait (first column) and
                                 the [x, y, z]^T desired position of each foot for each phase of the gait (12 other
                                 columns). For feet currently touching the ground the desired position is where they
                                 currently are.
            enable_hybrid_control (bool): whether hybrid control is enabled or not
            qmes (19x1 array): the position/orientation of the trunk and angular position of actuators of the real
                               robot (for hybrid control)
            vmes (18x1 array): the linear/angular velocity of the trunk and angular velocity of actuators of the real
                               robot (for hybrid control)
        """

        self.f_applied = f_applied

        # If hybrid control is turned on/off the CoM task needs to be turned on/off too
        """if self.enable_hybrid_control != enable_hybrid_control:
            if enable_hybrid_control:
                # Turn on CoM task
                self.invdyn.addMotionTask(self.comTask, self.w_com, 1, 0.0)
            else:
                # Turn off CoM task
                self.invdyn.removeTask(self.comTask, 0.0)"""

        # Update hybrid control parameters
        self.enable_hybrid_control = enable_hybrid_control
        if self.enable_hybrid_control:
            self.qmes = qmes
            self.vmes = vmes

        # Update state of TSID
        if k_simu == 0:  # Some initialization during the first iteration
            self.qtsid = qtsid
            self.qtsid[:3] = np.zeros((3, 1))  # Discard x and y drift and height position
            self.qtsid[2, 0] = 0.235 - 0.01205385

            self.feetGoal = 4*[None]
            self.sampleFeet = 4*[None]
            self.pos_contact = 4*[None]
            for i_foot in range(4):
                self.feetGoal[i_foot] = self.robot.framePosition(
                    self.invdyn.data(), self.ID_feet[i_foot])
                footTraj = tsid.TrajectorySE3Constant("foot_traj", self.feetGoal[i_foot])
                self.sampleFeet[i_foot] = footTraj.computeNext()

                self.pos_contact[i_foot] = np.matrix([self.footsteps[0, i_foot], self.footsteps[1, i_foot], 0.0])
        else:
            # Here is where we will merge the data from the state estimator and the internal state of TSID
            """# Encoders (position of joints)
            self.qtsid[7:] = qtsid[7:]

            # Gyroscopes (angular velocity of trunk)
            self.vtsid[3:6] = vtsid[3:6]

            # IMU estimation of orientation of the trunk
            self.qtsid[3:7] = qtsid[3:7]"""

            """self.qtsid = qtsid.copy()
            # self.qtsid[2] -= 0.015  # 0.01205385
            self.vtsid = vtsid.copy()

            self.qtsid[2, 0] += mpc.q_noise[0]
            self.qtsid[3:7] = utils.getQuaternion(utils.quaternionToRPY(
                qtsid[3:7, 0]) + np.vstack((np.array([mpc.q_noise[1:]]).transpose(), 0.0)))
            self.vtsid[:6, 0] += mpc.v_noise"""

            # Update internal state of TSID for the current interation
            self.update_state(qtsid, vtsid)

        #####################
        # FOOTSTEPS PLANNER #
        #####################

        looping = int(self.n_periods*self.T_gait/dt)  # Number of TSID iterations in one gait cycle
        k_loop = (k_simu - 0) % looping  # Current number of iterations since the start of the current gait cycle

        # Update the desired position of footholds thanks to the footstep planner
        self.update_footsteps(interface, fsteps)

        ######################################
        # UPDATE REFERENCE OF CONTACT FORCES #
        ######################################

        # Update the contact force tracking tasks to follow the forces computed by the MPC
        self.update_ref_forces(interface)

        ################
        # UPDATE TASKS #
        ################

        # Enable/disable contact and 3D tracking tasks depending on the state of the feet (swing or stance phase)
        self.update_tasks(k_simu, k_loop, looping, interface, gait, ftps_Ids_deb)

        ###############
        # HQP PROBLEM #
        ###############

        # Solve the inverse dynamics problem with TSID
        self.solve_HQP_problem(self.t)

        # Time incrementation
        self.t += dt

        ###########
        # DISPLAY #
        ###########

        # Refresh Gepetto Viewer
        solo.display(self.qtsid)

        return self.tau

    def update_state(self, qtsid, vtsid):
        """Update TSID's internal state.

        Currently we directly use the state of the simulator to perform the inverse dynamics

        Args:
            qtsid (19x1 array): the position/orientation of the trunk and angular position of actuators
            vtsid (18x1 array): the linear/angular velocity of the trunk and angular velocity of actuators
        """

        self.qtsid = qtsid.copy()
        self.vtsid = vtsid.copy()

        return 0

    def update_footsteps(self, interface, fsteps):
        """ Update desired location of footsteps using information coming from the footsteps planner

        Args:
            interface (object): Interface object of the control loop
            fsteps (20x13): duration of each phase of the gait sequence (first column)
                            and desired location of footsteps for these phases (other columns)
        """

        self.footsteps = np.zeros((2, 4))

        for i in range(4):
            index = next((idx for idx, val in np.ndenumerate(fsteps[:, 3*i+1]) if ((not (val==0)) and (not np.isnan(val)))), [-1])[0]
            pos_tmp = np.array(interface.oMl * (np.array([fsteps[index, (1+i*3):(4+i*3)]]).transpose()))
            self.footsteps[:, i] = pos_tmp[0:2, 0]

        return 0

    def update_ref_forces(self, interface):
        """ Update the reference contact forces that TSID should try to apply on the ground

        Args:
            interface (object): Interface object of the control loop
        """

        for j, i_foot in enumerate([0, 1, 2, 3]):
            self.contacts[i_foot].setForceReference((self.w_reg_f * interface.oMl.rotation @ self.f_applied[3*j:3*(j+1)]).T)

        return 0

    def update_tasks(self, k_simu, k_loop, looping, interface, gait, ftps_Ids_deb):
        """ Update TSID tasks (feet tracking, contacts, force tracking)

        Args:
            k_simu (int): number of TSID time steps since the start of the simulation
            k_loop (int): number of TSID time steps since the start of the current gait period
            looping (int): number of TSID time steps in one period of gait
            interface (object): Interface object of the control loop
            gait (20x5 array): contains information about the contact sequence with 1s and 0s
            fsteps (20x13): duration of each phase of the gait sequence (first column)
                            and desired location of footsteps for these phases (other columns)
        """

        # Update the foot tracking tasks
        self.update_feet_tasks(k_loop, gait, looping, interface, ftps_Ids_deb)

        # Index of the first blank line in the gait matrix
        index = next((idx for idx, val in np.ndenumerate(gait[:, 0]) if (((val==0)))), [-1])[0]

        # Check status of each foot
        for i_foot in range(4):

            # If foot entered swing phase
            if (k_loop % self.k_mpc == 0) and (gait[0, i_foot+1] == 0) and (gait[index-1, i_foot+1] == 1):
                # Disable contact
                self.invdyn.removeRigidContact(self.foot_frames[i_foot], 0.0)
                self.contacts_order.remove(i_foot)

                # Enable foot tracking task
                self.invdyn.addMotionTask(self.feetTask[i_foot], self.w_foot, 1, 0.0)

            # If foot entered stance phase
            if (k_loop % self.k_mpc == 0) and (gait[0, i_foot+1] == 1) and (gait[index-1, i_foot+1] == 0):

                # Update the position of contacts
                self.pos_foot.translation = interface.o_feet[:, i_foot]
                self.pos_foot.translation[2, 0] = 0.0
                self.pos_contact[i_foot] = self.pos_foot.translation.transpose()
                self.memory_contacts[:, i_foot] = interface.o_feet[0:2, i_foot]
                self.feetGoal[i_foot].translation = interface.o_feet[:, i_foot].transpose()
                self.feetGoal[i_foot].translation[2, 0] = 0.0
                self.contacts[i_foot].setReference(self.pos_foot)
                self.goals[:, i_foot] = interface.o_feet[:, i_foot].transpose()
                self.goals[2, i_foot] = 0.0

                if not ((k_loop == 0) and (k_simu < looping)):  # If it is not the first gait period
                    # Enable contact
                    self.invdyn.addRigidContact(self.contacts[i_foot], self.w_forceRef)
                    self.contacts_order.append(i_foot)

                    # Disable foot tracking task
                    self.invdyn.removeTask("foot_track_" + str(i_foot), 0.0)

        return 0

    def solve_HQP_problem(self, t):
        """ Solve the QP problem by calling TSID's solver

        Args:
            t (float): time elapsed since the start of the simulation
        """

        # Resolution of the HQP problem
        HQPData = self.invdyn.computeProblemData(t, self.qtsid, self.vtsid)
        self.sol = self.solver.solve(HQPData)

        # Torques, accelerations, velocities and configuration computation
        self.tau_ff = self.invdyn.getActuatorForces(self.sol)
        self.fc = self.invdyn.getContactForces(self.sol)
        self.ades = self.invdyn.getAccelerations(self.sol)
        if self.enable_hybrid_control:
            self.vtsid += self.ades * dt
            self.qtsid = pin.integrate(self.model, self.qtsid, self.vtsid * dt)

        # Check for NaN value in the output torques (means error during solving process)
        if np.any(np.isnan(self.tau_ff)):
            self.error = True
            self.tau = np.zeros((1, 12))
            raise ValueError('NaN value in feedforward torque')
        else:
            # Torque PD controller
            P = 3.0
            D = 0.3
            if self.enable_hybrid_control:
                torques12 = self.tau_ff + P * (self.qtsid[7:] - self.qmes[7:]) + D * (self.vtsid[6:] - self.vmes[6:])
            else:
                torques12 = self.tau_ff

            # Saturation to limit the maximal torque
            t_max = 2.5
            # clip is faster than np.maximum(a_min, np.minimum(a, a_max))
            self.tau = np.clip(torques12, -t_max, t_max).flatten()

        return 0

    def display(self, t, solo, k_simu, sequencer):
        """ To display debug spheres in Gepetto Viewer
        May not be up to date.
        """
        if self.verbose:
            # Display target 3D positions of footholds with green spheres (gepetto gui)
            rgbt = [0.0, 1.0, 0.0, 0.5]
            for i in range(0, 4):
                if (t == 0):
                    solo.viewer.gui.addSphere("world/sphere"+str(i)+"_target", .02, rgbt)  # .1 is the radius
                solo.viewer.gui.applyConfiguration(
                    "world/sphere"+str(i)+"_target", (self.feetGoal[i].translation[0, 0],
                                                      self.feetGoal[i].translation[1, 0],
                                                      self.feetGoal[i].translation[2, 0], 1., 0., 0., 0.))
                # print("Foothold " + str(i) + " : " + self.feetGoal[i].translation.transpose())

            # Display current 3D positions of footholds with magenta spheres (gepetto gui)
            rgbt = [1.0, 0.0, 1.0, 0.5]
            for i in range(0, 4):
                if (t == 0):
                    solo.viewer.gui.addSphere("world/sphere"+str(i)+"_pos", .02, rgbt)  # .1 is the radius
                self.pos_foot = self.robot.framePosition(self.invdyn.data(), self.ID_feet[i])
                solo.viewer.gui.applyConfiguration(
                    "world/sphere"+str(i)+"_pos", (self.pos_foot.translation[0, 0],
                                                   self.pos_foot.translation[1, 0],
                                                   self.pos_foot.translation[2, 0], 1., 0., 0., 0.))

            # Display target 3D positions of footholds with green spheres (gepetto gui)
            rgbt = [0.0, 0.0, 1.0, 0.5]
            for i in range(0, 4):
                if (t == 0):
                    solo.viewer.gui.addSphere("world/shoulder"+str(i), .02, rgbt)  # .1 is the radius
                solo.viewer.gui.applyConfiguration(
                    "world/shoulder"+str(i), (self.shoulders[0, i], self.shoulders[1, i], 0.0, 1., 0., 0., 0.))

            # Display 3D positions of sampleFeet
            """rgbt = [0.3, 1.0, 1.0, 0.5]
            for i in range(0, 4):
                if (t == 0):
                    solo.viewer.gui.addSphere("world/sfeet"+str(i), .02, rgbt)  # .1 is the radius
                solo.viewer.gui.applyConfiguration(
                    "world/sfeet"+str(i), (self.sampleFeet[i].pos()[0, 0],
                                           self.sampleFeet[i].pos()[1, 0],
                                           self.sampleFeet[i].pos()[2, 0], 1., 0., 0., 0.))"""

            # Display lines for contact forces
            if (t == 0):
                for i in range(4):
                    solo.viewer.gui.addCurve("world/force_curve"+str(i),
                                             [[0., 0., 0.], [0., 0., 0.]], [1.0, 0.0, 0.0, 0.5])
                    solo.viewer.gui.setCurveLineWidth("world/force_curve"+str(i), 8.0)
                    solo.viewer.gui.setColor("world/force_curve"+str(i), [1.0, 0.0, 0.0, 0.5])
            else:
                """if self.pair == 1:
                    feet = [1, 2]
                    feet_0 = [0, 3]
                else:
                    feet = [0, 3]
                    feet_0 = [1, 2]"""
                feet = [0, 1, 2, 3]
                feet_stance = np.where(sequencer.S[0, :] == 1)[0]
                cpt_foot = 0
                for i, i_foot in enumerate(feet):
                    if i_foot in feet_stance:
                        Kreduce = 0.04
                        solo.viewer.gui.setCurvePoints("world/force_curve"+str(i_foot),
                                                       [[self.memory_contacts[0, i_foot],
                                                         self.memory_contacts[1, i_foot], 0.0],
                                                        [self.memory_contacts[0, i_foot] + Kreduce * self.fc[3*cpt_foot+0, 0],
                                                         self.memory_contacts[1, i_foot] +
                                                         Kreduce * self.fc[3*cpt_foot+1, 0],
                                                         Kreduce * self.fc[3*cpt_foot+2, 0]]])
                        cpt_foot += 1
                    else:
                        solo.viewer.gui.setCurvePoints("world/force_curve"+str(i_foot),
                                                       [[self.memory_contacts[0, i_foot],
                                                         self.memory_contacts[1, i_foot], 0.0],
                                                        [self.memory_contacts[0, i_foot] + 0.0,
                                                         self.memory_contacts[1, i_foot] + 0.0,
                                                         0.0]])
                """for i, i_foot in enumerate(feet_0):
                    solo.viewer.gui.setCurvePoints("world/force_curve"+str(i_foot),
                                                   [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])"""

            """if (t == 0):
                solo.viewer.gui.addCurve("world/orientation_curve",
                                         [[0., 0., 0.], [0., 0., 0.]], [1.0, 0.0, 0.0, 0.5])
                solo.viewer.gui.setCurveLineWidth("world/orientation_curve", 8.0)
                solo.viewer.gui.setColor("world/orientation_curve", [1.0, 0.0, 0.0, 0.5])

            pos_trunk = self.robot.framePosition(self.invdyn.data(), self.ID_base)
            line_rot = np.dot(pos_trunk.rotation, np.array([[1, 0, 0]]).transpose())
            solo.viewer.gui.setCurvePoints("world/orientation_curve",
                                           [pos_trunk.translation.flatten().tolist()[0],
                                            (pos_trunk.translation + line_rot).flatten().tolist()[0]])"""

            """if k_simu == 0:
                solo.viewer.gui.setRefreshIsSynchronous(False)"""

            # Refresh gepetto gui with TSID desired joint position
            if k_simu % 1 == 0:
                solo.viewer.gui.refresh()
                solo.display(self.qtsid)

    def log(self, t, solo, k_simu, sequencer, interface):
        """ To log various quantities
        Not used anymore since logging is done by the Logger object
        """

        self.h_ref_feet[k_simu] = interface.mean_feet_z

        # Log pos, vel, acc of the flying foot
        for i_foot in range(4):
            self.f_pos_ref[i_foot, k_simu:(k_simu+1), :] = self.sampleFeet[i_foot].pos()[0:3].transpose()
            self.f_vel_ref[i_foot, k_simu:(k_simu+1), :] = self.sampleFeet[i_foot].vel()[0:3].transpose()
            self.f_acc_ref[i_foot, k_simu:(k_simu+1), :] = self.sampleFeet[i_foot].acc()[0:3].transpose()

            pos = self.robot.framePosition(self.invdyn.data(), self.ID_feet[i_foot])
            vel = self.robot.frameVelocityWorldOriented(
                self.invdyn.data(), self.ID_feet[i_foot])
            acc = self.robot.frameAccelerationWorldOriented(
                self.invdyn.data(), self.ID_feet[i_foot])
            self.f_pos[i_foot, k_simu:(k_simu+1), :] = interface.o_feet[:, i_foot]  # pos.translation[0:3].transpose()
            self.f_vel[i_foot, k_simu:(k_simu+1), :] = vel.vector[0:3].transpose()
            self.f_acc[i_foot, k_simu:(k_simu+1), :] = acc.vector[0:3].transpose()

        c_f = np.zeros((3, 4))
        for i, j in enumerate(np.where(sequencer.S[0, :] == 1)[0]):
            c_f[:, j:(j+1)] = self.fc[(3*i):(3*(i+1))]
        self.c_forces[:, k_simu, :] = c_f.transpose()  #  self.fc.reshape((4, 3))

        # Log position of the base
        pos_trunk = self.robot.framePosition(self.invdyn.data(), self.ID_base)
        self.b_pos[k_simu:(k_simu+1), 0:3] = pos_trunk.translation[0:3].transpose()
        vel_trunk = self.robot.frameVelocityWorldOriented(self.invdyn.data(), self.ID_base)
        self.b_vel[k_simu:(k_simu+1), 0:3] = vel_trunk.vector[0:3].transpose()

        # Log position and reference of the CoM
        com_ref = self.robot.com(self.invdyn.data())
        self.trajCom = tsid.TrajectoryEuclidianConstant("traj_com", com_ref)
        sample_com = self.trajCom.computeNext()

        self.com_pos_ref[k_simu:(k_simu+1), 0:3] = sample_com.pos().transpose()
        self.com_pos[k_simu:(k_simu+1), 0:3] = self.robot.com(self.invdyn.data()).transpose()

# Parameters for the controller


dt = 0.001			# controller time step
