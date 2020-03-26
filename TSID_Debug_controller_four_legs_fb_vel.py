# coding: utf8


########################################################################
#                                                                      #
#          Control law : tau = P(q*-q^) + D(v*-v^) + tau_ff            #
#                                                                      #
########################################################################

import pinocchio as pin
import numpy as np
import numpy.matlib as matlib
import tsid
import FootTrajectoryGenerator as ftg
import FootstepPlanner
import pybullet as pyb
import utils
import time
pin.switchToNumpyMatrix()


########################################################################
#            Class for a PD with feed-forward Controller               #
########################################################################

class controller:

    def __init__(self, q0, omega, t, N_simulation):

        self.q_ref = np.array([[0.0, 0.0, 0.235 - 0.01205385, 0.0, 0.0, 0.7071067811865475, 0.7071067811865475,
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
        mu = 1  				# friction coefficient
        fMin = 1.0				# minimum normal force
        fMax = 25.0  			# maximum normal force
        contactNormal = np.matrix([0., 0., 1.]).T  # direction of the normal to the contact surface

        # Coefficients of the posture task
        kp_posture = 10.0		# proportionnal gain of the posture task
        w_posture = 1.0         # weight of the posture task

        # Coefficients of the contact tasks
        kp_contact = 0.0         # proportionnal gain for the contacts
        self.w_forceRef = 100.0  # weight of the forces regularization
        self. w_reg_f = 100.0

        # Coefficients of the foot tracking task
        kp_foot = 100.0               # proportionnal gain for the tracking task
        self.w_foot = 10000.0       # weight of the tracking task

        # Coefficients of the trunk task
        kp_trunk = np.matrix([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).T
        w_trunk = 30.0

        # Coefficients of the CoM task
        self.kp_com = 300
        self.w_com = 1000.0  #  1000.0
        offset_x_com = - 0.00  # offset along X for the reference position of the CoM

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

        # Position of the shoulders in local frame
        self.shoulders = np.array([[0.19, 0.19, -0.19, -0.19], [0.15005, -0.15005, 0.15005, -0.15005]])
        R = np.array([[0.0, -1.0], [1.0, 0.0]])
        self.footsteps = R @ self.shoulders.copy()
        self.memory_contacts = R @ self.shoulders.copy()

        # Foot trajectory generator
        max_height_feet = 0.05
        t_lock_before_touchdown = 0.05
        self.ftgs = [ftg.Foot_trajectory_generator(max_height_feet, t_lock_before_touchdown) for i in range(4)]

        # Which pair of feet is active (0 for [1, 2] and 1 for [0, 3])
        self.pair = -1

        # For update_feet_tasks function
        self.dt = 0.001  #  [s], time step
        self.t1 = 0.16  # [s], duration of swing phase

        # Rotation along the vertical axis
        delta_yaw = (2 * np.pi / 10) * t
        c, s = np.cos(delta_yaw), np.sin(delta_yaw)
        self.R_yaw = np.array([[c, s], [-s, c]])

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
        self.fstep_planner = FootstepPlanner.FootstepPlanner(0.001, 16)
        self.v_ref = np.zeros((6, 1))
        self.vu_m = np.zeros((6, 1))
        self.t_stance = 0.16
        self.T_gait = 0.32
        self.t_remaining = np.zeros((1, 4))
        self.h_ref = 0.235 - 0.01205385

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

            """w_reg_f = 100
            if i in [0, 1]:
                self.contacts[i].setForceReference(np.matrix([0.0, 0.0, w_reg_f * 14.0]).T)
            else:
                self.contacts[i].setForceReference(np.matrix([0.0, 0.0, w_reg_f * 17.0]).T)
            self.contacts[i].setRegularizationTaskWeightVector(np.matrix([w_reg_f, w_reg_f, w_reg_f]).T)"""
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
        mask = np.matrix([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).T
        self.trunkTask.setKp(np.multiply(kp_trunk, mask))
        self.trunkTask.setKd(2.0 * np.sqrt(np.multiply(kp_trunk, mask)))
        self.trunkTask.useLocalFrame(False)
        self.trunkTask.setMask(mask)

        # Add the task to the HQP with weight = w_trunk, priority level = 1 (not real constraint)
        # and a transition duration = 0.0
        # if w_trunk > 0.0:
        # self.invdyn.addMotionTask(self.trunkTask, w_trunk, 1, 0.0)

        # TSID Trajectory (creating the trajectory object and linking it to the task)
        self.trunk_ref = self.robot.framePosition(self.invdyn.data(), self.ID_base)
        self.trajTrunk = tsid.TrajectorySE3Constant("traj_base_link", self.trunk_ref)
        self.sampleTrunk = self.trajTrunk.computeNext()
        self.sampleTrunk.pos(np.matrix([0.0, 0.0, 0.2027682, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).T)
        self.sampleTrunk.vel(np.matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T)
        self.sampleTrunk.acc(np.matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T)
        self.trunkTask.setReference(self.sampleTrunk)

        ############
        # COM TASK #
        ############

        # Task definition
        self.comTask = tsid.TaskComEquality("task-com", self.robot)
        self.comTask.setKp(self.kp_com * matlib.ones(3).T)
        self.comTask.setKd(2.0 * np.sqrt(self.kp_com) * matlib.ones(3).T)
        # if self.w_com > 0.0:
        #    self.invdyn.addMotionTask(self.comTask, self.w_com, 1, 0.0)

        # Task reference
        com_ref = self.robot.com(self.invdyn.data())
        self.trajCom = tsid.TrajectoryEuclidianConstant("traj_com", com_ref)
        self.sample_com = self.trajCom.computeNext()

        tmp = self.sample_com.pos()  # Temp variable to store CoM position
        tmp[0, 0] += offset_x_com
        self.sample_com.pos(tmp)
        self.comTask.setReference(self.sample_com)

        ##########
        # SOLVER #
        ##########

        # Use EiquadprogFast solver
        self.solver = tsid.SolverHQuadProgFast("qp solver")

        # Resize the solver to fit the number of variables, equality and inequality constraints
        self.solver.resize(self.invdyn.nVar, self.invdyn.nEq, self.invdyn.nIn)

    ####################################################################
    #           Method to updated desired foot position                #
    ####################################################################

    def update_feet_tasks(self, k_loop, pair, looping, mpc_interface):

        # Target (x, y) positions for both feet
        # x1 = self.footsteps[0, :]
        # y1 = self.footsteps[1, :]

        if k_loop == 19:
            deb = 1

        if pair == -1:
            return 0
        elif pair == 0:
            t0 = ((k_loop-1) / (looping*0.5-1)) * self.t1  #  ((k_loop-20) / 280) * t1
            feet = [1, 2]
        else:
            t0 = ((k_loop-(looping*0.5+1)) / (looping*0.5-1)) * self.t1  # ((k_loop-320) / 280) * t1
            feet = [0, 3]

        for i_foot in feet:

            # Get desired 3D position, velocity and acceleration
            [x0, dx0, ddx0,  y0, dy0, ddy0,  z0, dz0, ddz0, gx1, gy1] = (self.ftgs[i_foot]).get_next_foot(
                mpc_interface.o_feet[0, i_foot], mpc_interface.ov_feet[0, i_foot], mpc_interface.oa_feet[0, i_foot],
                mpc_interface.o_feet[1, i_foot], mpc_interface.ov_feet[1, i_foot], mpc_interface.oa_feet[0, i_foot],
                self.footsteps[0, i_foot], self.footsteps[1, i_foot], t0,  self.t1, self.dt)

            # Take into account vertical offset of Pybullet
            z0 += mpc_interface.mean_feet_z

            # Update desired pos, vel, acc
            self.sampleFeet[i_foot].pos(np.matrix([x0, y0, z0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).T)
            self.sampleFeet[i_foot].vel(np.matrix([dx0, dy0, dz0, 0.0, 0.0, 0.0]).T)
            self.sampleFeet[i_foot].acc(np.matrix([ddx0, ddy0, ddz0, 0.0, 0.0, 0.0]).T)

            # Set reference
            self.feetTask[i_foot].setReference(self.sampleFeet[i_foot])

            # Update footgoal for display purpose
            self.feetGoal[i_foot].translation = np.matrix([x0, y0, z0]).T

        return 0

    ####################################################################
    #                      Torque Control method                       #
    ####################################################################

    def control(self, qmes12, vmes12, t, k_simu, solo, sequencer, mpc_interface, v_ref, f_applied, fsteps):

        self.v_ref = v_ref
        self.f_applied = f_applied

        if k_simu == 0:
            self.qtsid = qmes12
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
            """# Encoders (position of joints)
            self.qtsid[7:] = qmes12[7:]

            # Gyroscopes (angular velocity of trunk)
            self.vtsid[3:6] = vmes12[3:6]

            # IMU estimation of orientation of the trunk
            self.qtsid[3:7] = qmes12[3:7]"""

            """self.qtsid = qmes12.copy()
            # self.qtsid[2] -= 0.015  # 0.01205385
            self.vtsid = vmes12.copy()

            self.qtsid[2, 0] += mpc.q_noise[0]
            self.qtsid[3:7] = utils.getQuaternion(utils.quaternionToRPY(
                qmes12[3:7, 0]) + np.vstack((np.array([mpc.q_noise[1:]]).transpose(), 0.0)))
            self.vtsid[:6, 0] += mpc.v_noise"""

            self.update_state(qmes12, vmes12)

        #####################
        # FOOTSTEPS PLANNER #
        #####################

        looping = int(self.T_gait/dt)
        k_loop = (k_simu - 0) % looping  # 120  # 600

        self.update_footsteps(k_simu, k_loop, looping, sequencer, mpc_interface, fsteps)

        #############################
        # UPDATE ROTATION ON ITSELF #
        #############################

        # self.footsteps = np.dot(self.R_yaw, self.footsteps)

        #######################
        # UPDATE CoM POSITION #
        #######################

        """tmp = self.sample_com.pos()  # Temp variable to store CoM position
        tmp[0, 0] = np.mean(self.footsteps[0, :])
        tmp[1, 0] = np.mean(self.footsteps[1, :])
        self.sample_com.pos(tmp)"""
        """if k_simu >= 1500 and k_simu < 2000:
            tmp = self.sample_com.vel()
            tmp[0, 0] = + 0.1 * np.min((1.0, 1.0 - (2000 - k_simu) / 500))
            self.sample_com.vel(tmp)"""

        """tmp = self.sample_com.pos()  # Temp variable to store CoM position
        tmp[0, 0] = np.mean(self.footsteps[0, :])
        tmp[1, 0] = np.mean(self.footsteps[1, :])
        self.sample_com.pos(tmp)"""

        """tmp[0:3, 0] = mpc.vu[0:3, 0:1]
        self.sample_com.vel(tmp)
        mass = 2.97784899
        tmp[0, 0] = np.sum(mpc.f_applied[0::3]) / mass
        tmp[1, 0] = np.sum(mpc.f_applied[2::3]) / mass
        tmp[2, 0] = np.sum(mpc.f_applied[3::3]) / mass - 9.81
        self.sample_com.acc(tmp)"""
        # self.comTask.setReference(self.sample_com)

        """self.sampleTrunk.pos(np.matrix([tmp[0, 0], tmp[1, 0], 0.235 - 0.01205385,
                                        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).T)
        self.trunkTask.setReference(self.sampleTrunk)"""

        # print("Desired position of CoM: ", tmp.transpose())

        #####################
        # UPDATE TRUNK TASK #
        #####################

        """RPY = mpc.qu[3:6]
        RPY[1, 0] *= -1  #  Pitch is inversed compared to MPC
        c, s = np.cos(RPY[1, 0]), np.sin(RPY[1, 0])
        R1 = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        c, s = np.cos(RPY[0, 0]), np.sin(RPY[0, 0])
        R2 = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
        c, s = np.cos(RPY[2, 0]), np.sin(RPY[2, 0])
        R3 = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        R = np.dot(R3, np.dot(R2, R1))
        self.sampleTrunk.pos(np.matrix([0.0, 0.0, 0.0, R[0, 0], R[0, 1], R[0, 2],
                                        R[1, 0], R[1, 1], R[1, 2], R[2, 0], R[2, 1], R[2, 2]]).T,)

        tmp = self.sampleTrunk.vel()
        tmp[3:6, 0] = mpc.vu[3:6, 0:1]
        tmp[4, 0] *= -1  #  Pitch is inversed compared to MPC
        self.sampleTrunk.vel(tmp)"""

        # TODO: Angular acceleration?

        #####################################
        # UPDATE REFERENC OF CONTACT FORCES #
        #####################################

        # TODO: Remove "w_reg_f *" in setForceReference once the tsid bug is fixed

        """if k_loop >= 61:  # 320:
            for j, i_foot in enumerate([1, 2]):
                self.contacts[i_foot].setForceReference(self.w_reg_f * np.matrix(mpc.f_applied[3*j:3*(j+1)]).T)
                self.contacts[i_foot].setRegularizationTaskWeightVector(
                    np.matrix([self.w_reg_f, self.w_reg_f, self.w_reg_f]).T)
        elif k_loop >= 60:  # 300:
            for j, i_foot in enumerate([0, 1, 2, 3]):
                self.contacts[i_foot].setForceReference(self.w_reg_f * np.matrix(mpc.f_applied[3*j:3*(j+1)]).T)
                self.contacts[i_foot].setRegularizationTaskWeightVector(
                    np.matrix([self.w_reg_f, self.w_reg_f, self.w_reg_f]).T)
        elif k_loop >= 1:  # 20:
            for j, i_foot in enumerate([0, 3]):
                self.contacts[i_foot].setForceReference(self.w_reg_f * np.matrix(mpc.f_applied[3*j:3*(j+1)]).T)
                self.contacts[i_foot].setRegularizationTaskWeightVector(
                    np.matrix([self.w_reg_f, self.w_reg_f, self.w_reg_f]).T)
        else:
            for j, i_foot in enumerate([0, 1, 2, 3]):
                self.contacts[i_foot].setForceReference(self.w_reg_f * np.matrix(mpc.f_applied[3*j:3*(j+1)]).T)
                self.contacts[i_foot].setRegularizationTaskWeightVector(
                    np.matrix([self.w_reg_f, self.w_reg_f, self.w_reg_f]).T)"""

        self.update_ref_forces(mpc_interface)

        """in_stance_phase = np.where(sequencer.S[0, :] == 1)
        for j, i_foot in enumerate(in_stance_phase[1]):
            self.contacts[i_foot].setForceReference(self.w_reg_f * np.matrix(mpc.f_applied[3*j:3*(j+1)]).T)
            self.contacts[i_foot].setRegularizationTaskWeightVector(
                np.matrix([self.w_reg_f, self.w_reg_f, self.w_reg_f]).T)"""

        ################
        # UPDATE TASKS #
        ################

        self.update_tasks(k_simu, k_loop, looping, mpc_interface)

        ###############
        # HQP PROBLEM #
        ###############

        self.solve_HQP_problem(t)

        return self.tau

    def update_state(self, qmes12, vmes12):

        self.qtsid = qmes12.copy()
        # self.qtsid[2] -= 0.015  # 0.01205385
        self.vtsid = vmes12.copy()

        return 0

    def update_footsteps(self, k_simu, k_loop, looping, sequencer, mpc_interface, fsteps):

        """# self.t_remaining[0, [1, 2]] = np.max((0.0, 0.16 * (looping*0.5 - k_loop) * 0.001))
        self.t_remaining[0, [1, 2]] = 0.16 * (looping*0.5 - k_loop) * 0.001
        (self.t_remaining[0, [1, 2]])[self.t_remaining[0, [1, 2]] < 0.0] = 0.0
        if k_loop < int(looping*0.5):
            self.t_remaining[0, [0, 3]] = 0.0
        else:
            self.t_remaining[0, [0, 3]] = 0.16 * (looping - k_loop) * 0.001"""
        self.test_tmp1(k_loop, looping)

        # Get PyBullet velocity in local frame
        """self.vu_m[0:2, 0:1] = mpc_interface.lV[0:2, 0:1]
        self.vu_m[5, 0] = mpc_interface.lW[2, 0]"""

        # Update desired location of footsteps using the footsteps planner
        self.fstep_planner.update_footsteps_tsid(sequencer, self.v_ref, mpc_interface.lV[0:2, 0:1], self.t_stance,
                                                 self.T_gait, self.qtsid[2, 0])

        # self.footsteps = self.memory_contacts + self.fstep_planner.footsteps_tsid
        """for i in range(4):
            self.footsteps[:, i:(i+1)] = mpc_interface.o_shoulders[0:2, i:(i+1)] + \
                (mpc_interface.oMl.rotation @ self.fstep_planner.footsteps_tsid[:, i]).T[0:2, :]"""
        # self.footsteps = np.array(mpc_interface.o_shoulders + (mpc_interface.oMl.rotation @ self.fstep_planner.footsteps_tsid))[0:2, :]
        self.test_tmp2(mpc_interface, fsteps)

        return 0

    def test_tmp1(self, k_loop, looping):
        # self.t_remaining[0, [1, 2]] = np.max((0.0, 0.16 * (looping*0.5 - k_loop) * 0.001))
        self.t_remaining[0, [1, 2]] = 0.16 * (looping*0.5 - k_loop) * 0.001
        (self.t_remaining[0, [1, 2]])[self.t_remaining[0, [1, 2]] < 0.0] = 0.0
        if k_loop < int(looping*0.5):
            self.t_remaining[0, [0, 3]] = 0.0
        else:
            self.t_remaining[0, [0, 3]] = 0.16 * (looping - k_loop) * 0.001
        return 0

    def test_tmp2(self, mpc_interface, fsteps):

        self.footsteps = np.array(mpc_interface.o_shoulders + (mpc_interface.oMl.rotation @ self.fstep_planner.footsteps_tsid))[0:2, :]

        for i in range(4):
            index = next((idx for idx, val in np.ndenumerate(fsteps[:, 3*i+1]) if (not (val==0))), [-1])[0]
            pos_tmp = np.array(mpc_interface.oMl * (np.array([fsteps[index, (1+i*3):(4+i*3)]]).transpose()))
            self.footsteps[:, i] = pos_tmp[0:2, 0]

        return 0

    def update_ref_forces(self, mpc_interface):

        """RPY = utils.rotationMatrixToEulerAngles(self.robot.framePosition(
            self.invdyn.data(), self.ID_base).rotation)
        c, s = np.cos(RPY[2]), np.sin(RPY[2])
        self.R[:2, :2] = np.array([[c, -s], [s, c]])"""

        for j, i_foot in enumerate([0, 1, 2, 3]):
            self.contacts[i_foot].setForceReference(
                self.w_reg_f * (mpc_interface.oMl.rotation @ self.f_applied[3*j:3*(j+1)]).T)
            """self.contacts[i_foot].setRegularizationTaskWeightVector(
                np.matrix([self.w_reg_f, self.w_reg_f, self.w_reg_f]).T)"""

        return 0

    def update_tasks(self, k_simu, k_loop, looping, mpc_interface):

        if k_simu >= 0:
            if k_loop == 0:  # Start swing phase

                # Update active feet pair
                self.pair = -1

                # Update the foot tracking tasks
                self.update_feet_tasks(k_loop, self.pair, looping, mpc_interface)

                if k_simu >= looping:  # 600:
                    for i_foot in [0, 3]:
                        # Update the position of the contacts and enable them
                        """self.pos_foot = self.robot.framePosition(
                            self.invdyn.data(), self.ID_feet[i_foot])"""
                        self.pos_foot.translation = mpc_interface.o_feet[:, i_foot]
                        self.pos_contact[i_foot] = self.pos_foot.translation.transpose()
                        self.memory_contacts[:, i_foot] = mpc_interface.o_feet[0:2, i_foot]
                        self.feetGoal[i_foot].translation = mpc_interface.o_feet[:, i_foot].transpose()
                        self.contacts[i_foot].setReference(self.pos_foot)
                        self.invdyn.addRigidContact(self.contacts[i_foot], self.w_forceRef)

                        # Disable both foot tracking tasks
                        self.invdyn.removeTask("foot_track_" + str(i_foot), 0.0)

            elif k_loop == 1:

                # Update active feet pair
                self.pair = 0

                # Update the foot tracking tasks
                self.update_feet_tasks(k_loop, self.pair, looping, mpc_interface)

                for i_foot in [1, 2]:
                    # Disable the contacts for both feet (1 and 2)
                    self.invdyn.removeRigidContact(self.foot_frames[i_foot], 0.0)

                    # Enable the foot tracking task for both feet (1 and 2)
                    self.invdyn.addMotionTask(self.feetTask[i_foot], self.w_foot, 1, 0.0)

                for i_foot in [0, 3]:
                    # Update position of contacts
                    """self.pos_foot = self.robot.framePosition(
                        self.invdyn.data(), self.ID_feet[i_foot])"""
                    self.pos_foot.translation = mpc_interface.o_feet[:, i_foot]
                    self.pos_contact[i_foot] = self.pos_foot.translation.transpose()
                    self.memory_contacts[:, i_foot] = mpc_interface.o_feet[0:2, i_foot]
                    self.feetGoal[i_foot].translation = mpc_interface.o_feet[:, i_foot].transpose()
                    self.contacts[i_foot].setReference(self.pos_foot)

            elif k_loop > 1 and k_loop < (looping*0.5):  # 300:

                # Update the foot tracking tasks
                self.update_feet_tasks(k_loop, self.pair, looping, mpc_interface)

                for i_foot in [0, 3]:
                    # Update position of contacts
                    """self.pos_foot = self.robot.framePosition(
                        self.invdyn.data(), self.ID_feet[i_foot])"""
                    self.pos_foot.translation = mpc_interface.o_feet[:, i_foot]
                    self.pos_contact[i_foot] = self.pos_foot.translation.transpose()
                    self.memory_contacts[:, i_foot] = mpc_interface.o_feet[0:2, i_foot]
                    self.feetGoal[i_foot].translation = mpc_interface.o_feet[:, i_foot].transpose()
                    self.contacts[i_foot].setReference(self.pos_foot)

            elif k_loop == (looping*0.5):  # :300:

                # Update active feet pair
                self.pair = -1

                # Update the foot tracking tasks
                self.update_feet_tasks(k_loop, self.pair, looping, mpc_interface)

                for i_foot in [1, 2]:
                    # Update the position of the contacts and enable them
                    """self.pos_foot = self.robot.framePosition(
                        self.invdyn.data(), self.ID_feet[i_foot])"""
                    self.pos_foot.translation = mpc_interface.o_feet[:, i_foot]
                    self.pos_contact[i_foot] = self.pos_foot.translation.transpose()
                    self.memory_contacts[:, i_foot] = mpc_interface.o_feet[0:2, i_foot]
                    self.feetGoal[i_foot].translation = mpc_interface.o_feet[:, i_foot].transpose()
                    self.contacts[i_foot].setReference(self.pos_foot)
                    self.invdyn.addRigidContact(self.contacts[i_foot], self.w_forceRef)

                    # Disable both foot tracking tasks
                    self.invdyn.removeTask("foot_track_" + str(i_foot), 0.0)

            elif k_loop == (looping*0.5+1):  # 320:

                # Update active feet pair
                self.pair = 1

                # Update the foot tracking tasks
                self.update_feet_tasks(k_loop, self.pair, looping, mpc_interface)

                for i_foot in [0, 3]:
                    # Disable the contacts for both feet (0 and 3)
                    self.invdyn.removeRigidContact(self.foot_frames[i_foot], 0.0)

                    # Enable the foot tracking task for both feet (0 and 3)
                    self.invdyn.addMotionTask(self.feetTask[i_foot], self.w_foot, 1, 0.0)

                for i_foot in [1, 2]:
                    # Update position of contacts
                    """self.pos_foot = self.robot.framePosition(
                        self.invdyn.data(), self.ID_feet[i_foot])"""
                    self.pos_foot.translation = mpc_interface.o_feet[:, i_foot]
                    self.pos_contact[i_foot] = self.pos_foot.translation.transpose()
                    self.memory_contacts[:, i_foot] = mpc_interface.o_feet[0:2, i_foot]
                    self.feetGoal[i_foot].translation = mpc_interface.o_feet[:, i_foot].transpose()
                    self.contacts[i_foot].setReference(self.pos_foot)

            else:

                # Update the foot tracking tasks
                self.update_feet_tasks(k_loop, self.pair, looping, mpc_interface)

                for i_foot in [1, 2]:
                    # Update position of contacts
                    """self.pos_foot = self.robot.framePosition(
                        self.invdyn.data(), self.ID_feet[i_foot])"""
                    self.pos_foot.translation = mpc_interface.o_feet[:, i_foot]
                    self.pos_contact[i_foot] = self.pos_foot.translation.transpose()
                    self.memory_contacts[:, i_foot] = mpc_interface.o_feet[0:2, i_foot]
                    self.feetGoal[i_foot].translation = mpc_interface.o_feet[:, i_foot].transpose()
                    self.contacts[i_foot].setReference(self.pos_foot)
        
        return 0

    def solve_HQP_problem(self, t):

        # Resolution of the HQP problem
        HQPData = self.invdyn.computeProblemData(t, self.qtsid, self.vtsid)
        self.sol = self.solver.solve(HQPData)

        # Torques, accelerations, velocities and configuration computation
        self.tau_ff = self.invdyn.getActuatorForces(self.sol)
        self.fc = self.invdyn.getContactForces(self.sol)
        # print(k_simu, " : ", self.fc.transpose())
        # print(self.fc.transpose())
        self.ades = self.invdyn.getAccelerations(self.sol)
        # self.vtsid += self.ades * dt
        # self.qtsid = pin.integrate(self.model, self.qtsid, self.vtsid * dt)

        # Call display and log function
        # self.display(t, solo, k_simu, sequencer)
        # self.log(t, solo, k_simu, sequencer, mpc_interface)

        # Placeholder torques for PyBullet
        # tau = np.zeros((12, 1))

        # Check for NaN value
        if np.any(np.isnan(self.tau_ff)):
            # self.error = True
            self.tau = np.zeros((12, ))
        else:
            # Torque PD controller
            P = 3.0  # 10.0  # 5  # 50
            D = 0.3  # 0.05  # 0.05  #  0.2
            # torques12 = P * (self.qtsid[7:] - qmes12[7:]) + D * (self.vtsid[6:] - vmes12[6:]) + self.tau_ff
            torques12 = self.tau_ff

            # Saturation to limit the maximal torque
            t_max = 2.5
            # faster than np.maximum(a_min, np.minimum(a, a_max))
            self.tau = np.clip(torques12, -t_max, t_max).flatten()

        return 0

    def display(self, t, solo, k_simu, sequencer):

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

    def log(self, t, solo, k_simu, sequencer, mpc_interface):

        self.h_ref_feet[k_simu] = mpc_interface.mean_feet_z

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
            self.f_pos[i_foot, k_simu:(k_simu+1), :] = mpc_interface.o_feet[:,
                                                                            i_foot]  # pos.translation[0:3].transpose()
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

q0 = np.zeros((19, 1))  # initial configuration

omega = 1  # Not used
