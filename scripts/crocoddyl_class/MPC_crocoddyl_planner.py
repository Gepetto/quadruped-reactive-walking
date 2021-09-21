# coding: utf8

import crocoddyl
import numpy as np
import quadruped_walkgen as quadruped_walkgen
import pinocchio as pin


class MPC_crocoddyl_planner():
    """Wrapper class for the MPC problem to call the ddp solver and
    retrieve the results.

    Args:
        params (obj): Params object containing the simulation parameters
        mu (float): Friction coefficient
        inner (float): Inside or outside approximation of the friction cone (mu * 1/sqrt(2))
    """

    def __init__(self, params, mu=1, inner=True):

        self.dt_mpc = params.dt_mpc  # Time step of the solver
        self.dt_wbc = params.dt_wbc  # Time step of the WBC
        self.T_gait = params.T_gait  # Period of the gait
        self.n_nodes = int(params.gait.shape[0])  # Number of nodes
        self.mass = params.mass  # Mass of the robot
        self.max_iteration = 10  # Max iteration ddp solver
        self.gI = np.array(params.I_mat.tolist()).reshape((3, 3))  # Inertia matrix in body frame

        # Friction coefficient
        if inner:
            self.mu = (1 / np.sqrt(2)) * mu
        else:
            self.mu = mu

        # Weights Vector : States
        # self.stateWeights = np.sqrt([2.0, 2.0, 20.0, 0.25, 0.25, 10.0, 0.2, 0.2, 0.2, 0.0, 0.0, 0.3])
        self.w_x = 0.3
        self.w_y = 0.3
        self.w_z = 20
        self.w_roll = 0.9
        self.w_pitch = 1.
        self.w_yaw = 0.9
        self.w_vx = 1.5 * np.sqrt(self.w_x)
        self.w_vy = 2 * np.sqrt(self.w_y)
        self.w_vz = 1 * np.sqrt(self.w_z)
        self.w_vroll = 0.05 * np.sqrt(self.w_roll)
        self.w_vpitch = 0.07 * np.sqrt(self.w_pitch)
        self.w_vyaw = 0.08 * np.sqrt(self.w_yaw)
        self.stateWeights = np.array([
            self.w_x, self.w_y, self.w_z, self.w_roll, self.w_pitch, self.w_yaw, self.w_vx, self.w_vy, self.w_vz,
            self.w_vroll, self.w_vpitch, self.w_vyaw
        ])

        self.forceWeights = 1 * np.array(4 * [0.007, 0.007, 0.007])  # Weight vector : Force Norm
        self.frictionWeights = 1.  # Weight vector : Friction cone penalisation
        self.heuristicWeights = 0. * np.array(4 * [0.03, 0.04])  # Weight vector : Heuristic term
        self.stepWeights = 0. * np.full(8, 0.005)  # Weight vector : Norm of step command (distance between steps)
        self.stopWeights = 0. * np.full(8, 0.005)  # Weight vector : Stop the optimisation (end flying phase)
        self.shoulderContactWeight = 1.5 * np.full(4, 1.)  # Weight vector : Shoulder-to-contact penalisation
        self.shoulder_hlim = 0.235  # Distance to activate the shoulder-to-contact penalisation
        self.shoulderReferencePosition = False  # Use the reference trajectory of the Com (True) or not (False) for shoulder-to-contact penalisation

        # TODO : create a proper warm-start with the previous optimisation
        self.warm_start = True

        # Minimum/Maximal normal force(N) and relative_forces activation
        self.min_fz = 0.
        self.max_fz = 25.
        self.relative_forces = True  # F_ref =  m*g/nb_contact

        # Acceleration penalty
        self.is_acc_activated = False
        self.acc_limit = np.array([60., 60.])  # Acceleration to activate the penalisation
        self.acc_weight = 0.0001
        self.n_sampling = 8
        self.flying_foot_old = 4 * [False]
        self.dx_new_phase = 0.
        self.dy_new_phase = 0.

        # Velocity penalty
        self.is_vel_activated = True
        self.vel_limit = np.array([5., 5.])  # Velocity to activate the penalisation
        self.vel_weight = 1.

        # Jerk penalty
        self.is_jerk_activated = True
        self.jerk_weight = 1e-7
        self.jerk_alpha = 21  # Define the slope of the cost, not used
        self.jerk_beta = 0.  # Define the slope of the cost, not used

        # Offset CoM
        self.offset_com = -0.03
        self.vert_time = params.vert_time

        # Gait matrix
        self.gait = np.zeros((self.n_nodes, 4))
        self.gait_old = np.zeros(4)

        # List to generate the problem
        self.action_models = []
        self.x_init = []
        self.u_init = []

        self.problem = None  # Shooting problem
        self.ddp = None  # ddp solver
        self.Xs = np.zeros((20, int(self.n_nodes / self.dt_mpc)))  # Xs results without the actionStepModel

        # Initial foot location (local frame, X,Y plan)
        self.shoulders = [0.1946, 0.14695, 0.1946, -0.14695, -0.1946, 0.14695, -0.1946, -0.14695]
        self.xref = np.full((12, self.n_nodes + 1), np.nan)

        # Index to stop the feet optimisation
        # Row index in the gait matrix when the optimisation of the feet should be stopped
        self.index_lock_time = int(params.lock_time / params.dt_mpc)
        self.index_stop_optimisation = []  # List of index to reset the stopWeights to 0 after optimisation

        # Usefull to optimise around the previous optimisation
        self.flying_foot = 4 * [False]  # Bool corresponding to the current flying foot (gait[0,foot_id] == 0)
        self.flying_foot_nodes = np.zeros(4)  # The number of nodes in the next phase of flight
        self.flying_max_nodes = int(
            params.T_gait / (2 * params.dt_mpc))  # TODO : get the maximum number of nodes from the gait_planner

        # Usefull to setup the shoulder-to-contact cost (no cost for the initial stance phase)
        self.stance_foot = 4 * [False]  # Bool corresponding to the current flying foot (gait[0,foot_id] == 0)
        self.stance_foot_nodes = np.zeros(4)  # The number of nodes in the next phase of flight
        self.index_inital_stance_phase = []  # List of index to reset the stopWeights to 0 after optimisation

        # Initialize the lists of models
        self.initialize_models()

    def initialize_models(self):
        ''' Reset the two lists of augmented and step-by-step models, to avoid recreating them at each loop.
        Not all models here will necessarily be used.
        '''
        self.models_augmented = []
        self.models_step = []

        for _ in range(self.n_nodes):
            model = quadruped_walkgen.ActionModelQuadrupedAugmented()
            self.update_model_augmented(model)
            self.models_augmented.append(model)

        n_period = int(self.dt_mpc * self.n_nodes / self.T_gait)
        for _ in range(4 * n_period):
            model = quadruped_walkgen.ActionModelQuadrupedStep()
            self.update_model_step(model)
            self.models_step.append(model)

        # Terminal node
        self.terminal_model = quadruped_walkgen.ActionModelQuadrupedAugmented()
        self.update_model_augmented(self.terminal_model, terminal=True)

    def solve(self, k, xref, footsteps, l_stop, position, velocity, acceleration, jerk, oRh, oTh, dt_flying):
        """ Solve the MPC problem.

        Args:
            k (int) : Iteration
            xref (Array 12xN)       : Desired state vector
            footsteps (Array 12xN)  : current position of the feet (given by planner)
            l_stop (Array 3x4)      : Current target position of the feet
            position (Array 3x4)    : Position of the flying feet
            velocity (Array 3x4)    : Velocity of the flying feet
            acceleration (Array 3x4): Acceleration position of the flying feet
            jerk (Array 3x4)        : Jerk position of the flying feet
            oRh (Array 3x3)         : Rotation matrix from base to ideal/world frame
            oTh (Array 3x1)         : Translation vector from base to ideal/world frame
            dt_flying (Array 4x)    : Remaining timing of flying feet
        """
        self.xref[:, :] = xref
        self.xref[2, :] += self.offset_com

        self.updateProblem(k, self.xref, footsteps, l_stop, position, velocity, acceleration, jerk, oRh, oTh,
                           dt_flying)
        self.ddp.solve(self.x_init, self.u_init, self.max_iteration)

        # Reset to 0 the stopWeights for next optimisation (already 0.)
        for index_stopped in self.index_stop_optimisation:
            self.models_augmented[index_stopped].stopWeights = np.zeros(8)

        for index_stance in self.index_inital_stance_phase:
            self.models_augmented[index_stance].shoulderContactWeight = self.shoulderContactWeight

    def updateProblem(self, k, xref, footsteps, l_stop, position, velocity, acceleration, jerk, oRh, oTh, dt_flying):
        """
        Update the models of the nodes according to parameters of the simulations.

        Args:
            k (int) : Iteration
            xref (Array 12xN)       : Desired state vector
            footsteps (Array 12xN)  : current position of the feet (given by planner)
            l_stop (Array 3x4)      : Current target position of the feet
            position (Array 3x4)    : Position of the flying feet
            velocity (Array 3x4)    : Velocity of the flying feet
            acceleration (Array 3x4): Acceleration position of the flying feet
            jerk (Array 3x4)        : Jerk position of the flying feet
            oRh (Array 3x3)         : Rotation matrix from base to ideal/world frame
            oTh (Array 3x1)         : Translation vector from base to ideal/world frame
            dt_flying (Array 4x)    : Remaining timing of flying feet
        """
        # Update internal matrix
        self.compute_gait_matrix(footsteps)

        # Set up intial feet position : shoulder for flying feet and current foot position for feet on the ground
        p0 = (1.0 - np.repeat(self.gait[0, :], 2)) * self.shoulders \
            + np.repeat(self.gait[0, :], 2) * footsteps[0, [0, 1, 3, 4, 6, 7, 9, 10]]

        # Clear lists of models
        self.x_init.clear()
        self.u_init.clear()
        self.action_models.clear()
        self.index_stop_optimisation.clear()
        self.index_inital_stance_phase.clear()

        index_step = 0
        index_augmented = 0
        j = 1

        stopping_needed = 4 * [False]
        if k > 110:
            for index_foot, is_flying in enumerate(self.flying_foot):
                if is_flying:
                    # Position optimized at the previous control cycle
                    stopping_needed[index_foot] = self.flying_foot_nodes[index_foot] != self.flying_max_nodes

        # Augmented model, first node, j = 0
        self.models_augmented[index_augmented].updateModel(np.reshape(footsteps[0, :], (3, 4), order='F'), l_stop,
                                                           xref[:, 0], self.gait[0, :])

        shoulder_weight = self.shoulderContactWeight.copy()
        for foot in range(4):
            if self.stance_foot[foot] and j < self.stance_foot_nodes[foot]:
                shoulder_weight[foot] = 0.

        if np.sum(shoulder_weight != self.shoulderContactWeight) > 0:
            # The shoulder-to-contact weight has been modified, needs to be added to the list
            self.models_augmented[index_augmented].shoulderContactWeight = shoulder_weight
            self.index_inital_stance_phase.append(index_augmented)

        self.action_models.append(self.models_augmented[index_augmented])
        index_augmented += 1

        # Warm-start
        self.x_init.append(np.concatenate([xref[:, 0], p0]))
        self.u_init.append(
            np.repeat(self.gait[0, :], 3) * np.array(4 * [0., 0., 2.5 * 9.81 / np.sum(self.gait[0, :])]))

        # Bool to activate the speed, acc or jerk on only current flying feet
        is_first_step = True

        while j < self.n_nodes:
            # TODO Adapt this [1,1,1,1] --> [1,0,0,1] True here but not step model to add
            if np.any(self.gait[j, :] - self.gait[j - 1, :]):

                flying_feet = np.where(self.gait[j - 1, :] == 0)[0]  # Current flying feet

                # Get remaining dt incurrent flying phase
                # TODO : dt_ is the same for all flying foot, could be different
                dt_ = dt_flying[flying_feet[0]]

                # part_curve explanation for foot trajectory :
                #           ------- <- 2 (foot stopped, during params.vert_time)
                #         /
                #        /          <- 1 (foot moving)
                #       /
                # ------            <- 0 (foot stopped, during params.vert_time)
                # TODO : get total flying period from gait and not T_gait/2
                part_curve = 0
                dt_ -= self.dt_wbc  # dt_ has been computed on the previous loop, dt_wbc late
                if dt_ <= 0.001 or dt_ >= self.T_gait / 2 - self.vert_time:  # Because of the delay, dt_ = 0 --> new flying phase
                    dt_ = self.T_gait / 2 - 2 * self.vert_time
                    part_curve = 0
                elif dt_ >= self.vert_time:
                    dt_ -= self.vert_time
                    part_curve = 1
                else:
                    dt_ = 0.002  # Close to 0 to avoid high jerk and thus keep the optimised fstep from previous cycle
                    part_curve = 2

                if is_first_step:
                    #  Acceleration cost
                    # if self.flying_foot_old != self.flying_foot :
                    #     self.dx_new_phase = abs(footsteps[j, 3*flying_feet[0] ] - position[0,flying_feet[0] ])
                    #     self.dy_new_phase = abs(footsteps[j, 3*flying_feet[0]+1 ] - position[1,flying_feet[0] ])
                    #     self.acc_limit = 2*5.625*np.array([ self.dx_new_phase / 0.24**2  , self.dy_new_phase / 0.24**2])
                    self.models_step[index_step].is_acc_activated = self.is_acc_activated
                    # self.models_step[index_step].acc_limit =  self.acc_limit

                    # Velocity cost
                    self.models_step[index_step].is_vel_activated = self.is_vel_activated

                    if part_curve == 1 or part_curve == 2:
                        self.models_step[index_step].is_jerk_activated = self.is_jerk_activated
                        self.models_step[index_step].jerk_weight = self.exponential_cost(self.T_gait / 2 - dt_)
                        # self.models_step[index_step].jerk_weight = self.jerk_weight
                    else:
                        self.models_step[index_step].jerk_weight = 0.

                    # Update is_first_step bool
                    is_first_step = False

                else:
                    # Cost on the feet trajectory disabled
                    # TODO : Only activate cost on the maximum speed (simplify the equations in step model, vmax = T/2)
                    self.models_step[index_step].is_acc_activated = False
                    self.models_step[index_step].is_vel_activated = False
                    self.models_step[index_step].is_jerk_activated = False
                    dt_ = 0.24

                self.models_step[index_step].updateModel(np.reshape(footsteps[j, :], (3, 4), order='F'), xref[:, j],
                                                         self.gait[j, :] - self.gait[j - 1, :], position, velocity,
                                                         acceleration, jerk, oRh, oTh, dt_)
                self.action_models.append(self.models_step[index_step])

                # Augmented model
                self.models_augmented[index_augmented].updateModel(np.reshape(footsteps[j, :], (3, 4), order='F'),
                                                                   l_stop, xref[:, j], self.gait[j, :])

                # Activation of the cost to stop the optimisation around l_stop (position locked by the footstepGenerator)
                # if j < self.index_lock_time:
                #     self.models_augmented[index_augmented].stopWeights = self.stopWeights
                #     self.index_stop_optimisation.append(index_augmented)

                feet_ground = np.where(self.gait[j, :] == 1)[0]
                activation_cost = False
                for foot in range(4):
                    if (stopping_needed[foot]) and (self.flying_foot[foot]) and (foot in feet_ground) and (
                            j < int(self.flying_foot_nodes[foot] + self.flying_max_nodes)):
                        coeff_activated = np.zeros(8)
                        coeff_activated[2 * foot:2 * foot + 2] = np.array([1, 1])
                        self.models_augmented[index_augmented].stopWeights = self.models_augmented[
                            index_augmented].stopWeights + coeff_activated * self.stopWeights
                        activation_cost = True

                if activation_cost:
                    self.index_stop_optimisation.append(index_augmented)

                shoulder_weight = self.shoulderContactWeight.copy()
                for foot in range(4):
                    if self.stance_foot[foot] and j < self.stance_foot_nodes[foot]:
                        shoulder_weight[foot] = 0.
                if np.sum(shoulder_weight != self.shoulderContactWeight) > 0:
                    # The shoulder-to-contact weight has been modified, needs to be added to the list
                    self.models_augmented[index_augmented].shoulderContactWeight = shoulder_weight
                    self.index_inital_stance_phase.append(index_augmented)

                self.action_models.append(self.models_augmented[index_augmented])

                index_step += 1
                index_augmented += 1
                # Warm-start
                self.x_init.append(np.concatenate([xref[:, j], p0]))
                self.u_init.append(np.zeros(8))
                self.x_init.append(np.concatenate([xref[:, j], p0]))
                self.u_init.append(
                    np.repeat(self.gait[j, :], 3) * np.array(4 * [0., 0., 2.5 * 9.81 / np.sum(self.gait[j, :])]))

            else:
                self.models_augmented[index_augmented].updateModel(np.reshape(footsteps[j, :], (3, 4), order='F'),
                                                                   l_stop, xref[:, j], self.gait[j, :])
                self.action_models.append(self.models_augmented[index_augmented])

                feet_ground = np.where(self.gait[j, :] == 1)[0]
                activation_cost = False
                for foot in range(4):
                    if (stopping_needed[foot]) and (self.flying_foot[foot]) and (foot in feet_ground) and (
                            j < int(self.flying_foot_nodes[foot] + self.flying_max_nodes)):
                        coeff_activated = np.zeros(8)
                        coeff_activated[2 * foot:2 * foot + 2] = np.array([1, 1])
                        self.models_augmented[index_augmented].stopWeights = self.models_augmented[
                            index_augmented].stopWeights + coeff_activated * self.stopWeights
                        activation_cost = True

                if activation_cost:
                    self.index_stop_optimisation.append(index_augmented)

                shoulder_weight = self.shoulderContactWeight.copy()
                for foot in range(4):
                    if self.stance_foot[foot] and j < self.stance_foot_nodes[foot]:
                        shoulder_weight[foot] = 0.
                if np.sum(shoulder_weight != self.shoulderContactWeight) > 0:
                    # The shoulder-to-contact weight has been modified, needs to be added to the list
                    self.models_augmented[index_augmented].shoulderContactWeight = shoulder_weight
                    self.index_inital_stance_phase.append(index_augmented)

                index_augmented += 1
                # Warm-start
                self.x_init.append(np.concatenate([xref[:, j], p0]))
                self.u_init.append(
                    np.repeat(self.gait[j, :], 3) * np.array(4 * [0., 0., 2.5 * 9.81 / np.sum(self.gait[j, :])]))

            # Update row matrix
            j += 1

        # Update terminal model
        self.terminal_model.updateModel(np.reshape(footsteps[j - 1, :], (3, 4), order='F'), l_stop, xref[:, -1],
                                        self.gait[j - 1, :])
        # Warm-start
        self.x_init.append(np.concatenate([xref[:, -1], p0]))

        self.problem = crocoddyl.ShootingProblem(np.zeros(20), self.action_models, self.terminal_model)
        self.problem.x0 = np.concatenate([xref[:, 0], p0])

        self.ddp = crocoddyl.SolverDDP(self.problem)

    def get_latest_result(self, oRh, oTh):
        """ 
        Return the desired contact forces that have been computed by the last iteration of the MPC
        Args : 
         - q ( Array 7x1 ) : pos, quaternion orientation
        """
        index = 0
        result = np.zeros((32, self.n_nodes))
        for i in range(len(self.action_models)):
            if self.action_models[i].__class__.__name__ != "ActionModelQuadrupedStep":
                if index >= self.n_nodes:
                    raise ValueError("Too many action model considering the current MPC prediction horizon")
                result[:12, index] = self.ddp.xs[i + 1][:12]  # First node correspond to current state
                result[12:24, index] = self.ddp.us[i]
                result[24:, index] = (oRh[:2, :2] @ (self.ddp.xs[i + 1][12:].reshape(
                    (2, 4), order="F")) + oTh[:2]).reshape((8), order="F")
                if i > 0 and self.action_models[i - 1].__class__.__name__ == "ActionModelQuadrupedStep":
                    pass
                index += 1

        return result

    def update_model_augmented(self, model, terminal=False):
        """ 
        Set intern parameters for augmented model type
        """
        # Model parameters
        model.dt = self.dt_mpc
        model.mass = self.mass
        model.gI = self.gI
        model.mu = self.mu
        model.min_fz = self.min_fz
        model.max_fz = self.max_fz
        model.relative_forces = True
        model.shoulderContactWeight = self.shoulderContactWeight
        model.shoulder_hlim = self.shoulder_hlim
        model.shoulderReferencePosition = self.shoulderReferencePosition
        # print(model.referenceShoulderPosition)

        # Weights vectors
        model.stateWeights = self.stateWeights
        model.stopWeights = np.zeros(8)

        if terminal:
            self.terminal_model.forceWeights = np.zeros(12)
            self.terminal_model.frictionWeights = 0.
            self.terminal_model.heuristicWeights = np.zeros(8)
        else:
            model.frictionWeights = self.frictionWeights
            model.forceWeights = self.forceWeights
            model.heuristicWeights = self.heuristicWeights

    def update_model_step(self, model):
        """
        Set intern parameters for step model type
        """
        model.heuristicWeights = np.zeros(8)
        model.stateWeights = self.stateWeights
        model.stepWeights = self.stepWeights
        model.acc_limit = self.acc_limit
        model.acc_weight = self.acc_weight
        model.is_acc_activated = False
        model.vel_limit = self.vel_limit
        model.vel_weight = self.vel_weight
        model.is_vel_activated = False
        model.is_jerk_activated = False
        model.jerk_weight = self.jerk_weight
        model.set_sample_feet_traj(self.n_sampling)

    def compute_gait_matrix(self, footsteps):
        """ 
        Recontruct the gait based on the computed footstepsC
        Args:
            footsteps : current and predicted position of the feet
        """

        self.gait_old = self.gait[0, :].copy()

        j = 0
        self.gait = np.zeros(np.shape(self.gait))
        while j < self.n_nodes:
            self.gait[j, :] = (footsteps[j, ::3] != 0.0).astype(int)
            j += 1

        # Get the current flying feet and the number of nodes
        self.flying_foot_old[:] = self.flying_foot[:]  # Update old matrix, usefull when new phase start
        for foot in range(4):
            row = 0
            if self.gait[0, foot] == 0:
                self.flying_foot[foot] = True
                while self.gait[row, foot] == 0:
                    row += 1
                self.flying_foot_nodes[foot] = int(row)
            else:
                self.flying_foot[foot] = False

        # Get the current stance feet and the number of nodes
        for foot in range(4):
            row = 0
            if self.gait[0, foot] == 1:
                self.stance_foot[foot] = True
                while self.gait[row, foot] == 1:
                    row += 1
                self.stance_foot_nodes[foot] = int(row)
            else:
                self.stance_foot[foot] = False

    def exponential_cost(self, t):
        """
        Returns weight_jerk * exp(alpha(t-beta)) - 1 
        Args:
            t (float) : time in s 
        """

        return self.jerk_weight * (np.exp(self.jerk_alpha * (t - self.jerk_beta)) - 1)

if __name__ == "__main__":
    """ Plot the exponential cost used to penalize the jerk difference withe the
    previous control cycle
    """

    import matplotlib.pyplot as plt
    import libquadruped_reactive_walking as lqrw

    params = lqrw.Params()
    mpc = MPC_crocoddyl_planner(params)
    mpc.jerk_alpha = 20  # Define the slope of the cost, not used
    mpc.jerk_beta = 0.  # Define the slope of the cost, not used

    plt.figure()
    n_points = 100
    T = np.linspace(1e-6, 0.23, n_points)
    X = [mpc.exponential_cost(0.24 - t) for t in T]
    plt.plot(T,X)
    plt.show(block = True)


