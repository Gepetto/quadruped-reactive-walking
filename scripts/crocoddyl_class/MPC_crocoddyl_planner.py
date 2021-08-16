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
        linearModel(bool) : Approximation in the cross product by using desired state
    """

    def __init__(self, params,  mu=1, inner=True, warm_start=True, min_fz=0.0):

        self.dt = params.dt_mpc                   # Time step of the solver        
        self.T_mpc = params.T_mpc                 # Period of the MPC    
        self.n_nodes = int(self.T_mpc/self.dt)    # Number of nodes    
        self.mass = params.mass                   # Mass of the robot
        self.max_iteration = 20                   # Max iteration ddp solver
        self.gI = np.array(params.I_mat.tolist()).reshape((3, 3)) # Inertia matrix in ody frame

        # Friction coefficient
        if inner:
            self.mu = (1/np.sqrt(2))*mu
        else:
            self.mu = mu

        # self.stateWeight = np.sqrt([2.0, 2.0, 20.0, 0.25, 0.25, 0.25, 0.2, 0.2, 5., 0.0, 0.0, 0.3]) 

        # Weights Vector : States
        self.w_x = 0.3
        self.w_y = 0.3
        self.w_z = 2
        self.w_roll = 0.9
        self.w_pitch = 1.
        self.w_yaw = 0.4
        self.w_vx = 1.5*np.sqrt(self.w_x)
        self.w_vy = 2*np.sqrt(self.w_y)
        self.w_vz = 1*np.sqrt(self.w_z)
        self.w_vroll = 0.05*np.sqrt(self.w_roll)
        self.w_vpitch = 0.07*np.sqrt(self.w_pitch)
        self.w_vyaw = 0.05*np.sqrt(self.w_yaw)
        self.stateWeights = np.array([self.w_x, self.w_y, self.w_z, self.w_roll, self.w_pitch, self.w_yaw,
                                      self.w_vx, self.w_vy, self.w_vz, self.w_vroll, self.w_vpitch, self.w_vyaw])

        self.forceWeights = 2*np.array(4*[0.01, 0.01, 0.01])  # Weight Vector : Force Norm
        self.frictionWeights = 1.                          # Weight Vector : Friction cone cost
        self.heuristicWeights = np.array(4*[0.3, 0.4])      # Weights on the heuristic term
        self.stepWeights = np.full(8, 0.005)                 # Weight on the step command (distance between steps)
        self.stopWeights = 2.5*np.ones(8)                       # Weights to stop the optimisation at the end of the flying phase
        self.shoulderContactWeight = 5.                      # Weight for shoulder-to-contact penalty
        self.shoulder_hlim = 0.235

        # TODO : create a proper warm-start with the previous optimisation
        self.warm_start = True

        # Minimum normal force(N) and reference force vector bool
        self.min_fz = 1.
        self.relative_forces = True 

        # Gait matrix
        self.gait = np.zeros((params.N_gait, 4))
        self.gait_old = np.zeros(4)

        # Position of the feet in local frame
        self.fsteps = np.zeros((params.N_gait, 12))

        # List to generate the problem
        self.action_models = []
        self.x_init = []
        self.u_init = []

        self.problem = None   # Shooting problem
        self.ddp = None  # ddp solver
        self.Xs = np.zeros((20, int(self.T_mpc/self.dt)))  # Xs results without the actionStepModel

        # Initial foot location (local frame, X,Y plan)
        self.shoulders = [0.1946, 0.14695, 0.1946, -0.14695, -0.1946,   0.14695, -0.1946,  -0.14695]

        # Index to stop the feet optimisation
        self.index_lock_time = int(params.lock_time / params.dt_mpc)  # Row index in the gait matrix when the optimisation of the feet should be stopped
        self.index_stop_optimisation = []  # List of index to reset the stopWeights to 0 after optimisation

        # USefull to optimise around the previous optimisation
        self.flying_foot = 4*[False] # Bool corresponding to the current flying foot (gait[0,foot_id] == 0)
        self.flying_foot_nodes = np.zeros(4) # The number of nodes in the next phase of flight
        self.flying_max_nodes = int(params.T_mpc / (2 * params.dt_mpc)) # TODO : get the maximum number of nodes from the gait_planner

        # Initialize the lists of models
        self.initialize_models(params)

    def initialize_models(self, params):
        ''' Reset the two lists of augmented and step-by-step models, to avoid recreating them at each loop.
        Not all models here will necessarily be used.  

        Args : 
            - params : object containing the parameters of the simulation
        '''
        self.models_augmented = []
        self.models_step = []

        for _ in range(params.N_gait):
            model = quadruped_walkgen.ActionModelQuadrupedAugmented()
            self.update_model_augmented(model)
            self.models_augmented.append(model)

        for _ in range(4 * int(params.T_gait / params.T_mpc)):
            model = quadruped_walkgen.ActionModelQuadrupedStep()
            self.update_model_step(model)
            self.models_step.append(model)

        # Terminal node
        self.terminal_model = quadruped_walkgen.ActionModelQuadrupedAugmented()
        self.update_model_augmented(self.terminal_model, terminal=True)

    def solve(self, k, xref, footsteps, l_stop):
        """ Solve the MPC problem

        Args:
            k : Iteration
            xref : the desired state vector
            footsteps : current position of the feet (given by planner)
            l_stop : current and target position of the feet (given by footstepTragectory generator)
        """
        self.updateProblem(k, xref, footsteps, l_stop)
        self.ddp.solve(self.x_init, self.u_init, self.max_iteration)

        # Reset to 0 the stopWeights for next optimisation
        for index_stopped in self.index_stop_optimisation:
            self.models_augmented[index_stopped].stopWeights = np.zeros(8)

    def updateProblem(self, k, xref, footsteps, l_stop):
        """
        Update the dynamic of the model list according to the predicted position of the feet,
        and the desired state.

        Args:
        """
        self.compute_gait_matrix(footsteps)

        p0 = (1.0 - np.repeat(self.gait[0, :], 2)) * self.shoulders \
            + np.repeat(self.gait[0, :], 2) * footsteps[0, [0, 1, 3, 4, 6, 7, 9, 10]]

        self.x_init.clear()
        self.u_init.clear()
        self.action_models.clear()
        self.index_stop_optimisation.clear()

        index_step = 0
        index_augmented = 0
        j = 1
        
        stopping_needed = 4*[False]
        if k > 110 :
            for index_foot, is_flying in enumerate(self.flying_foot):
                if is_flying:
                    stopping_needed[index_foot] = self.flying_foot_nodes[index_foot] != self.flying_max_nodes # Position optimized at the previous control cycle

        # Augmented model, first node, j = 0
        self.models_augmented[index_augmented].updateModel(np.reshape(footsteps[0, :], (3, 4), order='F'),
                                                            l_stop, xref[:, 1], self.gait[0, :])
        self.action_models.append(self.models_augmented[index_augmented])

        index_augmented += 1
        # Warm-start
        self.x_init.append(np.concatenate([xref[:, 1], p0]))
        self.u_init.append(np.repeat(self.gait[0, :], 3) * np.array(4*[0., 0., 2.5*9.81/np.sum(self.gait[0, :])]))

        while np.any(self.gait[j, :]):            
            if np.any(self.gait[j, :] - self.gait[j-1, :]):
                # Step model
                self.models_step[index_step].updateModel(np.reshape(footsteps[j, :], (3, 4), order='F'),
                                                            xref[:, j+1], self.gait[j, :] - self.gait[j-1, :])
                self.action_models.append(self.models_step[index_step])

                # Augmented model
                self.models_augmented[index_augmented].updateModel(np.reshape(footsteps[j, :], (3, 4), order='F'),
                                                                    l_stop, xref[:, j+1], self.gait[j, :])

                # Activation of the cost to stop the optimisation around l_stop (position locked by the footstepGenerator)
                # if j < self.index_lock_time:
                #     self.models_augmented[index_augmented].stopWeights = self.stopWeights
                #     self.index_stop_optimisation.append(index_augmented)

                feet_ground = np.where(self.gait[j,:] == 1)[0]
                activation_cost = False
                for foot in range(4): 
                    if (stopping_needed[foot]) and (self.flying_foot[foot]) and (foot in feet_ground) and (j < int(self.flying_foot_nodes[foot] + self.flying_max_nodes) ) :
                        coeff_activated = np.zeros(8)
                        coeff_activated[2*foot: 2*foot + 2] = np.array([1,1])
                        self.models_augmented[index_augmented].stopWeights = self.models_augmented[index_augmented].stopWeights + coeff_activated*self.stopWeights
                        activation_cost = True
                
                if activation_cost :
                    self.index_stop_optimisation.append(index_augmented)


                self.action_models.append(self.models_augmented[index_augmented])

                index_step += 1
                index_augmented += 1
                # Warm-start
                self.x_init.append(np.concatenate([xref[:, j+1], p0]))
                self.u_init.append(np.zeros(8))
                self.x_init.append(np.concatenate([xref[:, j+1], p0]))
                self.u_init.append(np.repeat(self.gait[j, :], 3) * np.array(4*[0., 0., 2.5*9.81/np.sum(self.gait[j, :])]))

            else:
                self.models_augmented[index_augmented].updateModel(np.reshape(footsteps[j, :], (3, 4), order='F'),
                                                                    l_stop, xref[:, j+1], self.gait[j, :])
                self.action_models.append(self.models_augmented[index_augmented])

                feet_ground = np.where(self.gait[j,:] == 1)[0]
                activation_cost = False
                for foot in range(4): 
                    if (stopping_needed[foot]) and (self.flying_foot[foot]) and (foot in feet_ground) and (j < int(self.flying_foot_nodes[foot] + self.flying_max_nodes) ) :
                        coeff_activated = np.zeros(8)
                        coeff_activated[2*foot: 2*foot + 2] = np.array([1,1])
                        self.models_augmented[index_augmented].stopWeights = self.models_augmented[index_augmented].stopWeights + coeff_activated*self.stopWeights
                        activation_cost = True
                
                if activation_cost :
                    self.index_stop_optimisation.append(index_augmented)

                index_augmented += 1
                # Warm-start
                self.x_init.append(np.concatenate([xref[:, j+1], p0]))
                self.u_init.append(np.repeat(self.gait[j, :], 3) * np.array(4*[0., 0., 2.5*9.81/np.sum(self.gait[j, :])]))

            # Update row matrix
            j += 1

        # Update terminal model
        self.terminal_model.updateModel(np.reshape(footsteps[j-1, :], (3, 4), order='F'), l_stop, xref[:, -1], self.gait[j-1, :])
        # Warm-start
        self.x_init.append(np.concatenate([xref[:, j-1], p0]))

        self.problem = crocoddyl.ShootingProblem(np.zeros(20),  self.action_models, self.terminal_model)
        self.problem.x0 = np.concatenate([xref[:, 0], p0])

        self.ddp = crocoddyl.SolverDDP(self.problem)

    def get_latest_result(self, oRh, oTh):
        """ 
        Return the desired contact forces that have been computed by the last iteration of the MPC
        Args : 
         - q ( Array 7x1 ) : pos, quaternion orientation
        """
        index = 0
        N = int(self.T_mpc / self.dt)
        result = np.zeros((32, N))
        for i in range(len(self.action_models)):
            if self.action_models[i].__class__.__name__ != "ActionModelQuadrupedStep":
                if index >= N:
                    raise ValueError("Too many action model considering the current MPC prediction horizon")
                result[:12, index] = self.ddp.xs[i+1][:12] # First node correspond to current state
                result[12:24, index] = self.ddp.us[i]
                result[24:, index] = ( oRh[:2,:2] @ (self.ddp.xs[i+1][12:].reshape((2,4) , order = "F")  ) + oTh[:2]).reshape((8), order = "F")
                if i > 0 and self.action_models[i-1].__class__.__name__ == "ActionModelQuadrupedStep":
                    pass
                index += 1

        return result

    def update_model_augmented(self, model, terminal=False):
        """ 
        Set intern parameters for augmented model type
        """
        # Model parameters
        model.dt = self.dt
        model.mass = self.mass
        model.gI = self.gI
        model.mu = self.mu
        model.min_fz = self.min_fz
        model.relative_forces = True
        model.shoulderContactWeight = self.shoulderContactWeight
        model.shoulder_hlim = self.shoulder_hlim 

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

    def compute_gait_matrix(self, footsteps):
        """ 
        Recontruct the gait based on the computed footstepsC
        Args:
            footsteps : current and predicted position of the feet
        """

        self.gait_old = self.gait[0, :].copy()

        j = 0
        self.gait = np.zeros(np.shape(self.gait))
        while np.any(footsteps[j, :]):
            self.gait[j, :] = (footsteps[j, ::3] != 0.0).astype(int)
            j += 1

        # Get the current flying feet and the number of nodes
        for foot in range(4):
            row = 0
            if self.gait[0, foot] == 0:
                self.flying_foot[foot] = True 
                while self.gait[row, foot] == 0:
                    row += 1
                self.flying_foot_nodes[foot] = int(row)
            else:
                self.flying_foot[foot] = False
    

