# coding: utf8

import crocoddyl
import numpy as np
import quadruped_walkgen as quadruped_walkgen
import pinocchio as pin

class MPC_crocoddyl_planner():
    """Wrapper class for the MPC problem to call the ddp solver and
    retrieve the results.

    Args:
        dt (float): time step of the MPC
        T_mpc (float): Duration of the prediction horizon
        mu (float): Friction coefficient
        inner(bool): Inside or outside approximation of the friction cone
    """

    def __init__(self, params,  mu=1, inner=True, warm_start=True, min_fz=0.0):

        # Time step of the solver
        self.dt = params.dt_mpc

        # Period of the MPC
        self.T_mpc = params.T_mpc

        # Mass of the robot
        self.mass = 2.50000279

        # Inertia matrix of the robot in body frame
        self.gI = np.array([[3.09249e-2, -8.00101e-7, 1.865287e-5],
                            [-8.00101e-7, 5.106100e-2, 1.245813e-4],
                            [1.865287e-5, 1.245813e-4, 6.939757e-2]])

        # Friction coefficient
        if inner:
            self.mu = (1/np.sqrt(2))*mu
        else:
            self.mu = mu

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

        # Weight Vector : Force Norm
        self.forceWeights = np.array(4*[0.01, 0.01, 0.01])

        # Weight Vector : Friction cone cost
        self.frictionWeights = 0.5

        # Weights on the heuristic term 
        self.heuristicWeights = np.array(4*[0.3, 0.4])

        # Weight on the step command (distance between steps)
        self.stepWeights = np.full(8, 0.05)

        # Weights to stop the optimisation at the end of the flying phase
        self.stopWeights = np.full(8, 1.)

        # Weight for shoulder-to-contact penalty
        self.shoulderContactWeight = 5
        self.shoulder_hlim = 0.225

        # Max iteration ddp solver
        self.max_iteration = 10

        # Warm-start for the solver
        # Always on, just an approximation, not the previous result
        # TODO : create a proper warm-start with the previous optimisation
        self.warm_start = warm_start

        # Minimum normal force(N) and reference force vector bool
        self.min_fz = min_fz
        self.relative_forces = True 

        # Gait matrix
        self.gait = np.zeros((params.N_gait, 4))
        self.gait_old = np.zeros((1, 4))

        # Position of the feet in local frame
        self.fsteps = np.full((params.N_gait, 12), 0.0)

        # List to generate the problem
        self.ListAction = []
        self.x_init = []
        self.u_init = []       

        self.l_fsteps = np.zeros((3, 4))
        self.o_fsteps = np.zeros((3, 4))

        # Shooting problem
        self.problem = None

        # ddp solver
        self.ddp = None

        # Xs results without the actionStepModel
        self.Xs = np.zeros((20, int(self.T_mpc/self.dt)))
        # self.Us = np.zeros((12,int(T_mpc/dt)))

        # Initial foot location (local frame, X,Y plan)
        self.p0 = [0.1946, 0.14695, 0.1946, -0.14695, -0.1946,   0.14695, -0.1946,  -0.14695]

        # Index to stop the feet optimisation 
        self.index_lock_time = int(params.lock_time / params.dt_mpc ) # Row index in the gait matrix when the optimisation of the feet should be stopped
        self.index_stop_optimisation = [] # List of index to reset the stopWeights to 0 after optimisation

        self.initializeModels(params)

    def initializeModels(self, params):
        ''' Reset the two lists of augmented and step-by-step models, to avoid recreating them at each loop.
        Not all models here will necessarily be used.  

        Args : 
            - params : object containing the parameters of the simulation
        '''
        self.models_augmented = []
        self.models_step = []

        for j in range(params.N_gait) :
            model = quadruped_walkgen.ActionModelQuadrupedAugmented()            
            
            self.update_model_augmented(model)
            self.models_augmented.append(model)

        for j in range(4 * int(params.T_gait / params.T_mpc) ) :
            model = quadruped_walkgen.ActionModelQuadrupedStep()   
            
            self.update_model_step(model)
            self.models_step.append(model)

        # Terminal node
        self.terminalModel = quadruped_walkgen.ActionModelQuadrupedAugmented() 
        self.update_model_augmented(self.terminalModel)
        # Weights vectors of terminal node (no command cost)
        self.terminalModel.forceWeights = np.zeros(12)
        self.terminalModel.frictionWeights = 0.
        self.terminalModel.heuristicWeights = np.full(8, 0.0)
        self.terminalModel.stopWeights = np.full(8, 0.0)

        return 0

    def solve(self, k, xref, l_feet, l_stop):
        """ Solve the MPC problem

        Args:
            k : Iteration
            xref : the desired state vector
            l_feet : current position of the feet (given by planner)
            l_stop : current and target position of the feet (given by footstepTragectory generator)
        """

        # Update the dynamic depending on the predicted feet position
        self.updateProblem(k, xref, l_feet, l_stop)

        # Solve problem
        self.ddp.solve(self.x_init, self.u_init, self.max_iteration)

        # Reset to 0 the stopWeights for next optimisation
        for index_stopped in self.index_stop_optimisation :
            self.models_augmented[index_stopped].stopWeights = np.zeros(8)

        return 0

    def updateProblem(self, k, xref, l_feet, l_stop):
        """Update the dynamic of the model list according to the predicted position of the feet,
        and the desired state.

        Args:
        """
        # Save previous gait state before updating the gait
        self.gait_old[0, :] = self.gait[0, :].copy()        

        # Recontruct the gait based on the computed footsteps
        a = 0
        while np.any(l_feet[a, :]):
            self.gait[a, :] = (l_feet[a, ::3] != 0.0).astype(int)
            a += 1
        self.gait[a:, :] = 0.0

        # On swing phase before --> initialised below shoulder
        p0 = (1.0 - np.repeat(self.gait[0, :], 2)) * self.p0
        # On the ground before -->  initialised with the current feet position
        p0 += np.repeat(self.gait[0, :], 2) * l_feet[0, [0, 1, 3, 4, 6, 7, 9, 10]]  # (x, y) of each foot       
        
        self.x_init.clear()
        self.u_init.clear()
        self.ListAction.clear()
        self.index_stop_optimisation.clear()

        index_step = 0
        index_augmented = 0   
        j = 0   

        # Iterate over all phases of the gait
        while np.any(self.gait[j, :]):
            if j == 0 : # First row, need to take into account previous gait
                if np.any(self.gait[0,:] - self.gait_old[0,:]) :
                    # Step model
                    self.models_step[index_step].updateModel(np.reshape(l_feet[j, :], (3, 4), order='F'),
                                                       xref[:, j+1], self.gait[0, :] - self.gait_old[0, :])
                    self.ListAction.append(self.models_step[index_step])

                    # Augmented model
                    self.models_augmented[index_augmented].updateModel(np.reshape(l_feet[j, :], (3, 4), order='F'),
                                                    l_stop, xref[:, j+1], self.gait[j, :])                  

                    # Activation of the cost to stop the optimisation around l_stop (position locked by the footstepGenerator)
                    if j < self.index_lock_time :
                        self.models_augmented[index_augmented].stopWeights = self.stopWeights
                        self.index_stop_optimisation.append(index_augmented)
                    
                    self.ListAction.append(self.models_augmented[index_augmented])


                    index_step += 1
                    index_augmented += 1
                    # Warm-start
                    self.x_init.append(np.concatenate([xref[:, j+1], p0]))
                    self.u_init.append(np.zeros(8))
                    self.x_init.append(np.concatenate([xref[:, j+1], p0]))   
                    self.u_init.append(np.repeat(self.gait[j, :], 3) * np.array(4*[0., 0., 2.5*9.81/np.sum(self.gait[j, :])] ))
                
                else : 
                    # Augmented model
                    self.models_augmented[index_augmented].updateModel(np.reshape(l_feet[j, :], (3, 4), order='F'),
                                                     l_stop, xref[:, j+1], self.gait[j, :])
                    self.ListAction.append(self.models_augmented[index_augmented])
                    
                    index_augmented += 1
                    # Warm-start
                    self.x_init.append(np.concatenate([xref[:, j+1], p0]))   
                    self.u_init.append(np.repeat(self.gait[j, :], 3) * np.array(4*[0., 0., 2.5*9.81/np.sum(self.gait[j, :])] ))
            
            else : 
                if np.any(self.gait[j,:] - self.gait[j-1,:]) :
                    # Step model
                    self.models_step[index_step].updateModel(np.reshape(l_feet[j, :], (3, 4), order='F'),
                                                       xref[:, j+1], self.gait[j, :] - self.gait[j-1, :])
                    self.ListAction.append(self.models_step[index_step])

                    # Augmented model
                    self.models_augmented[index_augmented].updateModel(np.reshape(l_feet[j, :], (3, 4), order='F'),
                                                     l_stop, xref[:, j+1], self.gait[j, :])

                    # Activation of the cost to stop the optimisation around l_stop (position locked by the footstepGenerator)
                    if j < self.index_lock_time :
                        self.models_augmented[index_augmented].stopWeights = self.stopWeights
                        self.index_stop_optimisation.append(index_augmented)
                    
                    self.ListAction.append(self.models_augmented[index_augmented])

                    index_step += 1
                    index_augmented += 1
                    # Warm-start
                    self.x_init.append(np.concatenate([xref[:, j+1], p0]))
                    self.u_init.append(np.zeros(8))
                    self.x_init.append(np.concatenate([xref[:, j+1], p0]))   
                    self.u_init.append(np.repeat(self.gait[j, :], 3) * np.array(4*[0., 0., 2.5*9.81/np.sum(self.gait[j, :])] ))
                    
                
                else :
                    self.models_augmented[index_augmented].updateModel(np.reshape(l_feet[j, :], (3, 4), order='F'),
                                                     l_stop, xref[:, j+1], self.gait[j, :])
                    self.ListAction.append(self.models_augmented[index_augmented])
                    
                    index_augmented += 1
                    # Warm-start
                    self.x_init.append(np.concatenate([xref[:, j+1], p0]))   
                    self.u_init.append(np.repeat(self.gait[j, :], 3) * np.array(4*[0., 0., 2.5*9.81/np.sum(self.gait[j, :])] ))

            # Update row matrix
            j += 1

        # Update terminal model 
        self.terminalModel.updateModel(np.reshape(l_feet[j-1, :], (3, 4), order='F'), l_stop, xref[:, -1], self.gait[j-1, :])
        # Warm-start
        self.x_init.append(np.concatenate([xref[:, j-1], p0])) 

        # Shooting problem
        self.problem = crocoddyl.ShootingProblem(np.zeros(20),  self.ListAction, self.terminalModel)

        self.problem.x0 = np.concatenate([xref[:, 0], p0])

        # DDP Solver
        self.ddp = crocoddyl.SolverDDP(self.problem)

        return 0

    def get_latest_result(self):
        """Return the desired contact forces that have been computed by the last iteration of the MPC
        Args:
        """

        cpt = 0
        N = int(self.T_mpc / self.dt)
        result = np.zeros((32, N))
        for i in range(len(self.ListAction)):
            if self.ListAction[i].__class__.__name__ != "ActionModelQuadrupedStep":
                if cpt >= N:
                    raise ValueError("Too many action model considering the current MPC prediction horizon")
                result[:12, cpt] = self.ddp.xs[i][:12]
                result[12:24, cpt] = self.ddp.us[i]  # * np.repeat(self.gait[cpt, :] , 3)
                result[24:, cpt] = self.ddp.xs[i][12:]
                cpt += 1
                if i > 0 and self.ListAction[i-1].__class__.__name__ == "ActionModelQuadrupedStep":
                    # print(self.ddp.xs[i][12:])
                    pass

        return result

    def update_model_augmented(self, model):
        '''Set intern parameters for augmented model type
        '''
        # Model parameters
        model.dt = self.dt
        model.mass = self.mass
        model.gI = self.gI
        model.mu = self.mu
        model.min_fz = self.min_fz
        model.relative_forces = self.relative_forces
        model.shoulderContactWeight = self.shoulderContactWeight

        # Weights vectors
        model.stateWeights = self.stateWeights
        model.forceWeights = self.forceWeights
        model.frictionWeights = self.frictionWeights

        # Weight on feet position
        model.stopWeights = np.full(8, 0.0) # Will be updated when necessary
        model.heuristicWeights = self.heuristicWeights

        return 0

    def update_model_step(self, model):
        """Set intern parameters for step model type
        """
        model.heuristicWeights = np.zeros(8)
        model.stateWeights = self.stateWeights
        model.stepWeights = self.stepWeights

        return 0