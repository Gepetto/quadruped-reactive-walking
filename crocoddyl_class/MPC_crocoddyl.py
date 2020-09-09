# coding: utf8

import crocoddyl
import numpy as np
import quadruped_walkgen 

class MPC_crocoddyl:
    """Wrapper class for the MPC problem to call the ddp solver and 
    retrieve the results. 

    Args:
        dt (float): time step of the MPC
        T_mpc (float): Duration of the prediction horizon
        mu (float): Friction coefficient
        inner(bool): Inside or outside approximation of the friction cone
        linearModel(bool) : Approximation in the cross product by using desired state
    """

    def __init__(self, dt = 0.02 , T_mpc = 0.32 ,  mu = 1, inner = True , linearModel = True , n_period = 1):

        # Time step of the solver
        self.dt = dt

        # Period of the MPC
        self.T_mpc = T_mpc

        # Mass of the robot 
        self.mass = 2.50000279 

        # Inertia matrix of the robot in body frame 
        # self.gI = np.diag([0.00578574, 0.01938108, 0.02476124])
        self.gI = np.array([[3.09249e-2, -8.00101e-7, 1.865287e-5],
                        [-8.00101e-7, 5.106100e-2, 1.245813e-4],
                        [1.865287e-5, 1.245813e-4, 6.939757e-2]]) 

        # Friction coefficient
        if inner :
            self.mu = (1/np.sqrt(2))*mu
        else:
            self.mu = mu

        # Gain from OSQP MPC
        self.w_x = np.sqrt(0.5)
        self.w_y = np.sqrt(0.5)
        self.w_z = np.sqrt(2.)
        self.w_roll = np.sqrt(0.11)
        self.w_pitch = np.sqrt(0.11)
        self.w_yaw = np.sqrt(0.11)
        self.w_vx =  np.sqrt(2.*np.sqrt(0.5))
        self.w_vy =  np.sqrt(2.*np.sqrt(0.5))
        self.w_vz =  np.sqrt(2.*np.sqrt(2.))
        self.w_vroll =  np.sqrt(0.05*np.sqrt(0.11))
        self.w_vpitch =  np.sqrt(0.05*np.sqrt(0.11))
        self.w_vyaw =  np.sqrt(0.05*np.sqrt(0.11))

        self.stateWeight = np.array([self.w_x,self.w_y,self.w_z,self.w_roll,self.w_pitch,self.w_yaw,
                                    self.w_vx,self.w_vy,self.w_vz,self.w_vroll,self.w_vpitch,self.w_vyaw])


        # Weight Vector : Force Norm
        self.forceWeights = np.array(4*[0.01,0.01,0.01])

        # Weight Vector : Friction cone cost
        self.frictionWeights = 1.0

        # Max iteration ddp solver
        self.max_iteration = 10

        # Warm Start for the solver
        self.warm_start =  True

        # Minimum normal force (N)
        self.min_fz = 0.2
        self.max_fz = 25

        # Gait matrix
        self.gait = np.zeros((20, 5))
        self.index = 0

        # Weight on the shoulder term : 
        self.shoulderWeights = 10.
        self.shoulder_hlim = 0.22

        # Position of the feet
        self.fsteps = np.full((20, 13), np.nan)

        # List of the actionModel
        self.ListAction = [] 

        # Initialisation of the List model using ActionQuadrupedModel()  
        # The same model cannot be used [model]*(T_mpc/dt) because the dynamic
        # model changes for each nodes.  
        for i in range(int(self.T_mpc/self.dt )*n_period):
            if linearModel :
                model = quadruped_walkgen.ActionModelQuadruped()   
            else : 
                model = quadruped_walkgen.ActionModelQuadrupedNonLinear()   

            # Model parameters    
            model.dt = self.dt 
            model.mass = self.mass      
            model.gI = self.gI 
            model.mu = self.mu
            model.min_fz = self.min_fz
            model.max_fz = self.max_fz

            # Weights vectors
            model.stateWeights = self.stateWeight
            model.forceWeights = self.forceWeights
            model.frictionWeights = self.frictionWeights     
            #shoulder term : 
            model.shoulderWeights = self.shoulderWeights
            model.shoulder_hlim = self.shoulder_hlim            

            # Add model to the list of model
            self.ListAction.append(model)


        # Terminal Node
        if linearModel :
            self.terminalModel = quadruped_walkgen.ActionModelQuadruped()   
        else : 
            self.terminalModel = quadruped_walkgen.ActionModelQuadrupedNonLinear()

        # Model parameters of terminal node    
        self.terminalModel.dt = self.dt 
        self.terminalModel.mass = self.mass      
        self.terminalModel.gI = self.gI 
        self.terminalModel.mu = self.mu
        self.terminalModel.min_fz = self.min_fz
        self.terminalModel.shoulderWeights = self.shoulderWeights
        self.terminalModel.shoulder_hlim = self.shoulder_hlim
       
        # Weights vectors of terminal node
        self.terminalModel.stateWeights = self.stateWeight
        self.terminalModel.forceWeights = np.zeros(12)
        self.terminalModel.frictionWeights = 0.

        # Shooting problem
        self.problem = crocoddyl.ShootingProblem(np.zeros(12),  self.ListAction, self.terminalModel)

        # DDP Solver
        self.ddp = crocoddyl.SolverDDP(self.problem)

        # Warm start
        self.x_init = []
        self.u_init = []
            

    def updateProblem(self,fsteps,xref):
        """Update the dynamic of the model list according to the predicted position of the feet, 
        and the desired state. 

        Args:
            fsteps (6x13): Position of the feet in local frame
            xref (12x17): Desired state vector for the whole gait cycle
            (the initial state is the first column) 
        """
        # Update position of the feet
        self.fsteps[:,:] = fsteps[:,:]

        # Update initial state of the problem
        self.problem.x0 = xref[:,0]
        
        #Construction of the gait matrix representing the feet in contact with the ground
        self.index = next((idx for idx, val in np.ndenumerate(self.fsteps[:, 0]) if val==0.0), 0.0)[0]
        self.gait[:, 0] = self.fsteps[:, 0]
        self.gait[:self.index, 1:] = 1.0 - (np.isnan(self.fsteps[:self.index, 1::3]) | (self.fsteps[:self.index, 1::3] == 0.0))
        # Replace NaN values by zeroes
        self.fsteps[np.isnan(self.fsteps)] = 0.0      
      
        j = 0
        k_cum = 0
        L = []
        
        # Iterate over all phases of the gait
        # The first column of xref correspond to the current state 
        while (self.gait[j, 0] != 0):
            for i in range(k_cum, k_cum+np.int(self.gait[j, 0])):
                # Update model   
                self.ListAction[i].updateModel(np.reshape(self.fsteps[j, 1:], (3, 4), order='F') , xref[:, i]  , self.gait[j, 1:])           
                
            k_cum += np.int(self.gait[j, 0])
            j += 1

        # Update model of the terminal model
        self.terminalModel.updateModel(np.reshape(self.fsteps[j-1, 1:], (3, 4), order='F') , xref[:,-1] , self.gait[j-1, 1:])  

        return 0       
        


    def solve(self, k, fstep_planner):
        """ Solve the MPC problem 

        Args:
            k : Iteration 
            fstep_planner : Object that includes the feet predicted position and the desired state vector
        """ 

        # Update the dynamic depending on the predicted feet position
        self.updateProblem(fstep_planner.fsteps , fstep_planner.xref )

        self.x_init.clear()
        self.u_init.clear()

        # Warm start : set candidate state and input vector
        if self.warm_start and k != 0:            
  
            self.u_init = self.ddp.us[1:] 
            self.u_init.append(np.repeat(self.gait[self.index-1,1:],3)*np.array(4*[0.5,0.5,5.]))
       

            self.x_init = self.ddp.xs[2:]
            self.x_init.insert(0,fstep_planner.xref[:,0])
            self.x_init.append(self.ddp.xs[-1])  

        self.ddp.solve(self.x_init ,  self.u_init, self.max_iteration )

        return 0

    def get_latest_result(self):
        """Returns the desired contact forces that have been computed by the last iteration of the MPC
        Args:
        """
        return np.reshape(np.asarray(self.ddp.us[0])  , (12,))

    def get_xrobot(self):
        """Returns the state vectors predicted by the mpc throughout the time horizon, the initial column is deleted as it corresponds
        initial state vector
        Args:
        """

        return np.array(self.ddp.xs)[1:,:].transpose()

    def get_fpredicted(self):
        """Returns the force vectors command predicted by the mpc throughout the time horizon, 
        Args:
        """

        return np.array(self.ddp.us)[:,:].transpose()[:,:]
    
    def updateActionModel(self) : 
        """Update the quadruped model with the new weights or model parameters. Useful to try new weights without modify this class
        """

        for elt in self.ListAction:
            elt.dt = self.dt 
            elt.mass = self.mass      
            elt.gI = self.gI 
            elt.mu = self.mu
            elt.min_fz = self.min_fz
            elt.max_fz = self.max_fz

            # Weights vectors
            elt.stateWeights = self.stateWeight
            elt.forceWeights = self.forceWeights
            elt.frictionWeights = self.frictionWeights     
           
            #shoulder term : 
            elt.shoulderWeights = self.shoulderWeights
            elt.shoulder_hlim = self.shoulder_hlim
        
        # Model parameters of terminal node    
        self.terminalModel.dt = self.dt 
        self.terminalModel.mass = self.mass      
        self.terminalModel.gI = self.gI 
        self.terminalModel.mu = self.mu
        self.terminalModel.min_fz = self.min_fz
       
        # Weights vectors of terminal node
        self.terminalModel.stateWeights = self.stateWeight
        self.terminalModel.forceWeights = np.zeros(12)
        self.terminalModel.frictionWeights = 0.

        #shoulder term : 
        self.terminalModel.shoulderWeights = self.shoulderWeights
        self.terminalModel.shoulder_hlim = self.shoulder_hlim       

        return 0


    

