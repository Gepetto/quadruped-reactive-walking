# coding: utf8

import crocoddyl
import numpy as np
import quadruped_walkgen as quadruped_walkgen


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

    def __init__(self, params,  mu=1, inner=True, linearModel=True):

        # Time step of the solver
        self.dt = params.dt_mpc

        # Period of the MPC
        self.T_mpc = params.T_mpc

        # Mass of the robot
        self.mass = params.mass

        # Inertia matrix of the robot in body frame
        self.gI = np.array(params.I_mat.tolist()).reshape((3, 3))

        # Friction coefficient
        if inner:
            self.mu = (1/np.sqrt(2))*mu
        else:
            self.mu = mu

        # Gain from OSQP MPC

        # self.w_x = np.sqrt(0.5)
        # self.w_y = np.sqrt(0.5)
        # self.w_z = np.sqrt(2.)
        # self.w_roll = np.sqrt(0.11)
        # self.w_pitch = np.sqrt(0.11)
        # self.w_yaw = np.sqrt(0.11)
        # self.w_vx = np.sqrt(2.*np.sqrt(0.5))
        # self.w_vy = np.sqrt(2.*np.sqrt(0.5))
        # self.w_vz = np.sqrt(2.*np.sqrt(2.))
        # self.w_vroll = np.sqrt(0.05*np.sqrt(0.11))
        # self.w_vpitch = np.sqrt(0.05*np.sqrt(0.11))
        # self.w_vyaw = np.sqrt(0.05*np.sqrt(0.11))

        # from osqp, config
        # self.stateWeight = np.sqrt([2.0, 2.0, 20.0, 0.25, 0.25, 0.25, 0.2, 0.2, 5., 0.0, 0.0, 0.3])

        # Set of gains to get a better behaviour with mpc height used in WBC
        self.w_x = 0.3
        self.w_y = 0.3
        self.w_z = 2
        self.w_roll = 0.9
        self.w_pitch = 1.
        self.w_yaw = 0.4
        self.w_vx = 1.5*np.sqrt(self.w_x)
        self.w_vy = 2*np.sqrt(self.w_y)
        self.w_vz = 2*np.sqrt(self.w_z)
        self.w_vroll = 0.05*np.sqrt(self.w_roll)
        self.w_vpitch = 0.07*np.sqrt(self.w_pitch)
        self.w_vyaw = 0.05*np.sqrt(self.w_yaw)

        self.stateWeight = np.array([self.w_x, self.w_y, self.w_z, self.w_roll, self.w_pitch, self.w_yaw,
                                     self.w_vx, self.w_vy, self.w_vz, self.w_vroll, self.w_vpitch, self.w_vyaw])

        # Weight Vector : Force Norm
        self.forceWeights = np.array(4*[0.007, 0.007, 0.007])

        # Weight Vector : Friction cone cost
        self.frictionWeights = 1.0

        # Max iteration ddp solver
        self.max_iteration = 10

        # Warm Start for the solver
        self.warm_start = True

        # Minimum normal force (N)
        self.min_fz = 0.2
        self.max_fz = 25    

        # Gait matrix
        self.gait = np.zeros((params.N_gait, 4))
        self.index = 0

        # Weight on the shoulder term :
        self.shoulderWeights = 5.
        self.shoulder_hlim = 0.23

        # Integration scheme
        self.implicit_integration = True

        # Position of the feet
        self.fsteps = np.full((params.N_gait, 12), np.nan)

        # List of the actionModel
        self.ListAction = []

        # Initialisation of the List model using ActionQuadrupedModel()
        # The same model cannot be used [model]*(T_mpc/dt) because the dynamic
        # model changes for each nodes.
        for i in range(int(self.T_mpc/self.dt)):
            if linearModel:
                model = quadruped_walkgen.ActionModelQuadruped()
            else:
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
            # shoulder term :
            model.shoulderWeights = self.shoulderWeights
            model.shoulder_hlim = self.shoulder_hlim

            # integration scheme
            model.implicit_integration = self.implicit_integration

            # Add model to the list of model
            self.ListAction.append(model)

        # Terminal Node
        if linearModel:
            self.terminalModel = quadruped_walkgen.ActionModelQuadruped()
        else:
            self.terminalModel = quadruped_walkgen.ActionModelQuadrupedNonLinear()

        # Model parameters of terminal node
        self.terminalModel.dt = self.dt
        self.terminalModel.mass = self.mass
        self.terminalModel.gI = self.gI
        self.terminalModel.mu = self.mu
        self.terminalModel.min_fz = self.min_fz
        self.terminalModel.shoulderWeights = self.shoulderWeights
        self.terminalModel.shoulder_hlim = self.shoulder_hlim
        self.terminalModel.implicit_integration = self.implicit_integration

        # Weights vectors of terminal node
        self.terminalModel.stateWeights = 10*self.stateWeight
        self.terminalModel.forceWeights = np.zeros(12)
        self.terminalModel.frictionWeights = 0.

        # Shooting problem
        self.problem = crocoddyl.ShootingProblem(np.zeros(12),  self.ListAction, self.terminalModel)

        # DDP Solver
        self.ddp = crocoddyl.SolverDDP(self.problem)

        # Warm start
        self.x_init = []
        self.u_init = []

    def updateProblem(self, fsteps, xref):
        """Update the dynamic of the model list according to the predicted position of the feet,
        and the desired state

        Args:
            fsteps (6x13): Position of the feet in local frame
            xref (12x17): Desired state vector for the whole gait cycle
            (the initial state is the first column)
        """
        # Update position of the feet
        self.fsteps[:, :] = fsteps[:, :]

        # Update initial state of the problem
        self.problem.x0 = xref[:, 0]

        # Construction of the gait matrix representing the feet in contact with the ground
        self.index = 0
        while (np.any(self.fsteps[self.index, :])):
            self.index += 1
        self.gait[:self.index, :] = 1.0 - (self.fsteps[:self.index, 0::3] == 0.0)
        self.gait[self.index:, :] = 0.0

        # Iterate over all phases of the gait
        # The first column of xref correspond to the current state
        for j in range(self.index):
            # Update model
            self.ListAction[j].updateModel(np.reshape(self.fsteps[j, :], (3, 4), order='F'),
                                           xref[:, j+1], self.gait[j, :])

        # Update model of the terminal model
        self.terminalModel.updateModel(np.reshape(
            self.fsteps[self.index-1, :], (3, 4), order='F'), xref[:, -1], self.gait[self.index-1, :])

        return 0

    def solve(self, k, xref, fsteps):
        """ Solve the MPC problem

        Args:
            k : Iteration
            xref : desired state vector
            fsteps : feet predicted positions
        """

        # Update the dynamic depending on the predicted feet position
        self.updateProblem(fsteps, xref)

        self.x_init.clear()
        self.u_init.clear()

        # Warm start : set candidate state and input vector
        if self.warm_start and k != 0:

            self.u_init = self.ddp.us[1:]
            self.u_init.append(np.repeat(self.gait[self.index-1, :], 3)*np.array(4*[0.5, 0.5, 5.]))

            self.x_init = self.ddp.xs[2:]
            self.x_init.insert(0, xref[:, 0])
            self.x_init.append(self.ddp.xs[-1])
        
        else :

            self.x_init.append(xref[:, 0] )

            for i in range(len(self.ListAction)) :
                self.x_init.append(np.zeros(12) )
                self.u_init.append(np.zeros(12) )
            

       

        """print("1")
        from IPython import embed
        embed()"""
        self.ddp.solve(self.x_init,  self.u_init, self.max_iteration)

        """print("3")
        from IPython import embed
        embed()"""

        return 0

    def get_latest_result(self):
        """Returns the desired contact forces that have been computed by the last iteration of the MPC
        Args:
        """

        output = np.zeros((24, int(self.T_mpc/self.dt)))
        for i in range(int(self.T_mpc/self.dt)):
            output[:12, i] = np.asarray(self.ddp.xs[i+1])
            output[12:, i] = np.asarray(self.ddp.us[i])
        return output

    def get_xrobot(self):
        """Returns the state vectors predicted by the mpc throughout the time horizon, the initial column
        is deleted as it corresponds initial state vector
        Args:
        """

        return np.array(self.ddp.xs)[1:, :].transpose()

    def get_fpredicted(self):
        """Returns the force vectors command predicted by the mpc throughout the time horizon,
        Args:
        """

        return np.array(self.ddp.us)[:, :].transpose()[:, :]

    def updateActionModel(self):
        """Update the quadruped model with the new weights or model parameters.
        Useful to try new weights without modify this class
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

            # shoulder term :
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

        # shoulder term :
        self.terminalModel.shoulderWeights = self.shoulderWeights
        self.terminalModel.shoulder_hlim = self.shoulder_hlim

        return 0
