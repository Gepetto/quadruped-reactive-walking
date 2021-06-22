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

    def __init__(self, dt=0.02, T_mpc=0.32,  mu=1, inner=True, warm_start=False, min_fz=0.0, N_gait=20):

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
        if inner:
            self.mu = (1/np.sqrt(2))*mu
        else:
            self.mu = mu

        # Weights Vector : States
        # self.stateWeights = np.array([1,1,150,35,30,8,20,20,15,4,4,8])
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
        # self.forceWeights = np.full(12,0.02)
        self.forceWeights = np.array(4*[0.01, 0.01, 0.01])

        # Weight Vector : Friction cone cost
        # self.frictionWeights = 10
        self.frictionWeights = 0.5

        # Max iteration ddp solver
        self.max_iteration = 10

        # Warm Start for the solver
        self.warm_start = warm_start

        # Minimum normal force (N)
        self.min_fz = min_fz

        # Gait matrix
        self.gait = np.zeros((20, 4))
        self.gait_old = np.zeros((1, 4))
        self.index = 0

        # Position of the feet in local frame
        self.fsteps = np.full((20, 12), 0.0)

        # List of the actionModel
        self.ListAction = []

        # Warm start
        self.x_init = []
        self.u_init = []

        # Weights on the shoulder term : term 1
        self.shoulderWeights = np.array(4*[0.3, 0.4])

        # symmetry & centrifugal term in foot position heuristic
        self.centrifugal_term = True
        self.symmetry_term = True

        # Weight on the step command
        self.stepWeights = np.full(4, 0.8)

        # Weights on the previous position predicted : term 2
        self.lastPositionWeights = np.full(8, 2.)

        # When the the foot reaches 10% of the flying phase, the optimisation of the foot
        # positions stops by setting the "lastPositionWeight" on.
        # For exemple, if T_mpc = 0.32s, dt = 0.02s, one flying phase period lasts 7 nodes.
        # When there are 6 knots left before changing steps, the next knot will have its relative weight activated
        self.stop_optim = 0.1
        self.index_stop = int((1 - self.stop_optim)*(int(0.5*self.T_mpc/self.dt) - 1))

        # Index of the control cycle to start the "stopping optimisation"
        self.start_stop_optim = 20

        # Predicted position of feet computed by previous cycle, it will be used with
        # the self.lastPositionWeights weight.
        self.oMl = pin.SE3.Identity()  #  transform from world to local frame ("L")

        self.l_fsteps = np.zeros((3, 4))
        self.o_fsteps = np.zeros((3, 4))

        # Shooting problem
        self.problem = None

        # ddp solver
        self.ddp = None

        # Xs results without the actionStepModel
        self.Xs = np.zeros((20, int(T_mpc/dt)))
        # self.Us = np.zeros((12,int(T_mpc/dt)))

        # Initial foot location (local frame, X,Y plan)
        self.p0 = [0.1946, 0.14695, 0.1946, -0.14695, -0.1946,   0.14695, -0.1946,  -0.14695]

    def solve(self, k, xref, l_feet, oMl=pin.SE3.Identity()):
        """ Solve the MPC problem

        Args:
            k : Iteration
            xref : the desired state vector
            l_feet : current position of the feet
        """

        # Update the dynamic depending on the predicted feet position
        self.updateProblem(k, xref, l_feet, oMl)

        # Solve problem
        self.ddp.solve(self.x_init, self.u_init, self.max_iteration)

        # Get the results
        self.get_fsteps()

        return 0

    def updateProblem(self, k, xref, l_feet, oMl=pin.SE3.Identity()):
        """Update the dynamic of the model list according to the predicted position of the feet,
        and the desired state.

        Args:
        """

        self.oMl = oMl

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

        if k == 0:
            # Create the list of models
            self.create_List_model()

            # By default we suppose we were in the same state previously
            self.gait_old[0, :] = self.gait[0, :].copy()
        else:
            # Update list of models
            self.roll_models()

        j = 0
        # Iterate over all phases of the gait
        # The first column of xref correspond to the current state
        # Gap introduced to take into account the Step model (more nodes than gait phases )
        self.x_init = []
        self.u_init = []
        gap = 0

        next_step_flag = True
        next_step_index = 0  # Get index of incoming step for updatePositionWeights

        while np.any(self.gait[j, :]):

            if self.ListAction[j+gap].__class__.__name__ == "ActionModelQuadrupedStep":

                if next_step_flag:
                    next_step_index = j+gap
                    next_step_flag = False

                self.x_init.append(np.zeros(20))
                self.u_init.append(np.zeros(4))

                if j == 0:
                    self.ListAction[j+gap].updateModel(np.reshape(l_feet[j, :], (3, 4), order='F'),
                                                       xref[:, j], self.gait[0, :] - self.gait_old[0, :])
                else:
                    self.ListAction[j+gap].updateModel(np.reshape(l_feet[j, :], (3, 4), order='F'),
                                                       xref[:, j], self.gait[j, :] - self.gait[j-1, :])

                self.ListAction[j+gap+1].updateModel(np.reshape(l_feet[j, :], (3, 4), order='F'),
                                                     xref[:, j], self.gait[j, :])
                self.x_init.append(np.zeros(20))
                self.u_init.append(np.zeros(12))

                gap += 1
                # self.ListAction[j+1].shoulderWeights = 2*np.array(4*[0.25,0.3])

            else:
                self.ListAction[j+gap].updateModel(np.reshape(l_feet[j, :], (3, 4),
                                                              order='F'), xref[:, j], self.gait[j, :])
                self.x_init.append(np.zeros(20))
                self.u_init.append(np.zeros(12))

            j += 1

        if k > self.start_stop_optim:
            # Update the lastPositionweight
            self.updatePositionWeights(next_step_index)

        # Update model of the terminal model # TODO: Check if correct row of l_feet
        self.terminalModel.updateModel(np.reshape(l_feet[j-1, :], (3, 4), order='F'), xref[:, -1], self.gait[j-1, :])
        self.x_init.append(np.zeros(20))

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
                    print(self.ddp.xs[i][12:])

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

        # Weights vectors
        model.stateWeights = self.stateWeights
        model.forceWeights = self.forceWeights
        model.frictionWeights = self.frictionWeights

        # Weight on feet position
        # will be set when needed
        model.lastPositionWeights = np.full(8, 0.0)
        model.shoulderWeights = self.shoulderWeights
        model.symmetry_term = self.symmetry_term
        model.centrifugal_term = self.centrifugal_term

        return 0

    def update_model_step(self, model):
        """Set intern parameters for step model type
        """
        model.shoulderWeights = self.shoulderWeights
        model.stateWeights = self.stateWeights
        model.stepWeights = self.stepWeights
        model.symmetry_term = self.symmetry_term
        model.centrifugal_term = self.centrifugal_term

        return 0

    def create_List_model(self):
        """Create the List model using ActionQuadrupedModel()
         The same model cannot be used [model]*(T_mpc/dt) because the dynamic changes for each nodes
        """

        j = 0

        # Iterate over all phases of the gait
        # The first column of xref correspond to the current state
        while np.any(self.gait[j, :]):

            model = quadruped_walkgen.ActionModelQuadrupedAugmented()

            # Update intern parameters
            self.update_model_augmented(model)

            # Add model to the list of model
            self.ListAction.append(model)

            # No optimisation on the first line
            if np.any(self.gait[j+1, :]) and not np.array_equal(self.gait[j, :], self.gait[j+1, :]):

                model = quadruped_walkgen.ActionModelQuadrupedStep()
                # Update intern parameters
                self.update_model_step(model)

                # Add model to the list of model
                self.ListAction.append(model)

            j += 1

        # Model parameters of terminal node
        self.terminalModel = quadruped_walkgen.ActionModelQuadrupedAugmented()
        self.update_model_augmented(self.terminalModel)
        # Weights vectors of terminal node
        self.terminalModel.forceWeights = np.zeros(12)
        self.terminalModel.frictionWeights = 0.
        self.terminalModel.shoulderWeights = np.full(8, 0.0)
        self.terminalModel.lastPositionWeights = np.full(8, 0.0)

        # Shooting problem
        self.problem = crocoddyl.ShootingProblem(np.zeros(20),  self.ListAction, self.terminalModel)

        # DDP Solver
        self.ddp = crocoddyl.SolverDDP(self.problem)

        return 0

    def roll_models(self):
        """
        Move one step further in the gait cycle
        Add and remove corresponding model in ListAction
        """

        # Remove first model
        flag = False
        if self.ListAction[0].__class__.__name__ == "ActionModelQuadrupedStep":
            self.ListAction.pop(0)
            flag = True
        model = self.ListAction.pop(0)

        # Add last model & step model if needed
        if flag:  # not np.array_equal(self.gait_old[0, :], self.gait[0, :]):
            modelStep = quadruped_walkgen.ActionModelQuadrupedStep()
            self.update_model_step(modelStep)

            # Add model to the list of model
            self.ListAction.append(modelStep)

        # reset to 0 the weight lastPosition
        model.lastPositionWeights = np.full(8, 0.0)
        self.ListAction.append(model)

        return 0

    def get_fsteps(self):
        """Create the matrices fstep, the position of the feet predicted during the control cycle.

        To be used after the solve function.
        """
        ##################################################
        # Get command vector without actionModelStep node
        ##################################################
        Us = self.ddp.us
        for elt in Us:
            if len(elt) == 4:
                Us.remove(elt)
        self.Us = np.array(Us)[:, :].transpose()

        ################################################
        # Get state vector without actionModelStep node
        ################################################
        # self.Xs[:,0 ] = np.array(self.ddp.xs[0])
        k = 1
        gap = 1
        for elt in self.ListAction:
            if elt.__class__.__name__ != "ActionModelQuadrupedStep":
                self.Xs[:, k - gap] = np.array(self.ddp.xs[k])
            else:
                gap += 1
            k = k + 1

        ########################################
        # Compute fsteps using the state vector
        ########################################

        j = 0

        # Iterate over all phases of the gait
        while np.any(self.gait[j, :]):
            self.fsteps[j, :] = np.repeat(self.gait[j, :], 3)*np.concatenate([self.Xs[12:14, j], [0.],
                                                                              self.Xs[14:16, j], [0.],
                                                                              self.Xs[16:18, j], [0.],
                                                                              self.Xs[18:20, j], [0.]])
            j += 1

        ####################################################
        # Compute the current position of feet in contact
        # and the position of desired feet in flying phase
        # in local frame
        #####################################################

        for i in range(4):
            index = next((idx for idx, val in np.ndenumerate(
                self.fsteps[:, 3*i]) if ((not (val == 0)) and (not np.isnan(val)))), [-1])[0]
            pos_tmp = np.reshape(
                np.array(self.oMl * (np.array([self.fsteps[index, (i*3):(3+i*3)]]).transpose())), (3, 1))
            self.o_fsteps[:2, i] = pos_tmp[0:2, 0]

        return self.fsteps

    def updatePositionWeights(self, next_step_index):
        """Update the parameters in the ListAction to keep the next foot position at the same position computed by the
         previous control cycle and avoid re-optimization at the end of the flying phase
        """

        if next_step_index == self.index_stop:
            self.ListAction[next_step_index + 1].lastPositionWeights = np.repeat(
                (np.array([1, 1, 1, 1]) - self.gait[0, :]), 2) * self.lastPositionWeights

        return 0

    def get_xrobot(self):
        """Returns the state vectors predicted by the mpc throughout the time horizon, the initial column is
        deleted as it corresponds initial state vector
        Args:
        """

        return np.array(self.ddp.xs)[1:, :].transpose()

    def get_fpredicted(self):
        """Returns the force vectors command predicted by the mpc throughout the time horizon,
        Args:
        """

        return np.array(self.ddp.us)[:, :].transpose()[:, :]
