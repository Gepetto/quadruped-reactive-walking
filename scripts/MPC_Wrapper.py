# coding: utf8

import numpy as np
import libquadruped_reactive_walking as MPC
from multiprocessing import Process, Value, Array
import crocoddyl_class.MPC_crocoddyl as MPC_crocoddyl
import crocoddyl_class.MPC_crocoddyl_planner as MPC_crocoddyl_planner
import pinocchio as pin

class Dummy:
    """Dummy class to store variables"""

    def __init__(self):

        self.xref = None  # Desired trajectory
        self.fsteps = None  # Desired location of footsteps

        pass


class MPC_Wrapper:
    """Wrapper to run both types of MPC (OQSP or Crocoddyl) with the possibility to run OSQP in
    a parallel process

    Args:
        mpc_type (int): 0 for OSQP MPC, 1, 2, 3 for Crocoddyl MPCs
        dt (float): Time step of the MPC
        n_steps (int): Number of time steps in one gait cycle
        k_mpc (int): Number of inv dyn time step for one iteration of the MPC
        T_gait (float): Duration of one period of gait
        q_init (array): the default position of the robot
        multiprocessing (bool): Enable/Disable running the MPC with another process
    """

    def __init__(self, params, q_init):

        self.f_applied = np.zeros((12,))
        self.not_first_iter = False

        self.params = params

        # Number of WBC steps for 1 step of the MPC
        self.k_mpc = int(params.dt_mpc/params.dt_wbc)

        self.dt = params.dt_mpc
        self.n_steps = np.int(params.T_mpc/params.dt_mpc)
        self.T_gait = params.T_gait
        self.N_gait = params.N_gait
        self.gait_past = np.zeros(4)
        self.gait_next = np.zeros(4)
        self.mass = params.mass

        self.mpc_type = params.type_MPC
        self.multiprocessing = params.enable_multiprocessing
        if self.multiprocessing:  # Setup variables in the shared memory
            self.newData = Value('b', False)
            self.newResult = Value('b', False)            
            if self.mpc_type == 3:  # Need more space to store optimized footsteps and l_fsteps to stop the optimization around it
                self.dataIn = Array('d', [0.0] * (1 + (np.int(self.n_steps)+1) * 12 + 12*self.N_gait + 12) )
                self.dataOut = Array('d', [0] * 32 * (np.int(self.n_steps)))
            else:
                self.dataIn = Array('d', [0.0] * (1 + (np.int(self.n_steps)+1) * 12 + 12*self.N_gait))
                self.dataOut = Array('d', [0] * 24 * (np.int(self.n_steps)))
            self.fsteps_future = np.zeros((self.N_gait, 12))
            self.running = Value('b', True)
        else:
            # Create the new version of the MPC solver object
            if self.mpc_type == 0:  # OSQP MPC
                self.mpc = MPC.MPC(params)  # self.dt, self.n_steps, self.T_gait, self.N_gait)
            elif self.mpc_type == 1:  # Crocoddyl MPC Linear
                self.mpc = MPC_crocoddyl.MPC_crocoddyl(params, mu=0.9, inner=False, linearModel=True)
            elif self.mpc_type == 2:  # Crocoddyl MPC Non-Linear
                self.mpc = MPC_crocoddyl.MPC_crocoddyl(params, mu=0.9, inner=False, linearModel=False)
            else:  # Crocoddyl MPC Non-Linear with footsteps optimization
                self.mpc = MPC_crocoddyl_planner.MPC_crocoddyl_planner(params, mu=0.9, inner=False)

        # Setup initial result for the first iteration of the main control loop
        x_init = np.zeros(12)
        x_init[0:3] = q_init[0:3, 0]
        x_init[3:6] = pin.rpy.matrixToRpy(pin.Quaternion(q_init[3:7, 0]).toRotationMatrix())
        if self.mpc_type == 3:  # Need more space to store optimized footsteps
            self.last_available_result = np.zeros((32, (np.int(self.n_steps))))
        else:
            self.last_available_result = np.zeros((24, (np.int(self.n_steps))))
        self.last_available_result[:24, 0] = np.hstack((x_init, np.array([0.0, 0.0, 8.0] * 4)))

    def solve(self, k, xref, fsteps, gait, l_targetFootstep):
        """Call either the asynchronous MPC or the synchronous MPC depending on the value of multiprocessing during
        the creation of the wrapper

        Args:
            k (int): Number of inv dynamics iterations since the start of the simulation
            xref (12xN): Desired state vector for the whole prediction horizon
            fsteps (12xN array): the [x, y, z]^T desired position of each foot for each time step of the horizon
            gait (4xN array): Contact state of feet (gait matrix)
            l_targetFootstep (3x4 array) : 4*[x, y, z]^T target position in local frame, to stop the optimisation of the feet location around it
        """

        if self.multiprocessing:  # Run in parallel process
            self.run_MPC_asynchronous(k, xref, fsteps, l_targetFootstep)
        else:  # Run in the same process than main loop
            self.run_MPC_synchronous(k, xref, fsteps, l_targetFootstep)

        if not np.allclose(gait[0, :], self.gait_past):  # If gait status has changed
            if np.allclose(gait[0, :], self.gait_next):  # If we're still doing what was planned the last time MPC was solved
                self.last_available_result[12:24, 0] = self.last_available_result[12:24, 1].copy()
            else:  # Otherwise use a default contact force command till we get the actual result of the MPC for this new sequence
                F = 9.81 * self.mass / np.sum(gait[0, :])
                self.last_available_result[12:24:3, 0] = 0.0
                self.last_available_result[13:24:3, 0] = 0.0
                self.last_available_result[14:24:3, 0] = F

            self.last_available_result[12:24, 1:] = 0.0
            self.gait_past = gait[0, :].copy()

        self.gait_next = gait[1, :].copy()

        return 0

    def get_latest_result(self):
        """Return the desired contact forces that have been computed by the last iteration of the MPC
        If a new result is available, return the new result. Otherwise return the old result again.
        """

        if (self.not_first_iter):
            if self.multiprocessing:
                if self.newResult.value:
                    self.newResult.value = False
                    # Retrieve desired contact forces with through the memory shared with the asynchronous
                    self.last_available_result = self.convert_dataOut()
                    return self.last_available_result
                else:
                    return self.last_available_result
            else:
                # Directly retrieve desired contact force of the synchronous MPC object
                return self.f_applied
        else:
            # Default forces for the first iteration
            self.not_first_iter = True
            return self.last_available_result

    def run_MPC_synchronous(self, k, xref, fsteps, l_targetFootstep):
        """Run the MPC (synchronous version) to get the desired contact forces for the feet currently in stance phase

        Args:
            k (int): Number of inv dynamics iterations since the start of the simulation
            xref (12xN): Desired state vector for the whole prediction horizon
            fsteps (12xN array): the [x, y, z]^T desired position of each foot for each time step of the horizon
            l_targetFootstep (3x4 array) : [x, y, z]^T target position in local frame, to stop the optimisation of the feet location around it
        """

        # Run the MPC to get the reference forces and the next predicted state
        # Result is stored in mpc.f_applied, mpc.q_next, mpc.v_next

        if self.mpc_type == 0:
            # OSQP MPC
            self.mpc.run(np.int(k), xref.copy(), fsteps.copy())
        elif self.mpc_type == 3: # Add goal position to stop the optimisation
            # Crocoddyl MPC
            self.mpc.solve(k, xref.copy(), fsteps.copy(), l_targetFootstep)
        else:
            # Crocoddyl MPC
            self.mpc.solve(k, xref.copy(), fsteps.copy())

        # Output of the MPC
        self.f_applied = self.mpc.get_latest_result()

    def run_MPC_asynchronous(self, k, xref, fsteps, l_targetFootstep):
        """Run the MPC (asynchronous version) to get the desired contact forces for the feet currently in stance phase

        Args:
            k (int): Number of inv dynamics iterations since the start of the simulation
            xref (12xN): Desired state vector for the whole prediction horizon
            fsteps (12xN array): the [x, y, z]^T desired position of each foot for each time step of the horizon
            params (object): stores parameters
            l_targetFootstep (3x4 array) : [x, y, z]^T target position in local frame, to stop the optimisation of the feet location around it
        """

        # If this is the first iteration, creation of the parallel process
        if (k == 0):
            p = Process(target=self.create_MPC_asynchronous, args=(
                self.newData, self.newResult, self.dataIn, self.dataOut, self.running))
            p.start()

        # Stacking data to send them to the parallel process
        self.compress_dataIn(k, xref, fsteps, l_targetFootstep)
        self.newData.value = True

        return 0

    def create_MPC_asynchronous(self, newData, newResult, dataIn, dataOut, running):
        """Parallel process with an infinite loop that run the asynchronous MPC

        Args:
            newData (Value): shared variable that is true if new data is available, false otherwise
            newResult (Value): shared variable that is true if a new result is available, false otherwise
            dataIn (Array): shared array that contains the data the asynchronous MPC will use as inputs
            dataOut (Array): shared array that contains the result of the asynchronous MPC
            running (Value): shared variable to stop the infinite loop when set to False
        """

        # print("Entering infinite loop")
        while running.value:
            # Checking if new data is available to trigger the asynchronous MPC
            if newData.value:

                # Set the shared variable to false to avoid re-trigering the asynchronous MPC
                newData.value = False
                # print("New data detected")

                # Retrieve data thanks to the decompression function and reshape it
                if self.mpc_type != 3:
                    kf, xref_1dim, fsteps_1dim = self.decompress_dataIn(dataIn)
                else:
                    kf, xref_1dim, fsteps_1dim, l_target_1dim = self.decompress_dataIn(dataIn)
                    l_target = np.reshape(l_target_1dim, (3,4))

                # Reshaping 1-dimensional data
                k = int(kf[0])
                xref = np.reshape(xref_1dim, (12, self.n_steps+1))
                fsteps = np.reshape(fsteps_1dim, (self.N_gait, 12))

                # Create the MPC object of the parallel process during the first iteration
                if k == 0:
                    # loop_mpc = MPC.MPC(self.dt, self.n_steps, self.T_gait)
                    if self.mpc_type == 0:
                        loop_mpc = MPC.MPC(self.params)
                    elif self.mpc_type == 1:  # Crocoddyl MPC Linear
                        loop_mpc = MPC_crocoddyl.MPC_crocoddyl(self.params, mu=0.9, inner=False, linearModel=True)
                    elif self.mpc_type == 2:  # Crocoddyl MPC Non-Linear
                        loop_mpc = MPC_crocoddyl.MPC_crocoddyl(self.params, mu=0.9, inner=False, linearModel=False)
                    else:  # Crocoddyl MPC Non-Linear with footsteps optimization
                        loop_mpc = MPC_crocoddyl_planner.MPC_crocoddyl_planner(self.params, mu=0.9, inner=False)

                # Run the asynchronous MPC with the data that as been retrieved
                if self.mpc_type == 0:
                    fsteps[np.isnan(fsteps)] = 0.0
                    loop_mpc.run(np.int(k), xref, fsteps)
                elif self.mpc_type == 3:
                    loop_mpc.solve(k, xref.copy(), fsteps.copy(), l_target.copy())
                else:
                    loop_mpc.solve(k, xref.copy(), fsteps.copy())

                # Store the result (predicted state + desired forces) in the shared memory
                # print(len(self.dataOut))
                # print((loop_mpc.get_latest_result()).shape)

                self.dataOut[:] = loop_mpc.get_latest_result().ravel(order='F')

                # Set shared variable to true to signal that a new result is available
                newResult.value = True

        return 0

    def compress_dataIn(self, k, xref, fsteps, l_targetFootstep):
        """Compress data in a single C-type array that belongs to the shared memory to send data from the main control
        loop to the asynchronous MPC

        Args:
            k (int): Number of inv dynamics iterations since the start of the simulation
            fstep_planner (object): FootstepPlanner object of the control loop
            l_targetFootstep (3x4 array) : [x, y, z]^T target position in local frame, to stop the optimisation of the feet location around it
        """

        # Replace NaN values by 0.0 to be stored in C-type arrays
        fsteps[np.isnan(fsteps)] = 0.0

        # Compress data in the shared input array
        if self.mpc_type == 3:
            self.dataIn[:] = np.concatenate([[(k/self.k_mpc)], xref.ravel(),
                                            fsteps.ravel(), l_targetFootstep.ravel()], axis=0)
        else:
            self.dataIn[:] = np.concatenate([[(k/self.k_mpc)], xref.ravel(),
                                            fsteps.ravel()], axis=0)

        return 0.0

    def decompress_dataIn(self, dataIn):
        """Decompress data from a single C-type array that belongs to the shared memory to retrieve data from the main control
        loop in the asynchronous MPC

        Args:
            dataIn (Array): shared array that contains the data the asynchronous MPC will use as inputs
        """

        # Sizes of the different variables that are stored in the C-type array
        if self.mpc_type == 3:
            sizes = [0, 1, (np.int(self.n_steps)+1) * 12, 12*self.N_gait, 12]
        else:
            sizes = [0, 1, (np.int(self.n_steps)+1) * 12, 12*self.N_gait]
        csizes = np.cumsum(sizes)

        # Return decompressed variables in a list
        return [dataIn[csizes[i]:csizes[i+1]] for i in range(len(sizes)-1)]

    def convert_dataOut(self):
        """Return the result of the asynchronous MPC (desired contact forces) that is stored in the shared memory
        """

        if self.mpc_type == 3:  # Need more space to store optimized footsteps
            return np.array(self.dataOut[:]).reshape((32, -1), order='F')
        else:
            return np.array(self.dataOut[:]).reshape((24, -1), order='F')

    def roll_asynchronous(self, fsteps):
        """Move one step further in the gait cycle. Since the output of the asynchronous MPC is retrieved by
        TSID during the next call to the MPC, it should not work with the current state of the gait but with the
        gait on step into the future. That way, when TSID retrieves the result, it is consistent with the current
        state of the gait.

        Decrease by 1 the number of remaining step for the current phase of the gait and increase
        by 1 the number of remaining step for the last phase of the gait (periodic motion).
        Simplification: instead of creating a new phase if required (see roll function of FootstepPlanner) we always
        increase the last one by 1 step. That way we don't need to call other functions to predict the position of
        footstep when a new phase is created.

        Args:
            fsteps (13xN_gait array): the remaining number of steps of each phase of the gait (first column)
            and the [x, y, z]^T desired position of each foot for each phase of the gait (12 other columns)
        """

        self.fsteps_future = fsteps.copy()

        # Index of the first empty line
        index = next((idx for idx, val in np.ndenumerate(self.fsteps_future[:, 0]) if val == 0.0), 0.0)[0]

        # Create a new phase if needed or increase the last one by 1 step
        self.fsteps_future[index-1, 0] += 1.0

        # Decrease the current phase by 1 step and delete it if it has ended
        if self.fsteps_future[0, 0] > 1.0:
            self.fsteps_future[0, 0] -= 1.0
        else:
            self.fsteps_future = np.roll(self.fsteps_future, -1, axis=0)
            self.fsteps_future[-1, :] = np.zeros((13, ))

        return 0

    def stop_parallel_loop(self):
        """Stop the infinite loop in the parallel process to properly close the simulation
        """

        self.running.value = False

        return 0
