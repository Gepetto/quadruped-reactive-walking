# coding: utf8

import numpy as np
import libquadruped_reactive_walking as MPC
from multiprocessing import Process, Value, Array
from utils_mpc import quaternionToRPY
# import crocoddyl_class.MPC_crocoddyl as MPC_crocoddyl


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
        mpc_type (bool): True to have PA's MPC, False to have Thomas's MPC
        dt (float): Time step of the MPC
        n_steps (int): Number of time steps in one gait cycle
        k_mpc (int): Number of inv dyn time step for one iteration of the MPC
        T_gait (float): Duration of one period of gait
        q_init (array): the default position of the robot
        multiprocessing (bool): Enable/Disable running the MPC with another process
    """

    def __init__(self, mpc_type, dt, n_steps, k_mpc, T_gait, q_init, multiprocessing=False):

        self.f_applied = np.zeros((12,))
        self.not_first_iter = False

        # Number of TSID steps for 1 step of the MPC
        self.k_mpc = k_mpc

        self.dt = dt
        self.n_steps = n_steps
        self.T_gait = T_gait
        self.gait_memory = np.zeros(4)

        self.mpc_type = mpc_type
        self.multiprocessing = multiprocessing
        if multiprocessing:  # Setup variables in the shared memory
            self.newData = Value('b', False)
            self.newResult = Value('b', False)
            self.dataIn = Array('d', [0.0] * (1 + (np.int(self.n_steps)+1) * 12 + 13*20))
            self.dataOut = Array('d', [0] * 24 * (np.int(self.n_steps)))
            self.fsteps_future = np.zeros((20, 13))
            self.running = Value('b', True)
        else:
            # Create the new version of the MPC solver object
            if mpc_type:
                self.mpc = MPC.MPC(dt, n_steps, T_gait)
            else:
                self.mpc = MPC_crocoddyl.MPC_crocoddyl(
                    dt=dt, T_mpc=T_gait, mu=0.9, inner=False, linearModel=True, n_period=int((dt * n_steps)/T_gait))

        # Setup initial result for the first iteration of the main control loop
        x_init = np.zeros(12)
        x_init[0:3] = q_init[0:3, 0]
        x_init[3:6] = quaternionToRPY(q_init[3:7, 0]).ravel()
        self.last_available_result = np.zeros((24, (np.int(self.n_steps))))
        self.last_available_result[:, 0] = np.hstack((x_init, np.array([0.0, 0.0, 8.0] * 4)))

    def solve(self, k, fstep_planner):
        """Call either the asynchronous MPC or the synchronous MPC depending on the value of multiprocessing during
        the creation of the wrapper

        Args:
            k (int): Number of inv dynamics iterations since the start of the simulation
            fstep_planner (object): FootstepPlanner object of the control loop
        """

        if self.multiprocessing:  # Run in parallel process
            self.run_MPC_asynchronous(k, fstep_planner)
        else:  # Run in the same process than main loop
            self.run_MPC_synchronous(k, fstep_planner)

        if k > 2:
            self.last_available_result[12:(12+self.n_steps), :] = np.roll(self.last_available_result[12:(12+self.n_steps), :], -1, axis=1)

        pt = 0
        while (fstep_planner.gait[pt, 0] != 0):
            pt += 1
        if k > 2 and not np.array_equal(fstep_planner.gait[0, 1:], fstep_planner.gait[pt-1, 1:]):
            mass = 2.5  # Todo: grab from URDF?
            nb_ctc = np.sum(fstep_planner.gait[pt-1, 1:])
            F = 9.81 * mass / nb_ctc
            self.last_available_result[12:, self.n_steps-1] = np.zeros(12)
            for i in range(4):
                if (fstep_planner.gait[pt-1, 1+i] == 1):
                    self.last_available_result[12+3*i+2, self.n_steps-1] = F

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

    def run_MPC_synchronous(self, k, fstep_planner):
        """Run the MPC (synchronous version) to get the desired contact forces for the feet currently in stance phase

        Args:
            k (int): Number of inv dynamics iterations since the start of the simulation
            fstep_planner (object): FootstepPlanner object of the control loop
        """

        # Run the MPC to get the reference forces and the next predicted state
        # Result is stored in mpc.f_applied, mpc.q_next, mpc.v_next

        if self.mpc_type:
            # OSQP MPC
            # Replace NaN values by 0.0 (shared memory cannot handle np.nan)
            fstep_planner.fsteps_mpc[np.isnan(fstep_planner.fsteps_mpc)] = 0.0
            self.mpc.run(np.int(k), fstep_planner.xref.copy(), fstep_planner.fsteps_mpc.copy())
        else:
            # Crocoddyl MPC
            self.mpc.solve(k, fstep_planner)

        # Output of the MPC
        self.f_applied = self.mpc.get_latest_result()

    def run_MPC_asynchronous(self, k, fstep_planner):
        """Run the MPC (asynchronous version) to get the desired contact forces for the feet currently in stance phase

        Args:
            k (int): Number of inv dynamics iterations since the start of the simulation
            fstep_planner (object): FootstepPlanner object of the control loop
        """

        # If this is the first iteration, creation of the parallel process
        if (k == 0):
            p = Process(target=self.create_MPC_asynchronous, args=(
                self.newData, self.newResult, self.dataIn, self.dataOut, self.running))
            p.start()

        # Stacking data to send them to the parallel process
        self.compress_dataIn(k, fstep_planner)
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
                kf, xref_1dim, fsteps_1dim = self.decompress_dataIn(dataIn)

                # Reshaping 1-dimensional data
                k = int(kf[0])
                xref = np.reshape(xref_1dim, (12, self.n_steps+1))
                fsteps = np.reshape(fsteps_1dim, (20, 13))

                # Create the MPC object of the parallel process during the first iteration
                if k == 0:
                    # loop_mpc = MPC.MPC(self.dt, self.n_steps, self.T_gait)
                    if self.mpc_type:
                        loop_mpc = MPC.MPC(self.dt, self.n_steps, self.T_gait)
                    else:
                        loop_mpc = MPC_crocoddyl.MPC_crocoddyl(self.dt, self.T_gait, 1, True)
                        dummy_fstep_planner = Dummy()

                # Run the asynchronous MPC with the data that as been retrieved
                if self.mpc_type:
                    fsteps[np.isnan(fsteps)] = 0.0
                    loop_mpc.run(np.int(k), xref, fsteps)
                else:
                    dummy_fstep_planner.xref = xref
                    dummy_fstep_planner.fsteps = fsteps
                    loop_mpc.solve(k, dummy_fstep_planner)

                # Store the result (predicted state + desired forces) in the shared memory
                # print(len(self.dataOut))
                # print((loop_mpc.get_latest_result()).shape)

                self.dataOut[:] = loop_mpc.get_latest_result().ravel(order='F')

                # Set shared variable to true to signal that a new result is available
                newResult.value = True

        return 0

    def compress_dataIn(self, k, fstep_planner):
        """Compress data in a single C-type array that belongs to the shared memory to send data from the main control
        loop to the asynchronous MPC

        Args:
            k (int): Number of inv dynamics iterations since the start of the simulation
            fstep_planner (object): FootstepPlanner object of the control loop
        """

        # Replace NaN values by 0.0 to be stored in C-type arrays
        fstep_planner.fsteps[np.isnan(fstep_planner.fsteps)] = 0.0

        # Compress data in the shared input array
        self.dataIn[:] = np.concatenate([[(k/self.k_mpc)], fstep_planner.xref.ravel(),
                                         fstep_planner.fsteps.ravel()], axis=0)

        return 0.0

    def decompress_dataIn(self, dataIn):
        """Decompress data from a single C-type array that belongs to the shared memory to retrieve data from the main control
        loop in the asynchronous MPC

        Args:
            dataIn (Array): shared array that contains the data the asynchronous MPC will use as inputs
        """

        # Sizes of the different variables that are stored in the C-type array
        sizes = [0, 1, (np.int(self.n_steps)+1) * 12, 13*20]
        csizes = np.cumsum(sizes)

        # Return decompressed variables in a list
        return [dataIn[csizes[i]:csizes[i+1]] for i in range(len(sizes)-1)]

    def convert_dataOut(self):
        """Return the result of the asynchronous MPC (desired contact forces) that is stored in the shared memory
        """

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
            fsteps (13x20 array): the remaining number of steps of each phase of the gait (first column)
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
