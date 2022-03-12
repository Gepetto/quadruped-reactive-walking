import ctypes
from ctypes import Structure
from enum import Enum
from multiprocessing import Process, Value, Array
from time import time

import numpy as np

from . import quadruped_reactive_walking as qrw
from .Crocoddyl import MPC_crocoddyl, MPC_crocoddyl_planner


class MPC_type(Enum):
    OSQP = 0
    CROCODDYL_LINEAR = 1
    CROCODDYL_NON_LINEAR = 2
    CROCODDYL_PLANNER = 3
    CROCODDYL_PLANNER_TIME = 4


class DataInCtype(Structure):
    """Ctype data structure for the shared memory between processes."""

    params = qrw.Params()  # Object that holds all controller parameters
    mpc_type = MPC_type(params.type_MPC)  # MPC type
    n_steps = np.int(params.gait.shape[0])  # Colomn size for xref (12 x n_steps)
    N_gait = int(
        params.gait.shape[0]
    )  # Row size for fsteps  (N_gait x 12), from utils_mpc.py

    if mpc_type == MPC_type.CROCODDYL_PLANNER:
        _fields_ = [
            ("k", ctypes.c_int64),
            ("xref", ctypes.c_double * 12 * (n_steps + 1)),
            ("fsteps", ctypes.c_double * 12 * N_gait),
            ("l_fsteps_target", ctypes.c_double * 3 * 4),
            ("oRh", ctypes.c_double * 3 * 3),
            ("oTh", ctypes.c_double * 3 * 1),
            ("position", ctypes.c_double * 3 * 4),
            ("velocity", ctypes.c_double * 3 * 4),
            ("acceleration", ctypes.c_double * 3 * 4),
            ("jerk", ctypes.c_double * 3 * 4),
            ("dt_flying", ctypes.c_double * 4),
        ]
    else:
        _fields_ = [
            ("k", ctypes.c_int64),
            ("xref", ctypes.c_double * 12 * (n_steps + 1)),
            ("fsteps", ctypes.c_double * 12 * N_gait),
        ]


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

        self.t_mpc_solving_start = 0.
        self.t_mpc_solving_duration = 0.

        # Number of WBC steps for 1 step of the MPC
        self.k_mpc = int(params.dt_mpc / params.dt_wbc)

        self.dt = params.dt_mpc
        self.n_steps = np.int(params.gait.shape[0])
        self.N_gait = params.gait.shape[0]
        self.T_gait = params.gait.shape[0] * params.dt_mpc
        self.gait_past = np.zeros(4)
        self.gait_next = np.zeros(4)
        self.mass = params.mass

        self.mpc_type = MPC_type(params.type_MPC)
        self.multiprocessing = params.enable_multiprocessing

        if self.multiprocessing:
            self.newData = Value("b", False)
            self.newResult = Value("b", False)
            self.cost = Value("d", 0.0)
            self.dataIn = Value(DataInCtype)
            if self.mpc_type == MPC_type.CROCODDYL_PLANNER:
                self.dataOut = Array("d", [0] * 32 * (np.int(self.n_steps)))
            else:
                self.dataOut = Array("d", [0] * 24 * (np.int(self.n_steps)))
            self.fsteps_future = np.zeros((self.N_gait, 12))
            self.running = Value("b", True)
        else:
            if self.mpc_type == MPC_type.CROCODDYL_LINEAR:
                self.mpc = MPC_crocoddyl.MPC_crocoddyl(params, mu=0.9, inner=False, linearModel=True)
            elif (self.mpc_type == MPC_type.CROCODDYL_NON_LINEAR):
                self.mpc = MPC_crocoddyl.MPC_crocoddyl(params, mu=0.9, inner=False, linearModel=False)
            elif self.mpc_type == MPC_type.CROCODDYL_PLANNER:
                self.mpc = MPC_crocoddyl_planner.MPC_crocoddyl_planner(params, mu=0.9, inner=False)
            else:
                self.mpc = qrw.MPC(params)
                if self.mpc_type != MPC_type.OSQP:
                    print("Unknown MPC type, using OSQP")
                    self.type = MPC_type.OSQP

        x_init = np.zeros(12)
        x_init[0:6] = q_init[0:6, 0].copy()
        if (self.mpc_type == MPC_type.CROCODDYL_PLANNER):
            self.last_available_result = np.zeros((32, (np.int(self.n_steps))))

        if self.mpc_type == MPC_type.CROCODDYL_PLANNER:
            self.last_available_result = np.zeros((32, np.int(self.n_steps)))
        else:
            self.last_available_result = np.zeros((24, np.int(self.n_steps)))
        self.last_available_result[:24, 0] = np.hstack((x_init, np.array([0., 0., 8.] * 4)))
        self.last_cost = 0.

    def solve(self, k, xref, fsteps, gait, l_fsteps_target=np.zeros((3, 4)), oRh=np.eye(3), oTh=np.zeros((3, 1)),
              position=np.zeros((3, 4)), velocity=np.zeros((3, 4)), acceleration=np.zeros((3, 4)), jerk=np.zeros((3, 4)),
              dt_flying=np.zeros(4)):
        """
        Call either the asynchronous MPC or the synchronous MPC depending on the value of multiprocessing during
        the creation of the wrapper

        Args:
            k (int): Number of inv dynamics iterations since the start of the simulation
            xref (12xN): Desired state vector for the whole prediction horizon
            fsteps (12xN array): the [x, y, z]^T desired position of each foot for each time step of the horizon
            gait (4xN array): Contact state of feet (gait matrix)
            l_fsteps_target (3x4 array) : 4*[x, y, z]^T target position in local frame, to stop the optimisation of the feet location around it
        """

        self.t_mpc_solving_start = time()

        if self.multiprocessing:
            self.run_MPC_asynchronous(k, xref, fsteps, l_fsteps_target, oRh, oTh, position, velocity, acceleration, jerk, dt_flying)
        else:
            self.run_MPC_synchronous(k, xref, fsteps, l_fsteps_target, oRh, oTh, position, velocity, acceleration, jerk, dt_flying)

        if not np.allclose(gait[0, :], self.gait_past):  # If gait status has changed
            if np.allclose(
                gait[0, :], self.gait_next
            ):  # If we're still doing what was planned the last time MPC was solved
                self.last_available_result[12:24, 0] = self.last_available_result[
                    12:24, 1
                ].copy()
            else:  # Otherwise use a default contact force command till we get the actual result of the MPC for this new sequence
                F = 9.81 * self.mass / np.sum(gait[0, :])
                self.last_available_result[12:24:3, 0] = 0.
                self.last_available_result[13:24:3, 0] = 0.
                self.last_available_result[14:24:3, 0] = F

            self.last_available_result[12:24, 1:] = 0.0
            self.gait_past = gait[0, :].copy()

        self.gait_next = gait[1, :].copy()

    def get_latest_result(self):
        """Return the desired contact forces that have been computed by the last iteration of the MPC
        If a new result is available, return the new result. Otherwise return the old result again.
        """

        if self.not_first_iter:
            if self.multiprocessing:
                if self.newResult.value:
                    self.newResult.value = False
                    self.t_mpc_solving_duration = time() - self.t_mpc_solving_start
                    # Retrieve desired contact forces with through the memory shared with the asynchronous
                    self.last_available_result = self.convert_dataOut()
                    self.last_cost = self.cost.value
                    return self.last_available_result, self.last_cost
                else:
                    return self.last_available_result, self.last_cost
            else:
                # Directly retrieve desired contact force of the synchronous MPC object
                return self.f_applied, self.last_cost
        else:
            # Default forces for the first iteration
            self.not_first_iter = True
            return self.last_available_result, self.last_cost

    def run_MPC_synchronous(
        self,
        k,
        xref,
        fsteps,
        l_fsteps_target,
        oRh,
        oTh,
        position,
        velocity,
        acceleration,
        jerk,
        dt_flying,
    ):
        """Run the MPC (synchronous version) to get the desired contact forces for the feet currently in stance phase

        Args:
            k (int): Number of inv dynamics iterations since the start of the simulation
            xref (12xN): Desired state vector for the whole prediction horizon
            fsteps (12xN array): the [x, y, z]^T desired position of each foot for each time step of the horizon
            l_fsteps_target (3x4 array) : [x, y, z]^T target position in local frame, to stop the optimisation of the feet location around it
        """

        # Run the MPC to get the reference forces and the next predicted state
        # Result is stored in mpc.f_applied, mpc.q_next, mpc.v_next

        if self.mpc_type == MPC_type.OSQP:
            # OSQP MPC
            self.mpc.run(np.int(k), xref.copy(), fsteps.copy())
        elif (
            self.mpc_type == MPC_type.CROCODDYL_PLANNER
        ):  # Add goal position to stop the optimisation
            # Crocoddyl MPC
            self.mpc.solve(
                k,
                xref.copy(),
                fsteps.copy(),
                l_fsteps_target,
                position,
                velocity,
                acceleration,
                jerk,
                oRh,
                oTh,
                dt_flying,
            )
        else:
            # Crocoddyl MPC
            self.mpc.solve(k, xref.copy(), fsteps.copy())

        # Output of the MPC
        if self.mpc_type == MPC_type.CROCODDYL_PLANNER:
            self.f_applied = self.mpc.get_latest_result(oRh, oTh)
        else:
            self.f_applied = self.mpc.get_latest_result()

        if self.mpc_type == MPC_type.OSQP:
            self.last_cost = self.mpc.retrieve_cost()

    def run_MPC_asynchronous(
        self,
        k,
        xref,
        fsteps,
        l_fsteps_target,
        oRh,
        oTh,
        position,
        velocity,
        acceleration,
        jerk,
        dt_flying,
    ):
        """Run the MPC (asynchronous version) to get the desired contact forces for the feet currently in stance phase

        Args:
            k (int): Number of inv dynamics iterations since the start of the simulation
            xref (12xN): Desired state vector for the whole prediction horizon
            fsteps (12xN array): the [x, y, z]^T desired position of each foot for each time step of the horizon
            params (object): stores parameters
            l_fsteps_target (3x4 array) : [x, y, z]^T target position in local frame, to stop the optimisation of the feet location around it
        """
        if k == 0:
            p = Process(
                target=self.create_MPC_asynchronous,
                args=(
                    self.newData,
                    self.newResult,
                    self.dataIn,
                    self.dataOut,
                    self.running,
                ),
            )
            p.start()

        self.compress_dataIn(
            k,
            xref,
            fsteps,
            l_fsteps_target,
            oRh,
            oTh,
            position,
            velocity,
            acceleration,
            jerk,
            dt_flying,
        )
        self.newData.value = True

    def create_MPC_asynchronous(self, newData, newResult, dataIn, dataOut, running):
        """Parallel process with an infinite loop that run the asynchronous MPC

        Args:
            newData (Value): shared variable that is true if new data is available, false otherwise
            newResult (Value): shared variable that is true if a new result is available, false otherwise
            dataIn (Array): shared array that contains the data the asynchronous MPC will use as inputs
            dataOut (Array): shared array that contains the result of the asynchronous MPC
            running (Value): shared variable to stop the infinite loop when set to False
        """
        while running.value:
            if newData.value:
                newData.value = False

                # Retrieve data thanks to the decompression function and reshape it
                if self.mpc_type != MPC_type.CROCODDYL_PLANNER:
                    k, xref, fsteps = self.decompress_dataIn(dataIn)
                else:
                    (
                        k,
                        xref,
                        fsteps,
                        l_fsteps_target,
                        oRh,
                        oTh,
                        position,
                        velocity,
                        acceleration,
                        jerk,
                        dt_flying,
                    ) = self.decompress_dataIn(dataIn)

                # Create the MPC object of the parallel process during the first iteration
                if k == 0:
                    if self.mpc_type == MPC_type.OSQP:
                        loop_mpc = qrw.MPC(self.params)
                    elif self.mpc_type == MPC_type.CROCODDYL_LINEAR:
                        loop_mpc = MPC_crocoddyl(self.params, mu=0.9, inner=False, linearModel=True)
                    elif self.mpc_type == MPC_type.CROCODDYL_NON_LINEAR:
                        loop_mpc = MPC_crocoddyl(self.params, mu=0.9, inner=False, linearModel=False)
                    elif self.mpc_type == MPC_type.CROCODDYL_PLANNER:
                        loop_mpc = MPC_crocoddyl_planner(self.params, mu=0.9, inner=False)
                    else:
                        self.mpc_type = MPC_type.OSQP
                        loop_mpc = qrw.MPC(self.params)

                # Run the asynchronous MPC with the data that as been retrieved
                if self.mpc_type == MPC_type.OSQP:
                    fsteps[np.isnan(fsteps)] = 0.0
                    loop_mpc.run(np.int(k), xref, fsteps)
                elif self.mpc_type == MPC_type.CROCODDYL_PLANNER:
                    loop_mpc.solve(
                        k,
                        xref.copy(),
                        fsteps.copy(),
                        l_fsteps_target.copy(),
                        position.copy(),
                        velocity.copy(),
                        acceleration.copy(),
                        jerk.copy(),
                        oRh.copy(),
                        oTh.copy(),
                        dt_flying.copy(),
                    )
                else:
                    loop_mpc.solve(k, xref.copy(), fsteps.copy())

                if self.mpc_type == MPC_type.CROCODDYL_PLANNER:
                    self.dataOut[:] = loop_mpc.get_latest_result(oRh, oTh).ravel(
                        order="F"
                    )
                else:
                    self.dataOut[:] = loop_mpc.get_latest_result().ravel(order="F")

                if self.mpc_type == MPC_type.OSQP:
                    self.cost.value = loop_mpc.retrieve_cost()

                newResult.value = True

    def compress_dataIn(
        self,
        k,
        xref,
        fsteps,
        l_fsteps_target,
        oRh,
        oTh,
        position,
        velocity,
        acceleration,
        jerk,
        dt_flying,
    ):
        """Compress data in a C-type structure that belongs to the shared memory to send data from the main control
        loop to the asynchronous MPC

        Args:
            k (int): Number of inv dynamics iterations since the start of the simulation
            fstep_planner (object): FootstepPlanner object of the control loop
            l_fsteps_target (3x4 array) : [x, y, z]^T target position in local frame, to stop the optimisation of the feet location around it
        """

        # Replace NaN values by 0.0 to be stored in C-type arrays
        fsteps[np.isnan(fsteps)] = 0.0

        # Compress data in the shared input array
        with self.dataIn.get_lock():
            if self.mpc_type == MPC_type.CROCODDYL_PLANNER:
                self.dataIn.k = k
                np.frombuffer(self.dataIn.xref).reshape((12, self.n_steps + 1))[
                    :, :
                ] = xref
                np.frombuffer(self.dataIn.fsteps).reshape((self.N_gait, 12))[
                    :, :
                ] = fsteps
                np.frombuffer(self.dataIn.l_fsteps_target).reshape((3, 4))[
                    :, :
                ] = l_fsteps_target
                np.frombuffer(self.dataIn.oRh).reshape((3, 3))[:, :] = oRh
                np.frombuffer(self.dataIn.oTh).reshape((3, 1))[:, :] = oTh
                np.frombuffer(self.dataIn.position).reshape((3, 4))[:, :] = position
                np.frombuffer(self.dataIn.velocity).reshape((3, 4))[:, :] = velocity
                np.frombuffer(self.dataIn.acceleration).reshape((3, 4))[
                    :, :
                ] = acceleration
                np.frombuffer(self.dataIn.jerk).reshape((3, 4))[:, :] = jerk
                np.frombuffer(self.dataIn.dt_flying).reshape(4)[:] = dt_flying

            else:
                self.dataIn.k = k
                np.frombuffer(self.dataIn.xref).reshape((12, self.n_steps + 1))[
                    :, :
                ] = xref
                np.frombuffer(self.dataIn.fsteps).reshape((self.N_gait, 12))[
                    :, :
                ] = fsteps

    def decompress_dataIn(self, dataIn):
        """Decompress data from a C-type structure that belongs to the shared memory to retrieve data from the main control
        loop in the asynchronous MPC

        Args:
            dataIn (Array): shared C-type structure that contains the data the asynchronous MPC will use as inputs
        """

        with dataIn.get_lock():
            if self.mpc_type == MPC_type.CROCODDYL_PLANNER:
                k = self.dataIn.k
                xref = np.frombuffer(self.dataIn.xref).reshape((12, self.n_steps + 1))
                fsteps = np.frombuffer(self.dataIn.fsteps).reshape((self.N_gait, 12))
                l_fsteps_target = np.frombuffer(self.dataIn.l_fsteps_target).reshape(
                    (3, 4)
                )
                oRh = np.frombuffer(self.dataIn.oRh).reshape((3, 3))
                oTh = np.frombuffer(self.dataIn.oTh).reshape((3, 1))
                position = np.frombuffer(self.dataIn.position).reshape((3, 4))
                velocity = np.frombuffer(self.dataIn.velocity).reshape((3, 4))
                acceleration = np.frombuffer(self.dataIn.acceleration).reshape((3, 4))
                jerk = np.frombuffer(self.dataIn.jerk).reshape((3, 4))
                dt_flying = np.frombuffer(self.dataIn.dt_flying).reshape(4)

                return (
                    k,
                    xref,
                    fsteps,
                    l_fsteps_target,
                    oRh,
                    oTh,
                    position,
                    velocity,
                    acceleration,
                    jerk,
                    dt_flying,
                )

            else:
                k = self.dataIn.k
                xref = np.frombuffer(self.dataIn.xref).reshape((12, self.n_steps + 1))
                fsteps = np.frombuffer(self.dataIn.fsteps).reshape((self.N_gait, 12))

                return k, xref, fsteps

    def convert_dataOut(self):
        """Return the result of the asynchronous MPC (desired contact forces) that is stored in the shared memory"""

        if (
            self.mpc_type == MPC_type.CROCODDYL_PLANNER
        ):  # Need more space to store optimized footsteps
            return np.array(self.dataOut[:]).reshape((32, -1), order="F")
        else:
            return np.array(self.dataOut[:]).reshape((24, -1), order="F")

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
        index = next(
            (
                idx
                for idx, val in np.ndenumerate(self.fsteps_future[:, 0])
                if val == 0.0
            ),
            0.0,
        )[0]

        # Create a new phase if needed or increase the last one by 1 step
        self.fsteps_future[index - 1, 0] += 1.0

        # Decrease the current phase by 1 step and delete it if it has ended
        if self.fsteps_future[0, 0] > 1.0:
            self.fsteps_future[0, 0] -= 1.0
        else:
            self.fsteps_future = np.roll(self.fsteps_future, -1, axis=0)
            self.fsteps_future[-1, :] = np.zeros((13,))

        return 0

    def stop_parallel_loop(self):
        """Stop the infinite loop in the parallel process to properly close the simulation"""

        self.running.value = False

        return 0
