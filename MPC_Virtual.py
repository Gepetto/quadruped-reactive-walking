# coding: utf8

import MPC_Wrapper
import crocoddyl_class.MPC_crocoddyl as MPC_crocoddyl


class MPC_Virtual():
    """Wrapper for different MPC solvers. Only two functions are used rom the main loop:
    solve() is used to run one iteration of the MPC
    get_latest_result() is used to retrieve the reference contact forces
    Automatically call the solver chosen by the user

    Args:
        mpc_type (bool): True to have PA's MPC, False to have Thomas's MPC
        dt_mpc (float): Duration of one time step of the MPC
        n_steps (int): Number of time steps in one gait cycle
        k_mpc (int): Number of inv dyn time step for one iteration of the MPC
        T_gait (float): Duration of one period of gait
    """

    def __init__(self, mpc_type, dt_mpc, n_steps, k_mpc, T_gait):

        if mpc_type:

            # Enable/Disable multiprocessing (MPC running in a parallel process)
            enable_multiprocessing = False
            self.solver = MPC_Wrapper.MPC_Wrapper(dt_mpc, n_steps, k_mpc, T_gait,
                                                  multiprocessing=enable_multiprocessing)

        else:

            # Crocoddyl MPC
            self.solver = MPC_crocoddyl.MPC_crocoddyl(dt_mpc, T_gait , 1 , True)  # mu = 1 & inner = True -> mu = 0.7

        self.solve = self.solver.solve
        self.get_latest_result = self.solver.get_latest_result
