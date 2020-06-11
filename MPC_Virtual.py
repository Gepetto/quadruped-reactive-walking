# coding: utf8

import MPC_Wrapper


class MPC_Virtual():

    def __init__(self, MPC_type, dt_mpc, sequencer, k_mpc):

        if MPC_type:

            # Enable/Disable multiprocessing (MPC running in a parallel process)
            enable_multiprocessing = False
            self.solver = MPC_Wrapper.MPC_Wrapper(dt_mpc, sequencer.S.shape[0], k_mpc,
                                                  multiprocessing=enable_multiprocessing)

        else:

            # Thomas' MPC
            a = 0

        self.solve = self.solver.solve
        self.get_latest_result = self.solver.get_latest_result
