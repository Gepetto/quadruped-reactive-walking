# coding: utf8

import numpy as np
import matplotlib.pylab as plt
from TSID_Debug_controller_four_legs_fb_vel import controller, dt
from main import run_scenario
from IPython import embed

envID = 0
velID = 0

dt_mpc = 0.02  # Time step of the MPC
k_mpc = int(dt_mpc / dt)  # dt is dt_tsid, defined in the TSID controller script
t = 0.0  # Time
n_periods = 1  # Number of periods in the prediction horizon
T_gait = 0.48  # Duration of one gait period
N_SIMULATION = 500  # number of simulated TSID time steps

# Which MPC solver you want to use
# True to have PA's MPC, to False to have Thomas's MPC
type_MPC = True

#################
# RUN SCENARIOS #
#################

# Run a scenario and retrieve data thanks to the logger
result_logger = run_scenario(envID, velID, dt_mpc, k_mpc, t, n_periods, T_gait, N_SIMULATION, type_MPC)

# Display what has been logged by the logger
# result_logger.plot_graphs(enable_multiprocessing=False)

# Run a scenario and retrieve data thanks to the logger
result_logger2 = run_scenario(envID, velID, dt_mpc, k_mpc, t, n_periods, T_gait, N_SIMULATION, type_MPC)

embed()

quit()
