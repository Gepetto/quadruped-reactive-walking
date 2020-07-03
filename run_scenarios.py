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
T_gait = 0.72  # Duration of one gait period
N_SIMULATION = 7000  # number of simulated TSID time steps

# Which MPC solver you want to use
# True to have PA's MPC, to False to have Thomas's MPC
type_MPC = True

# Whether PyBullet feedback is enabled or not
pyb_feedback = True

#################
# RUN SCENARIOS #
#################

# Run a scenario and retrieve data thanks to the logger
result_logger1 = run_scenario(envID, velID, dt_mpc, k_mpc, t, n_periods, T_gait, N_SIMULATION, type_MPC, pyb_feedback)

# Run a scenario and retrieve data thanks to the logger
result_logger2 = run_scenario(envID, velID, dt_mpc, k_mpc, t, n_periods, T_gait, N_SIMULATION, type_MPC, False)

# Display what has been logged by the loggers
result_logger1.plot_graphs(enable_multiprocessing=False, show_block=False)
result_logger2.plot_graphs(enable_multiprocessing=False)

#result_logger1.plot_state()
#result_logger2.plot_state()
plt.show(block=True)

embed()

quit()
