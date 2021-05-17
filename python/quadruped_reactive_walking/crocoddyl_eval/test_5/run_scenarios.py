# coding: utf8
import sys
import os 
from sys import argv
sys.path.insert(0, os.getcwd()) # adds current directory to python path

import numpy as np
import matplotlib.pylab as plt
from TSID_Debug_controller_four_legs_fb_vel import controller, dt
from crocoddyl_eval.test_5.main import run_scenario
from IPython import embed

envID = 0
velID = 0

dt_mpc = 0.02  # Time step of the MPC
k_mpc = int(dt_mpc / dt)  # dt is dt_tsid, defined in the TSID controller script
t = 0.0  # Time
n_periods = 1  # Number of periods in the prediction horizon
T_gait = 0.32  # Duration of one gait period
N_SIMULATION = 2000  # number of simulated TSID time steps

# Which MPC solver you want to use
# True to have PA's MPC, to False to have Thomas's MPC
type_MPC = True

# Whether PyBullet feedback is enabled or not
pyb_feedback = True

# Desired speed
# increasing by 0.1m.s-1 each second, and then 10s of simulation
desired_speed =  [0.3,  0.0 , 0. , 0. , 0. ,0.0]

#################
# RUN SCENARIOS #
#################

# Run a scenario and retrieve data thanks to the logger
logger_ddp  = run_scenario(envID, velID, dt_mpc, k_mpc, t, n_periods, T_gait, N_SIMULATION, type_MPC, pyb_feedback , desired_speed)

#################
# RECORD LOGGERS
#################

pathIn = "crocoddyl_eval/test_5/log_eval/"

print("Saving logs...")

np.save(pathIn +  "ddp_xs.npy" , logger_ddp.pred_trajectories )
np.save(pathIn +  "ddp_us.npy" , logger_ddp.pred_forces )

np.save(pathIn +  "o_feet.npy" , logger_ddp.feet_pos )
np.save(pathIn +  "fsteps.npy" , logger_ddp.fsteps )
np.save(pathIn +  "xref.npy" , logger_ddp.xref )
np.save(pathIn +  "oC.npy" , logger_ddp.oC )
np.save(pathIn +  "o_shoulders.npy" , logger_ddp.o_shoulders )

logger_ddp.plot_state()
logger_ddp.plot_footsteps()
plt.show(block=True)

quit()