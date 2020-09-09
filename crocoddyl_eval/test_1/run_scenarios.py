# coding: utf8
from IPython import embed
from crocoddyl_eval.test_1.main import run_scenario
from TSID_Debug_controller_four_legs_fb_vel import controller, dt
import matplotlib.pylab as plt
import numpy as np
import sys
import os
from sys import argv
sys.path.insert(0, os.getcwd())  # adds current directory to python path


envID = 0
velID = 0

dt_mpc = 0.02  # Time step of the MPC
k_mpc = int(dt_mpc / dt)  # dt is dt_tsid, defined in the TSID controller script
t = 0.0  # Time
n_periods = 1  # Number of periods in the prediction horizon
T_gait = 0.64  # Duration of one gait period
N_SIMULATION = 2000  # number of simulated TSID time steps

# Which MPC solver you want to use
# True to have PA's MPC, to False to have Thomas's MPC
type_MPC = False

# Whether PyBullet feedback is enabled or not
pyb_feedback = True

# Whether we are working with solo8 or not
on_solo8 = False

# If True the ground is flat, otherwise it has bumps
use_flat_plane = True

# If we are using a predefined reference velocity (True) or a joystick (False)
predefined_vel = True

# Enable or disable PyBullet GUI
enable_pyb_GUI = False

#################
# RUN SCENARIOS #
#################

# Run a scenario and retrieve data thanks to the logger
logger_ddp, logger_osqp = run_scenario(envID, velID, dt_mpc, k_mpc, t, n_periods,
                                       T_gait, N_SIMULATION, type_MPC, pyb_feedback,
                                       on_solo8, use_flat_plane, predefined_vel, enable_pyb_GUI)

#################
# RECORD LOGGERS
#################

pathIn = "crocoddyl_eval/test_1/log_eval/"

print("Saving logs...")

np.save(pathIn + "ddp_xs.npy", logger_ddp.pred_trajectories)
np.save(pathIn + "ddp_us.npy", logger_ddp.pred_forces)

np.save(pathIn + "osqp_xs.npy", logger_osqp.pred_trajectories)
np.save(pathIn + "osqp_us.npy", logger_osqp.pred_forces)

np.save(pathIn + "l_feet.npy", logger_ddp.feet_pos)
np.save(pathIn + "fsteps.npy", logger_ddp.fsteps)
np.save(pathIn + "xref.npy", logger_ddp.xref)

logger_ddp.plot_state()
plt.show(block=True)

quit()
