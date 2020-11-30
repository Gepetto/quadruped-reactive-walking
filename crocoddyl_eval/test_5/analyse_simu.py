# coding: utf8
# coding: utf8

import sys
import os 
from sys import argv
sys.path.insert(0, os.getcwd()) # adds current directory to python path


import numpy as np
import matplotlib.pylab as plt ; plt.ion()
import utils
import time
from TSID_Debug_controller_four_legs_fb_vel import controller, dt
from crocoddyl_class.MPC_crocoddyl_2 import MPC_crocoddyl_2
import MPC_Wrapper 
import FootstepPlanner

####################
# Recovery of Data
####################

folder_name = "log_eval/"
pathIn = "crocoddyl_eval/test_5/"
ddp_xs = np.load(pathIn + folder_name + "ddp_xs.npy")
ddp_us = np.load(pathIn + folder_name + "ddp_us.npy")
fsteps = np.load(pathIn + folder_name + "fsteps.npy")
xref = np.load(pathIn + folder_name + "xref.npy") 
# l_feet = np.load(pathIn + folder_name + "l_feet.npy") 


####################
# Iteration 
####################

iteration = 158

dt_mpc = 0.02  # Time step of the MPC
k_mpc = int(dt_mpc / dt)  # dt is dt_tsid, defined in the TSID controller script
n_periods = 1  # Number of periods in the prediction horizon
T_gait = 0.32  # Duration of one gait period

# Create footstep planner object
fstep_planner = FootstepPlanner.FootstepPlanner(dt_mpc, n_periods, T_gait)
fstep_planner.xref = xref[:,:, iteration  ]
fstep_planner.fsteps = fsteps[:,:,iteration ]
fstep_planner.fsteps_mpc = fstep_planner.fsteps.copy()

######################################
#  Relaunch DDP to adjust the gains  #
######################################

Relaunch_DDP = True

enable_multiprocessing = False
mpc_wrapper_ddp = MPC_Wrapper.MPC_Wrapper(False, dt_mpc, fstep_planner.n_steps,
                                        k_mpc, fstep_planner.T_gait, enable_multiprocessing)

mpc_wrapper_ddp_2 = MPC_crocoddyl_2( dt = dt_mpc , T_mpc = T_gait , mu = 0.9, inner = False, linearModel = False  , n_period = 1 , dt_tsid = dt)


# # Run ddp solver
mpc_wrapper_ddp.mpc.max_iteration = 10 # The solver is not warm started here
mpc_wrapper_ddp.solve(1,fstep_planner)

k_tsid = 156
# Reference velocity in local frame
v_ref = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
v_ref[0,0] = xref[6,1,iteration]
v_ref[1,0] = xref[7,1,iteration]
v_ref[2,0] = xref[8,1,iteration]
v_ref[3,0] = xref[9,1,iteration]
v_ref[4,0] = xref[10,1,iteration]
v_ref[5,0] = xref[11,1,iteration]
x0 = xref[:,0,iteration]

mpc_wrapper_ddp_2.updateProblem(k_tsid,fstep_planner.fsteps , fstep_planner.xref , np.array([x0[:3]]).T, np.array([x0[3:6]]).T , np.array([x0[6:9]]).T,np.array([x0[9:12]]).T, v_ref)
mpc_wrapper_ddp_2.ddp.solve([],[],50)

#############
#  Plot     #
#############

# Predicted evolution of state variables
l_t = np.linspace(dt_mpc, n_periods*T_gait, np.int(n_periods*(T_gait/dt_mpc)))
l_str2 = ["X_ddp", "Y_ddp", "Z_ddp", "Roll_ddp", "Pitch_ddp", "Yaw_ddp", "Vx_ddp", "Vy_ddp", "Vz_ddp", "VRoll_ddp", "VPitch_ddp", "VYaw_ddp"]
Xs = np.array(mpc_wrapper_ddp_2.ddp.xs).T
index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
plt.figure()
for i in range(12):
    plt.subplot(3, 4, index[i])
    
    pl1, = plt.plot(l_t, ddp_xs[i,:,iteration], linewidth=2, marker='x')
    pl2, = plt.plot(mpc_wrapper_ddp_2.dt_vector ,Xs[i,:] , linewidth=2, marker='x' )

    plt.legend([pl1,pl2] , [l_str2[i] , "2"])

mu = 0.9
dt_vector2 = np.zeros(len(mpc_wrapper_ddp_2.dt_vector) - 1)
dt_vector2[:] = mpc_wrapper_ddp_2.dt_vector[:len(mpc_wrapper_ddp_2.dt_vector) - 1]
Us = np.array(mpc_wrapper_ddp_2.ddp.us).T
# Desired evolution of contact forces
l_t = np.linspace(0, n_periods*T_gait - dt_mpc, np.int(n_periods*(T_gait/dt_mpc)))
l_str2 = ["FL_X_ddp", "FL_Y_ddp", "FL_Z_ddp", "FR_X_ddp", "FR_Y_ddp", "FR_Z_ddp", "HL_X_ddp", "HL_Y_ddp", "HL_Z_ddp", "HR_X_ddp", "HR_Y_ddp", "HR_Z_ddp"]
index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
plt.figure()
for i in range(12):
    plt.subplot(3, 4, index[i])
    pl1, = plt.plot(l_t, ddp_us[i,:,iteration], linewidth=2, marker='x')
    pl2, = plt.plot(dt_vector2 ,Us[i,:] , linewidth=2, marker='x' )    
    plt.legend([pl1,pl2] , [l_str2[i] , "2"])
    

plt.show(block=True)
