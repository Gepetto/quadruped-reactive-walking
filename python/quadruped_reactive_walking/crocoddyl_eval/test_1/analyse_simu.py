# coding: utf8
# coding: utf8

import sys
import os 
from sys import argv
sys.path.insert(0, os.getcwd()) # adds current directory to python path


import numpy as np
import matplotlib.pylab as plt
import utils
import time
# from TSID_Debug_controller_four_legs_fb_vel import controller, dt
import MPC_Wrapper 
import FootstepPlanner

####################
# Recovery of Data
####################

folder_name = "log_eval/"
pathIn = "crocoddyl_eval/test_1/"
ddp_xs = np.load(pathIn + folder_name + "ddp_xs.npy")
ddp_us = np.load(pathIn + folder_name + "ddp_us.npy")
osqp_xs = np.load(pathIn + folder_name + "osqp_xs.npy")
osqp_us = np.load(pathIn + folder_name + "osqp_us.npy")
fsteps = np.load(pathIn + folder_name + "fsteps.npy")
xref = np.load(pathIn + folder_name + "xref.npy") 
l_feet = np.load(pathIn + folder_name + "l_feet.npy") 


####################
# Iteration 
####################

iteration = 202 

dt_mpc = 0.02  # Time step of the MPC
dt = 0.004
k_mpc = int(dt_mpc / dt)  # dt is dt_tsid, defined in the TSID controller script
n_periods = 1  # Number of periods in the prediction horizon
T_gait = 0.64  # Duration of one gait period

# Create footstep planner object
fstep_planner = FootstepPlanner.FootstepPlanner(dt_mpc, n_periods, T_gait)
fstep_planner.xref = xref[:,:, iteration]
fstep_planner.fsteps = fsteps[:,:,iteration]
fstep_planner.fsteps_mpc = fstep_planner.fsteps.copy()

######################################
#  Relaunch DDP to adjust the gains  #
######################################

Relaunch_DDP = True

enable_multiprocessing = False
mpc_wrapper_ddp = MPC_Wrapper.MPC_Wrapper(False, dt_mpc, fstep_planner.n_steps,
                                        k_mpc, fstep_planner.T_gait, enable_multiprocessing)

w_x = 0.2
w_y = 0.2
w_z = 2
w_roll = 0.8 # diff from MPC_crocoddyl
w_pitch = 1.5 # diff from MPC_crocoddyl
w_yaw = 0.11
w_vx =  1*np.sqrt(w_x)
w_vy =  2*np.sqrt(w_y)
w_vz =  1*np.sqrt(w_z)
w_vroll =  0.05*np.sqrt(w_roll)
w_vpitch =  0.05*np.sqrt(w_pitch)
w_vyaw =  0.03*np.sqrt(w_yaw)

# Weight Vector : State 
mpc_wrapper_ddp.mpc.stateWeight = np.array([w_x,w_y,w_z,w_roll,w_pitch,w_yaw,
                            w_vx,w_vy,w_vz,w_vroll,w_vpitch,w_vyaw])

# Weight Vector : Force Norm
mpc_wrapper_ddp.mpc.forceWeights = np.array(4*[0.01,0.01,0.01])

# Weight Vector : Friction cone cost
mpc_wrapper_ddp.mpc.frictionWeights = 0.5

# Update weights inside the models 
mpc_wrapper_ddp.mpc.updateActionModel()

# Run ddp solver
mpc_wrapper_ddp.mpc.max_iteration = 10 # The solver is not warm started here
mpc_wrapper_ddp.solve(0,fstep_planner)

#############
#  Plot     #
#############

# Predicted evolution of state variables
l_t = np.linspace(dt_mpc, T_gait, np.int(T_gait/dt_mpc))
l_str = ["X_osqp", "Y_osqp", "Z_osqp", "Roll_osqp", "Pitch_osqp", "Yaw_osqp", "Vx_osqp", "Vy_osqp", "Vz_osqp", "VRoll_osqp", "VPitch_osqp", "VYaw_osqp"]
l_str2 = ["X_ddp", "Y_ddp", "Z_ddp", "Roll_ddp", "Pitch_ddp", "Yaw_ddp", "Vx_ddp", "Vy_ddp", "Vz_ddp", "VRoll_ddp", "VPitch_ddp", "VYaw_ddp"]

index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
plt.figure()
for i in range(12):
    plt.subplot(3, 4, index[i])
    
    pl1, = plt.plot(l_t, ddp_xs[i,:,iteration], linewidth=2, marker='x')
    pl2, = plt.plot(l_t, osqp_xs[i,:,iteration], linewidth=2, marker='x')

    if Relaunch_DDP : 
        pl3, = plt.plot(l_t, mpc_wrapper_ddp.mpc.get_xrobot()[i,:], linewidth=2, marker='x')
        plt.legend([pl1,pl2,pl3] , [l_str2[i] , l_str[i], "ddp_redo" ])
    
    else : 
        plt.legend([pl1,pl2] , [l_str2[i] , l_str[i] ])
    

    

# Desired evolution of contact forces
l_t = np.linspace(dt_mpc, T_gait, np.int(T_gait/dt_mpc))
l_str = ["FL_X_osqp", "FL_Y_osqp", "FL_Z_osqp", "FR_X_osqp", "FR_Y_osqp", "FR_Z_osqp", "HL_X_osqp", "HL_Y_osqp", "HL_Z_osqp", "HR_X_osqp", "HR_Y_osqp", "HR_Z_osqp"]
l_str2 = ["FL_X_ddp", "FL_Y_ddp", "FL_Z_ddp", "FR_X_ddp", "FR_Y_ddp", "FR_Z_ddp", "HL_X_ddp", "HL_Y_ddp", "HL_Z_ddp", "HR_X_ddp", "HR_Y_ddp", "HR_Z_ddp"]
index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
plt.figure()
for i in range(12):
    plt.subplot(3, 4, index[i])
    pl1, = plt.plot(l_t, ddp_us[i,:,iteration], linewidth=2, marker='x')
    pl2, = plt.plot(l_t, osqp_us[i,:,iteration], linewidth=2, marker='x')
   
    if Relaunch_DDP : 
        pl3, = plt.plot(l_t, mpc_wrapper_ddp.mpc.get_fpredicted()[i,:], linewidth=2, marker='x')
        plt.legend([pl1,pl2,pl3] , [l_str2[i] , l_str[i], "ddp_redo" ])
    
    else : 
        plt.legend([pl1,pl2] , [l_str2[i] , l_str[i] ])
    

plt.show(block=True)
