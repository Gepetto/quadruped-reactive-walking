# coding: utf8
import sys
import os 
from sys import argv
sys.path.insert(0, os.getcwd()) # adds current directory to python path

import numpy as np
import matplotlib.pylab as plt
from TSID_Debug_controller_four_legs_fb_vel import controller, dt
from crocoddyl_eval.test_4.main import run_scenario 
from IPython import embed
import Joystick

import multiprocessing as mp
import time

envID = 0
velID = 0

dt_mpc = 0.02  # Time step of the MPC
k_mpc = int(dt_mpc / dt)  # dt is dt_tsid, defined in the TSID controller script
t = 0.0  # Time
n_periods = 1  # Number of periods in the prediction horizon
T_gait = 0.64  # Duration of one gait period
N_SIMULATION = 5000  # number of simulated TSID time steps

# Which MPC solver you want to use
# True to have PA's MPC, to False to have Thomas's MPC
type_MPC = False 

# Whether PyBullet feedback is enabled or not
pyb_feedback = True

#################
# RUN SCENARIOS #
#################

def run_simu_wyaw(speed) : 
    desired_speed = np.zeros(6)
    desired_speed[0] = speed[0]
    desired_speed[5] = speed[1]

    return run_scenario(envID, velID, dt_mpc, k_mpc, t, n_periods, T_gait, N_SIMULATION, type_MPC, pyb_feedback , desired_speed)

def run_simu_vy(speed) : 
    desired_speed = np.zeros(6)
    desired_speed[0] = speed[0]
    desired_speed[1] = speed[1]

    return run_scenario(envID, velID, dt_mpc, k_mpc, t, n_periods, T_gait, N_SIMULATION, type_MPC, pyb_feedback , desired_speed)

# List of arguments 
X = np.linspace(-1,1,25)
W = np.linspace(-2.2,2.2,25)
list_param = []
for i in range(X.size) : 
    for j in range(W.size) : 
        list_param.append([X[i] , W[j]])



start_time = time.time()

# Multiprocess lauch : cpu -1 --> avoid freeze
with mp.Pool(mp.cpu_count()-1) as pool : 
    res = pool.map(run_simu_wyaw,list_param)

print("Temps d execution ddp ywaw: %s secondes ---" % (time.time() - start_time)) 
mem_time = time.time() - start_time

# Logger of the results
pathIn = "crocoddyl_eval/test_4/log_eval/"
print("Saving logs...")
np.save(pathIn +  "results_wyaw_all_true.npy" , np.array(res) )

# ###########################
# # New simulation with osqp, 
# #Vx and angular yaw : 
# ###########################
# # type_MPC = True

# # #time
# # start_time = time.time()

# # # Multiprocess lauch : cpu -1 --> avoid freeze
# # with mp.Pool(mp.cpu_count()-1) as pool : 
# #     res1 = pool.map(run_simu_wyaw,list_param)

# # mem_time1 = time.time() - start_time
# # print("Temps d execution osqp: %s secondes ---" % (time.time() - start_time)) 

# # # Logger of the results
# # pathIn = "crocoddyl_eval/test_4/log_eval/"
# # print("Saving logs...")
# # np.save(pathIn +  "results_osqp_wyaw.npy" , np.array(res1) )


###########################
# New simulation with osqp, 
#Vx and angular yaw : 
###########################

# List of arguments 
X = np.linspace(-1,1,25)
Y = np.linspace(-1,1,25)
list_param = []
for i in range(X.size) : 
    for j in range(Y.size) : 
        list_param.append([X[i] , Y[j]])

type_MPC = False

#time
start_time = time.time()

# Multiprocess lauch : cpu -1 --> avoid freeze
with mp.Pool(mp.cpu_count()-1) as pool : 
    res2 = pool.map(run_simu_vy,list_param)

# print("Temps d execution ddp wyaw: %s secondes ---" % (mem_time)) 
print("Temps d execution ddp vy: %s secondes ---" % (time.time() - start_time)) 

# Logger of the results
pathIn = "crocoddyl_eval/test_4/log_eval/"
print("Saving logs...")
np.save(pathIn +  "results_vy_all_true.npy" , np.array(res2) )

quit()