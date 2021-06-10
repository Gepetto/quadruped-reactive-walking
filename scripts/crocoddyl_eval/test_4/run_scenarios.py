# coding: utf8

import time
import multiprocessing as mp
import numpy as np
from main_solo12_control import control_loop

#################
# RUN SCENARIOS #
#################


def run_simu_wyaw(speed):
    desired_speed = np.zeros(6)
    desired_speed[0] = speed[0]
    desired_speed[5] = speed[1]

    return control_loop("test", None, desired_speed)


def run_simu_vy(speed):
    desired_speed = np.zeros(6)
    desired_speed[0] = speed[0]
    desired_speed[1] = speed[1]

    return control_loop("test", None, desired_speed)


"""
# List of arguments
X = np.linspace(-1, 1, 25)
W = np.linspace(-2.2, 2.2, 25)
list_param = []
for i in range(X.size):
    for j in range(W.size):
        list_param.append([X[i], W[j]])


start_time = time.time()

# Multiprocess lauch : cpu -1 --> avoid freeze
with mp.Pool(mp.cpu_count()-1) as pool:
    res = pool.map(run_simu_wyaw, list_param)

print("Temps d execution ddp ywaw: %s secondes ---" % (time.time() - start_time))
mem_time = time.time() - start_time

# Logger of the results
pathIn = "crocoddyl_eval/test_4/log_eval/"
print("Saving logs...")
np.save(pathIn + "results_wyaw_all_true.npy", np.array(res))
"""

###########################
# New simulation with osqp,
# Vx and angular yaw :
###########################
# type_MPC = True

# #time
# start_time = time.time()

# # Multiprocess lauch : cpu -1 --> avoid freeze
# with mp.Pool(mp.cpu_count()-1) as pool :
#     res1 = pool.map(run_simu_wyaw,list_param)

# mem_time1 = time.time() - start_time
# print("Temps d execution osqp: %s secondes ---" % (time.time() - start_time))

# # Logger of the results
# pathIn = "crocoddyl_eval/test_4/log_eval/"
# print("Saving logs...")
# np.save(pathIn +  "results_osqp_wyaw.npy" , np.array(res1) )


###########################
# New simulation with osqp,
# Vx and Vy :
###########################

# List of arguments
X = np.hstack((np.linspace(-1.4, 1.4, 15)))
Y = np.hstack((np.linspace(-1.4, 1.4, 15)))
list_param = []
for i in range(X.size):
    for j in range(Y.size):
        if (np.abs(X[i]) >= 0.79) and (np.abs(Y[j]) >= 0.79):
            list_param.append([X[i], Y[j]])

"""from IPython import embed
embed()"""

type_MPC = False

# time
start_time = time.time()

# Multiprocess lauch : cpu -1 --> avoid freeze
with mp.Pool(mp.cpu_count()-1) as pool:
    res2 = pool.map(run_simu_vy, list_param)

# print("Temps d execution ddp wyaw: %s secondes ---" % (mem_time))
print("Temps d execution osqp vy: %s secondes ---" % (time.time() - start_time))

# Logger of the results
pathIn = "crocoddyl_eval/test_4/log_eval/"
print("Saving logs...")
np.save(pathIn + "results_osqp_vy.npy", np.array(res2))
# np.save(pathIn + "results_vy_all_true.npy", np.array(res2))

quit()
