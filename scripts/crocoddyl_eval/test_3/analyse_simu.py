# coding: utf8
# coding: utf8

import sys
import os 
from sys import argv
sys.path.insert(0, os.getcwd()) # adds current directory to python path

import random
import numpy as np
import matplotlib.pylab as plt
import utils
import time
from TSID_Debug_controller_four_legs_fb_vel import controller, dt
import MPC_Wrapper 
import FootstepPlanner
from crocoddyl_class.MPC_crocoddyl_planner import *
####################
# Recovery of Data
####################

folder_name = "log_eval/"
pathIn = "crocoddyl_eval/test_3/"
ddp_xs = np.load(pathIn + folder_name + "ddp_xs.npy")
ddp_us = np.load(pathIn + folder_name + "ddp_us.npy")
ddp_fsteps = np.load(pathIn + folder_name + "ddp_fsteps.npy")
ddp_gait = np.load(pathIn + folder_name + "ddp_gait.npy")
l_feet_local = np.load(pathIn + folder_name + "l_feet_local.npy") 

osqp_xs = np.load(pathIn + folder_name + "osqp_xs.npy")
osqp_us = np.load(pathIn + folder_name + "osqp_us.npy")
fsteps = np.load(pathIn + folder_name + "fsteps.npy")
xref = np.load(pathIn + folder_name + "xref.npy") 
l_feet = np.load(pathIn + folder_name + "l_feet.npy") 


####################
# Iteration 
####################

iteration = 172
dt_mpc = 0.02  # Time step of the MPC
k_mpc = int(dt_mpc / dt)  # dt is dt_tsid, defined in the TSID controller script
n_periods = 1  # Number of periods in the prediction horizon
T_gait = 0.64  # Duration of one gait period

# Create footstep planner object
fstep_planner = FootstepPlanner.FootstepPlanner(dt_mpc, n_periods, T_gait)
fstep_planner.xref = xref[:,:, iteration]
fstep_planner.fsteps = fsteps[:,:,iteration]
fstep_planner.fsteps_mpc = fstep_planner.fsteps.copy()
gait_matrix = ddp_gait[:,:,iteration]

######################################
#  Relaunch DDP to adjust the gains  #
######################################
Relaunch_DDP = False

# Initialization of the crocoddyl gait problem

# Ref problem

mpc_planner = MPC_crocoddyl_planner(dt = dt_mpc , T_mpc = T_gait)
# Weights on the shoulder term : term 1 
mpc_planner.shoulderWeights = np.array(4*[1,10])

# Weights on the previous position predicted : term 2 
mpc_planner.lastPositionWeights = np.full(8,0.0)

# Weight on the step command
mpc_planner.stepWeights = np.full(4,10)



for i in range(iteration  +  1) :

    mpc_planner.updateProblem(i, xref[:,:, i] , l_feet_local[:,:, i ] )
    mpc_planner.ddp.solve()
    mpc_planner.get_fsteps()

   



# xref_test =  np.zeros((12,int(T_gait/dt_mpc) + 1))
# xref_test[2,:] = xref[2,5, iteration]
# xref_test[6,0] = 0.4
# xref_test[0,0] = 0.0
# mpc_planner.updateProblem(iteration, xref_test[:,: ] , l_feet_local[:,:, iteration ] )
# mpc_planner.ddp.solve()
# mpc_planner.get_fsteps()

# gaitProblem.updateProblem(iteration, xref[:,:, i],  l_feet_local[:,:, iteration ])
# gaitProblem.ddp.solve()
# gaitProblem.get_fsteps()

# print(mpc_planner.gait)
# for elt in mpc_planner.ListAction : 
#     print(elt.__class__.__name__)
#     if(elt.__class__.__name__ != "ActionModelQuadrupedStep") : 
#         print(elt.stateWeights)
#         print(elt.forceWeights)
#         print(elt.frictionWeights)
#         print(elt.lastPositionWeights )
#         print(elt.shoulderWeights ) 
#     else : 
#         print(elt.lastPositionWeights )
#         print(elt.shoulderWeights ) 
#         print(elt.stateWeights)
#         print(elt.stepWeights)


# ################################################
# ## CHECK DERIVATIVE WITH NUMDDIFF 
# #################################################


# model_diff = crocoddyl.ActionModelNumDiff(gaitProblemCpp.ListAction[0])
# data = model_diff.createData()
# N_trial = 100
# epsilon = 0.001
# a = -1 
# b = 1

# # RUN CALC DIFF
# def run_calcDiff_numDiff(epsilon) :
#   Lx = 0
#   Lx_err = 0
#   Lu = 0
#   Lu_err = 0
#   Lxx = 0
#   Lxx_err = 0
#   Luu = 0
#   Luu_err = 0
#   Fx = 0
#   Fx_err = 0 
#   Fu = 0
#   Fu_err = 0    

#   for k in range(N_trial):    


#     x = a + (b-a)*np.random.rand(20)
#     u = a + (b-a)*np.random.rand(12)
#     N_model = random.randint(0,16)

#     while (mpc_planner.ListAction[N_model].__class__.__name__ == "ActionModelQuadrupedStep") : 
#         N_model = random.randint(0,16)

#     # Run calc & calcDiff function
#     actionCpp = mpc_planner.ListAction[N_model]
#     model_diff = crocoddyl.ActionModelNumDiff(actionCpp)

#     dataCpp = mpc_planner.ListAction[N_model].createData()
#     data = model_diff.createData()

#     model_diff.calc(data , x , u )
#     model_diff.calcDiff(data , x , u )


#     actionCpp.calc(dataCpp ,x , u )
#     actionCpp.calcDiff(dataCpp ,x , u )

    
#     Lx +=  np.sum( abs((data.Lx - dataCpp.Lx )) >= epsilon  ) 
#     Lx_err += np.sum( abs((data.Lx - dataCpp.Lx )) )  

#     Lu +=  np.sum( abs((data.Lu - dataCpp.Lu )) >= epsilon  ) 
#     Lu_err += np.sum( abs((data.Lu - dataCpp.Lu )) )  

#     Lxx +=  np.sum( abs((data.Lxx - dataCpp.Lxx )) >= epsilon  ) 
#     Lxx_err += np.sum( abs((data.Lxx - dataCpp.Lxx )) )  

#     Luu +=  np.sum( abs((data.Luu - dataCpp.Luu )) >= epsilon  ) 
#     Luu_err += np.sum( abs((data.Luu - dataCpp.Luu )) ) 

#     Fx +=  np.sum( abs((data.Fx - dataCpp.Fx )) >= epsilon  ) 
#     Fx_err += np.sum( abs((data.Fx - dataCpp.Fx )) )  

#     Fu +=  np.sum( abs((data.Fu - dataCpp.Fu )) >= epsilon  ) 
#     Fu_err += np.sum( abs((data.Fu - dataCpp.Fu )) )  
  
#   Lx_err = Lx_err /N_trial
#   Lu_err = Lu_err/N_trial
#   Lxx_err = Lxx_err/N_trial    
#   Luu_err = Luu_err/N_trial
#   Fx_err = Fx_err/N_trial
#   Fu_err = Fu_err/N_trial
  
#   return Lx , Lx_err , Lu , Lu_err , Lxx , Lxx_err , Luu , Luu_err, Fx, Fx_err, Fu , Fu_err


# Lx , Lx_err , Lu , Lu_err , Lxx , Lxx_err , Luu , Luu_err , Fx, Fx_err, Fu , Fu_err = run_calcDiff_numDiff(epsilon)

# print("\n \n ---------- Check derivative with NumDiff method : cpp check -------------------")
# print("\n------------CalcDiff Function :-----------")
# if Lx == 0:  print("Lx : OK    (error : %f)" %Lx_err)
# else :     print("Lx : NOT OK !!!    (error : %f)" %Lx_err )

# if Lu == 0:  print("Lu : OK    (error : %f)" %Lu_err)
# else :     print("Lu : NOT OK !!!    (error : %f)" %Lu_err)

# if Lxx == 0:  print("Lxx : OK    (error : %f)" %Lxx_err)
# else :     print("Lxx : NOT OK !!!   (error : %f)" %Lxx_err)

# if Luu == 0:  print("Luu : OK    (error : %f)" %Luu_err)
# else :     print("Luu : NOT OK !!!   (error : %f)" %Luu_err)

# if Fx == 0:  print("Fx : OK    (error : %f)" %Fx_err)
# else :     print("Fx : NOT OK !!!   (error : %f)" %Fx_err)

# if Fu == 0:  print("Fu : OK    (error : %f)" %Fu_err)
# else :     print("Fu : NOT OK !!!   (error : %f)" %Fu_err)


# if Lx == 0 and Lu == 0 and Lxx == 0 and Luu == 0 and Fu == 0 and Fx == 0: print("\n Calc function : OK")
# else : print("\n Calc Diff function : NOT OK !!!")



# Logger ddp
# pred_trajectories[:,:,int(k/k_mpc)] = mpc_planner.Xs
# pred_forces[:,:,int(k/k_mpc)] = mpc_planner.Us
# fsteps[:,:,int(k/k_mpc)] = mpc_planner.fsteps.copy()
# gait_[:,:,int(k/k_mpc)] = mpc_planner.gait


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
    
    pl2, = plt.plot(l_t, osqp_xs[i,:,iteration], linewidth=2, marker='x')
    pl1, = plt.plot(l_t, ddp_xs[i,:,iteration], linewidth=2, marker='x')
    

    if Relaunch_DDP : 
        pl3, = plt.plot(l_t, mpc_planner.Xs[i,:], linewidth=2, marker='x')
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

    pl2, = plt.plot(l_t, osqp_us[i,:,iteration], linewidth=2, marker='x')
    pl1, = plt.plot(l_t, ddp_us[i,:,iteration], linewidth=2, marker='x')
    
   
    if Relaunch_DDP : 
        pl3, = plt.plot(l_t,mpc_planner.Us[i,:], linewidth=2, marker='x')
        plt.legend([pl1,pl2,pl3] , [l_str2[i] , l_str[i], "ddp_redo" ])
    
    else : 
        plt.legend([pl1,pl2] , [l_str2[i] , l_str[i] ])



# d = 7

# for i in range(4) : 
#   # plt.plot(log_fsteps[:,0,3*i+1] + log_xref[:,0,0]  , log_fsteps[:,0,3*i+2] +  log_xref[:,1,0]  , 'rx' , markerSize = 8)
#   # plt.plot(log_fsteps_ref[:,0,3*i+1] +  log_xref[:,0,0] , log_fsteps_ref[:,0,3*i+2] +  log_xref[:,1,0] , 'bo', markerSize= 8 ,  markerfacecolor='none')
  
#   if i == 0 or i == 1 : 
#     for k in range(iteration_begin , iteration) : 
#       if ddp_gait[k,0,i+1] == 1 : 
#         if ddp_gait[k,0,0] == 7 :  
#           pl1, = plt.plot(ofeet[k,0,i]   , ofeet[k,1,i]   , 'bo' , markerSize = 8)
#         # plt.plot(ofeet_ref[k,0,i]   , ofeet_ref[k,1,i] , 'go', markerSize= 8 ,  markerfacecolor='none')
#       if ddp_gait[k,0,i+1] == 0 :  
#         pl2, = plt.plot(ofeet[k,0,i]   , ofeet[k,1,i]   , 'gx' , markerSize = 8)
#         pl3, = plt.plot(ofeet_ref[k,0,i]   , ofeet_ref[k,1,i] , 'go', markerSize= 8 ,  markerfacecolor='none')
#   if i == 2 or i == 3 : 
#     for k in range(iteration_begin , iteration) : 
#       if ddp_gait[k,0,i+1] == 1 : 
#           if ddp_gait[k,0,0] == 7 :  
#             pl11, = plt.plot(ofeet[k,0,i]   , ofeet[k,1,i]   , 'mo' , markerSize = 8)
#           # plt.plot(ofeet_ref[k,0,i]   , ofeet_ref[k,1,i] , 'go', markerSize= 8 ,  markerfacecolor='none')
#       if ddp_gait[k,0,i+1] == 0 :  
#         pl22, = plt.plot(ofeet[k,0,i]   , ofeet[k,1,i]   , 'rx' , markerSize = 8)
#         pl33, = plt.plot(ofeet_ref[k,0,i]   , ofeet_ref[k,1,i] , 'ro', markerSize= 8 ,  markerfacecolor='none')

# # for k in range(gait_matrix.shape[0]) :
# pl4, = plt.plot(oC[iteration_begin:iteration+d,0]  ,  oC[iteration_begin:iteration+d,1] , "kx")

# plt.legend([pl1,pl2,pl3,pl4] , ["Feet on the ground" , "MPC planner" , "1st planner" , "CoM" ])
# plt.grid()

plt.figure()

# CoM position predicted 
for k in range(len(l_t)) : 
    plt.plot(ddp_xs[0,k,iteration],  ddp_xs[1,k,iteration] ,"kx" ,  markerSize= int(20/np.sqrt(k+1)) )
    plt.plot(xref[0,k+1,iteration],  xref[1,k+1,iteration] ,"gx" ,  markerSize= int(20/(k+1)) )

# First foot on the ground using previous gait matrix : iteration - 1
for i in range(4) :       
    if gait_matrix[0,i+1] == 1 : 
        plt.plot(ddp_xs[12+2*i , 0 , iteration ] , ddp_xs[12+2*i + 1 , 0 , iteration  ] ,  'ro', markerSize= 8   )
        # plt.plot(mpc_planner.Xs[12+2*i , 0  ] , mpc_planner.Xs[12+2*i + 1 , 0  ] ,  'yo', markerSize= 8   )

print(gait_matrix)
j = 0
k_cum = 0
L = []
# Iterate over all phases of the gait
# The first column of xref correspond to the current state 
while (gait_matrix[j, 0] != 0):
    for k in range(k_cum, k_cum+np.int(gait_matrix[j, 0])):

        for l in range(4) : 
            if gait_matrix[j,l+1] == 1 : # next position
                plt.plot(ddp_xs[12+2*l , k , iteration ] , ddp_xs[12+2*l + 1 , k , iteration ] ,  'rx', markerSize= int(16/(j+1))   )
                # plt.plot(mpc_planner.Xs[12+2*l , k  ] , mpc_planner.Xs[12+2*l + 1 , k  ] ,  'yo', markerSize= int(16/(j+1)) ,markerfacecolor='none'   )
            if gait_matrix[j,l+1] == 0 : # state before optimisation
                plt.plot(ddp_xs[12+2*l , k , iteration ] , ddp_xs[12+2*l + 1 , k , iteration ] ,  'bx', markerSize= int(16/(j+1))   )
                # plt.plot(mpc_planner.Xs[12+2*l , k  ] , mpc_planner.Xs[12+2*l + 1 , k  ] ,  'yo', markerSize= int(16/(j+1)) , markerfacecolor='none'   )
  
    k_cum += np.int(gait_matrix[j, 0])
    j += 1 

# Adding shoulder
p0 = [ 0.1946,0.15005, 0.1946,-0.15005, -0.1946,   0.15005 ,-0.1946,  -0.15005]
for i in range(4) :
  pl4, = plt.plot(p0[2*i] , p0[2*i+1] , "ko " , markerSize= 10 ,  markerfacecolor='none' )

plt.grid()




plt.show(block=True)



