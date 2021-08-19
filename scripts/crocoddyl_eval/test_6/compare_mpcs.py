# coding: utf8

import sys
import os 
from sys import argv
sys.path.insert(0, os.getcwd()) # adds current directory to python path

import numpy as np
import matplotlib.pylab as plt
import pinocchio as pin
import libquadruped_reactive_walking as lqrw

def get_mocap_logs(path):
    ''' Get mocap logs and store state in (N,12) array in Base frame (local).
    Position: x,y,z
    Orientation: Roll, Pitch, Yaw
    Linear Velocity: Vx, Vy, Vz
    Angular Velocity: Wroll, Wpitch, Wyaw
    Args:
    - path (str) : path to the .npz file object containing the measures

    Returns:
    - array (Nx12) : Array containing the data
    '''
    # Recover MOCAP logs
    logs = np.load(path)
    mocapPosition = logs.get("mocapPosition")
    mocapOrientationQuat = logs.get("mocapOrientationQuat")
    mocapOrientationMat9 = logs.get("mocapOrientationMat9")
    mocapVelocity = logs.get("mocapVelocity")
    mocapAngularVelocity = logs.get('mocapAngularVelocity')
    N = mocapPosition.shape[0]

    state_measured = np.zeros((N,12))
    # Roll, Pitch, Yaw
    for i in range(N):
        state_measured[i,3:6] = pin.rpy.matrixToRpy(pin.Quaternion(mocapOrientationQuat[i]).toRotationMatrix())

    # Robot world to Mocap initial translationa and rotation
    mTo = np.array([mocapPosition[0, 0], mocapPosition[0, 1], 0.02])  
    mRo = pin.rpy.rpyToMatrix(0.0, 0.0, state_measured[0, 5])

    for i in range(N):
        oRb = mocapOrientationMat9[i]
        oRh = pin.rpy.rpyToMatrix(0.0, 0.0, state_measured[i, 5] - state_measured[0, 5])

        state_measured[i,:3] = (mRo.transpose() @ (mocapPosition[i, :] - mTo).reshape((3, 1))).ravel()
        state_measured[i,6:9] = (oRh.transpose() @ mRo.transpose() @ mocapVelocity[i].reshape((3, 1))).ravel()
        state_measured[i,9:12] = (oRb.transpose() @ mocapAngularVelocity[i].reshape((3, 1))).ravel()       

    return state_measured

def compute_RMSE(array, norm):
    return np.sqrt((array**2).mean()) / norm

##############
# PARAMETERS 
##############

# [Linear, Non Linear, Planner, OSQP]
MPCs = [True, True, True, True] # Boolean to choose which MPC to plot
MPCs_names = ["Linear", "Non Linear", "Planner", "OSQP"]
name_files = ["data_lin.npz", "data_nl.npz", "data_planner.npz", "data_osqp_2.npz"] # Names of the files
folder_path = "crocoddyl_eval/logs/experience_19_08_21/Experiments_Walk_19_08_2021/" # Folder containing the 4 .npz files

# Common data shared by 4 MPCs
params = lqrw.Params()  # Object that holds all controller parameters
logs = np.load(folder_path + name_files[0])
joy_v_ref = logs.get('joy_v_ref')       # Ref velocity (Nx6) given by the joystick  
planner_xref = logs.get("planner_xref") # Ref state
N = joy_v_ref.shape[0]                  # Size of the measures
data_ = np.zeros((N,12,4))              # Store states measured by MOCAP, 4 MPCs (pos,orientation,vel,ang vel)

# Get state measured
for i in range(4):
    if MPCs[i]:
        data_[:,:,i] = get_mocap_logs(folder_path + name_files[i])


##########
# PLOTS 
##########
lgd = ["Position X", "Position Y", "Position Z", "Position Roll", "Position Pitch", "Position Yaw"]
index6 = [1, 3, 5, 2, 4, 6]
t_range = np.array([k*params.dt_wbc for k in range(N)])

color = ["k", "b", "r", "g--"]
legend = []
for i in range(4):
    if MPCs[i]:
        legend.append(MPCs_names[i])


plt.figure()
for i in range(6):
    plt.subplot(3, 2, index6[i])
    
    for j in range(4):
        if MPCs[j]:
            plt.plot(t_range, data_[:,i,j], color[j], linewidth=3)
    
   
    plt.legend(legend, prop={'size': 8})
    plt.ylabel(lgd[i])
plt.suptitle("Measured postion and orientation - MOCAP - ")

lgd = ["Linear vel X", "Linear vel Y", "Linear vel Z",
               "Angular vel Roll", "Angular vel Pitch", "Angular vel Yaw"]
plt.figure()
for i in range(6):
    plt.subplot(3, 2, index6[i])
    
    for j in range(4):
        if MPCs[j]:
            plt.plot(t_range, data_[:,i+6,j], color[j], linewidth=3)    
   
    plt.legend(legend, prop={'size': 8})
    plt.ylabel(lgd[i])
plt.suptitle("Measured postion and orientation - MOCAP - ")

# Compute difference measured - reference
data_diff = np.zeros((N, 12,4))

for i in range(4):
    if MPCs[i]:
        data_diff[:,:6,i] = data_[:,:6,i] - planner_xref[:, :6, 1]  # Position and orientation
        data_diff[:,6:,i] = data_[:,6:,i] - joy_v_ref[:,:]          # Linear and angular velocities

lgd = ["Linear vel X", "Linear vel Y", "Position Z",
               "Position Roll", "Position Pitch", "Ang vel Yaw"]
index_error = [6,7,2,3,4,11]




# Compute the mean of the difference (measured - reference)
# Using a window mean (valid, excludes the boundaries). 
# The size of the data output are then reduced by period - 1

period = int( 2* (params.T_gait / 2 ) / params.dt_wbc ) # Period of the window 
data_diff_valid = data_diff[int(period/2 - 1) : -int(period/2) , :,:] # Reshape of the (measure - ref) arrays
t_range_valid = t_range[int(period/2 - 1) : -int(period/2) ] # Reshape of the timing for plottings
data_diff_mean_valid = np.zeros(data_diff_valid.shape) # Mean array
data_mean_valid = np.zeros(data_diff_valid.shape) # Mean array

for j in range(4):
    for i in range(12):
        data_mean_valid[:,i,j]  = np.convolve(data_diff[:,i,j], np.ones((period,)) / period, mode = "valid")
        data_diff_mean_valid[:,i,j] = data_diff_valid[:,i,j] - np.convolve(data_diff[:,i,j], np.ones((period,)) / period, mode = "valid")

plt.figure()
for i in range(6):
    plt.subplot(3, 2, index6[i])
    
    for j in range(4):
        if MPCs[j]:
            plt.plot(t_range, data_diff[:,index_error[i],j], color[j], linewidth=3)    

# Add mean on graph
# for i in range(6):
#     plt.subplot(3, 2, index6[i])    
#     for j in range(4):
#         if MPCs[j]:
#             plt.plot(t_range_valid, data_mean_valid[:,index_error[i],j], color[j] + "x-", linewidth=3)   

    plt.legend(legend, prop={'size': 8})
    plt.ylabel(lgd[i])
plt.suptitle("Error wrt reference state")

plt.figure()
for i in range(6):
    plt.subplot(3, 2, index6[i])
    
    for j in range(4):
        if MPCs[j]:
            plt.plot(t_range_valid, data_diff_mean_valid[:,index_error[i],j], color[j], linewidth=3)    
   
    plt.legend(legend, prop={'size': 8})
    plt.ylabel(lgd[i])
plt.suptitle("Error wrt reference state - smooth mean (window of 2 period) - ")



data_RMSE = np.zeros((12,4))
data_RMSE_mean = np.zeros((12,4))

norm_max = np.max(abs(data_[:,:,0]) , axis = 0) # Max of first MPC as norm for each component
norm_max_mean = np.max(abs(data_[:,:,0]) , axis = 0) # Max of first MPC as norm for each component

for i in range(12):
    for j in range(4):
        if MPCs[j]:
            data_RMSE[i,j] = compute_RMSE(data_diff[:,i,j], norm_max[i])
            data_RMSE_mean[i,j] = compute_RMSE(data_diff_mean_valid[:,i,j], norm_max_mean[i])

lgd = ["Linear vel X", "Linear vel Y", "Position Z",
               "Position Roll", "Position Pitch", "Ang vel Yaw"]
index_error = [6,7,2,3,4,11]
bars = []
bars_names = ["Lin", "NL", "Plan", "OSQP"]
for j in range(4):
    if MPCs[j]:
        bars.append(bars_names[j])

plt.figure()
for i in range(6):
    plt.subplot(3, 2, index6[i])    
    heights = []

    for j in range(4):
        if MPCs[j]:
            heights.append(data_RMSE[index_error[i] , j])

        
    y_pos = range(len(bars))
    plt.bar(y_pos, heights)
    plt.ylim([0., 0.6])
    # Rotation of the bars names
    plt.xticks(y_pos, bars, rotation=0)
    plt.ylabel(lgd[i])
plt.suptitle("NORMALIZED RMSE : sqrt(  (measures - ref**2).mean() ) / measure_max")



plt.figure()
for i in range(6):
    plt.subplot(3, 2, index6[i])    
    heights = []

    for j in range(4):
        if MPCs[j]:
            heights.append(data_RMSE_mean[index_error[i] , j])

        
    y_pos = range(len(bars))
    plt.bar(y_pos, heights)
    plt.ylim([0., 0.6])
    # Rotation of the bars names
    plt.xticks(y_pos, bars, rotation=0)
    plt.ylabel(lgd[i])
plt.suptitle("NORMALIZED RMSE -MEAN: sqrt(  (mes - ref - mean(mes-ref))  **2).mean() ) / measure_max")




# # period = 2*250 
# # mean_window_pos_lin = np.zeros((N,3))
# # mean_window_pos_nl = np.zeros((N,3))
# # mean_window_pos_planner = np.zeros((N,3))
# # mean_window_pos_osqp = np.zeros((N,3))
# # for i in range(3):
# #     mean_window_pos_lin[:,i] = np.convolve(diff_pos_lin[:,i] , np.ones((period,))/period, mode = "valid")
# #     mean_window_pos_nl[:,i] = np.convolve(diff_pos_lin[:,i] , np.ones((period,))/period, mode = "valid")
# #     mean_window_pos_planner[:,i] = np.convolve(diff_pos_lin[:,i] , np.ones((period,))/period, mode = "valid")
# #     mean_window_pos_osqp[:,i] = np.convolve(diff_pos_lin[:,i] , np.ones((period,))/period, mode = "valid")

# # diff_pos_lin[int(period/2)-1:-int(period/2)]



# diff_mean_pos_lin = abs((mocap_pos_mpc_lin[:,:] - planner_xref[:, :3, 1] ) - np.mean((mocap_pos_mpc_lin[:,:] - planner_xref[:, :3, 1] )))
# diff_mean_pos_nl = abs((mocap_pos_mpc_nl[:,:] - planner_xref[:, :3, 1] ) - np.mean((mocap_pos_mpc_nl[:,:] - planner_xref[:, :3, 1] )))
# diff_mean_pos_planner = abs((mocap_pos_mpc_planner[:,:] - planner_xref[:, :3, 1]) - np.mean((mocap_pos_mpc_planner[:,:] - planner_xref[:, :3, 1] )))
# diff_mean_pos_osqp = abs( (mocap_pos_mpc_osqp[:,:] - planner_xref[:, :3, 1]) - np.mean((mocap_pos_mpc_osqp[:,:] - planner_xref[:, :3, 1] )))

# diff_mean_rpy_lin = (mocap_RPY_mpc_lin[:,:] - planner_xref[:, 3:6, 1] )
# diff_mean_rpy_nl = (mocap_RPY_mpc_nl[:,:] - planner_xref[:, 3:6, 1] )
# diff_mean_rpy_planner = (mocap_RPY_mpc_planner[:,:] - planner_xref[:, 3:6, 1] )
# diff_mean_rpy_osqp = (mocap_RPY_mpc_osqp[:,:] - planner_xref[:, 3:6, 1] )


# diff_mean_vel_lin = (mocap_h_v_mpc_lin[:,:] - joy_v_ref[:,:3])
# diff_mean_vel_nl = (mocap_h_v_mpc_nl[:,:] - joy_v_ref[:,:3])
# diff_mean_vel_planner = (mocap_h_v_mpc_planner[:,:] - joy_v_ref[:,:3])
# diff_mean_vel_osqp = (mocap_h_v_mpc_osqp[:,:] - joy_v_ref[:,:3])

# diff_mean_ang_lin = (mocap_b_w_mpc_lin[:,:] - joy_v_ref[:,3:6])
# diff_mean_ang_nl = (mocap_b_w_mpc_nl[:,:] - joy_v_ref[:,3:6])
# diff_mean_ang_planner = (mocap_b_w_mpc_planner[:,:] - joy_v_ref[:,3:6])
# diff_mean_ang_osqp = (mocap_b_w_mpc_osqp[:,:] - joy_v_ref[:,3:6])



# # print('RMSE Vx [Lin, Nl, PLanner]: ', np.linalg.norm(diff_vel_lin))

# lgd = ["Linear vel X", "Linear vel Y", "Position Z",
#                "Position Roll", "Position Pitch", "Ang vel Yaw"]
# plt.figure()
# for i in range(6):
#     if i == 0:
#         ax0 = plt.subplot(3, 2, index6[i])
#     else:
#         plt.subplot(3, 2, index6[i], sharex=ax0)
    
#     if i < 2:
#         if Linear:      plt.plot(t_range, diff_vel_lin[:, i], "k", linewidth=3)
#         if Non_linear:  plt.plot(t_range, diff_vel_nl[:, i], "b", linewidth=3)
#         if Planner:     plt.plot(t_range, diff_vel_planner[:, i], "r", linewidth=3)
#         if OSQP:        plt.plot(t_range, diff_vel_osqp[:, i], "g--", linewidth=3)
    
#     elif i == 2 : 
#         if Linear:      plt.plot(t_range, diff_pos_lin[:, i], "k", linewidth=3)
#         if Non_linear:  plt.plot(t_range, diff_pos_nl[:, i], "b", linewidth=3)
#         if Planner:     plt.plot(t_range, diff_pos_planner[:, i], "r", linewidth=3)
#         if OSQP:        plt.plot(t_range, diff_pos_osqp[:, i], "g--", linewidth=3)


#     elif i == 3 or i == 4:
#         if Linear:      plt.plot(t_range, diff_rpy_lin[:, i-3], "k", linewidth=3)
#         if Non_linear:  plt.plot(t_range, diff_rpy_nl[:, i-3], "b", linewidth=3)
#         if Planner:     plt.plot(t_range, diff_rpy_planner[:, i-3], "r", linewidth=3)
#         if OSQP:        plt.plot(t_range, diff_rpy_osqp[:, i-3], "g--", linewidth=3)
    
#     else :

#         if Linear:      plt.plot(t_range, diff_ang_lin[:, i-3], "k", linewidth=3)
#         if Non_linear:  plt.plot(t_range, diff_ang_nl[:, i-3], "b", linewidth=3)
#         if Planner:     plt.plot(t_range, diff_ang_planner[:, i-3], "r", linewidth=3)
#         if OSQP:        plt.plot(t_range, diff_ang_osqp[:, i-3], "g--", linewidth=3)

#         """N = 2000
#         y = np.convolve(self.mocap_b_w[:, i-3], np.ones(N)/N, mode='valid')
#         plt.plot(t_range[int(N/2)-1:-int(N/2)], y, linewidth=3, linestyle="--")"""

#     # plt.plot(t_range, self.log_dq[i, :], "g", linewidth=2)
#     # plt.plot(t_range[:-2], self.log_dx_invkin[i, :-2], "g", linewidth=2)
#     # plt.plot(t_range[:-2], self.log_dx_ref_invkin[i, :-2], "violet", linewidth=2, linestyle="--")
#     plt.legend(legend, prop={'size': 8})
#     plt.ylabel(lgd[i])
# plt.suptitle("Absolute error wrt reference")



# # Diff with reference 
# diff_pos_lin = (mocap_pos_mpc_lin[:,:] - planner_xref[:, :3, 1] )
# diff_pos_nl = (mocap_pos_mpc_nl[:,:] - planner_xref[:, :3, 1] )
# diff_pos_planner = (mocap_pos_mpc_planner[:,:] - planner_xref[:, :3, 1] )
# diff_pos_osqp = (mocap_pos_mpc_osqp[:,:] - planner_xref[:, :3, 1] )

# diff_rpy_lin = (mocap_RPY_mpc_lin[:,:] - planner_xref[:, 3:6, 1] )
# diff_rpy_nl = (mocap_RPY_mpc_nl[:,:] - planner_xref[:, 3:6, 1] )
# diff_rpy_planner = (mocap_RPY_mpc_planner[:,:] - planner_xref[:, 3:6, 1] )
# diff_rpy_osqp = (mocap_RPY_mpc_osqp[:,:] - planner_xref[:, 3:6, 1] )


# diff_vel_lin = (mocap_h_v_mpc_lin[:,:] - joy_v_ref[:,:3])
# diff_vel_nl = (mocap_h_v_mpc_nl[:,:] - joy_v_ref[:,:3])
# diff_vel_planner = (mocap_h_v_mpc_planner[:,:] - joy_v_ref[:,:3])
# diff_vel_osqp = (mocap_h_v_mpc_osqp[:,:] - joy_v_ref[:,:3])


# diff_ang_lin = (mocap_b_w_mpc_lin[:,:] - joy_v_ref[:,3:6])
# diff_ang_nl = (mocap_b_w_mpc_nl[:,:] - joy_v_ref[:,3:6])
# diff_ang_planner = (mocap_b_w_mpc_planner[:,:] - joy_v_ref[:,3:6])
# diff_ang_osqp = (mocap_b_w_mpc_osqp[:,:] - joy_v_ref[:,3:6])

# # abs(measure - reference) - mean(measure - reference) 

# # Max measures 
# max_pos_lin = np.max(abs(mocap_pos_mpc_lin), axis = 0)
# max_pos_nl = np.max(abs(mocap_pos_mpc_nl), axis = 0)
# max_pos_planner = np.max(abs(mocap_pos_mpc_planner), axis = 0)
# max_pos_osqp = np.max(abs(mocap_pos_mpc_osqp), axis = 0)

# max_rpy_lin = np.max(abs(mocap_RPY_mpc_lin), axis = 0)
# max_rpy_nl = np.max(abs(mocap_RPY_mpc_nl), axis = 0)
# max_rpy_planner = np.max(abs(mocap_RPY_mpc_planner), axis = 0)
# max_rpy_osqp = np.max(abs(mocap_RPY_mpc_osqp), axis = 0)

# max_vel_lin = np.max(abs(mocap_h_v_mpc_lin), axis = 0)
# max_vel_nl = np.max(abs(mocap_h_v_mpc_nl), axis = 0)
# max_vel_planner = np.max(abs(mocap_h_v_mpc_planner), axis = 0)
# max_vel_osqp = np.max(abs(mocap_h_v_mpc_osqp), axis = 0)

# max_ang_lin = np.max(abs(mocap_b_w_mpc_lin), axis = 0)
# max_ang_nl = np.max(abs(mocap_b_w_mpc_nl), axis = 0)
# max_ang_planner = np.max(abs(mocap_b_w_mpc_planner), axis = 0)
# max_ang_osqp = np.max(abs(mocap_b_w_mpc_osqp), axis = 0)

# # RMSE normalized
# RMSE_pos_lin = np.sqrt((diff_pos_lin**2).mean(axis=0)) / max_pos_lin
# RMSE_pos_nl =  np.sqrt((diff_pos_nl**2).mean(axis=0)) / max_pos_lin
# RMSE_pos_planner = np.sqrt((diff_pos_planner**2).mean(axis=0)) / max_pos_lin
# RMSE_pos_osqp = np.sqrt((diff_pos_osqp**2).mean(axis=0)) / max_pos_lin

# RMSE_rpy_lin = np.sqrt((diff_rpy_lin**2).mean(axis=0)) / max_rpy_lin
# RMSE_rpy_nl = np.sqrt((diff_rpy_nl**2).mean(axis=0))  / max_rpy_lin
# RMSE_rpy_planner = np.sqrt((diff_rpy_planner**2).mean(axis=0)) / max_rpy_lin
# RMSE_rpy_osqp = np.sqrt((diff_rpy_osqp**2).mean(axis=0)) / max_rpy_lin

# RMSE_vel_lin = np.sqrt((diff_vel_lin**2).mean(axis=0))  / max_vel_lin
# RMSE_vel_nl = np.sqrt((diff_vel_nl**2).mean(axis=0)) / max_vel_lin
# RMSE_vel_planner = np.sqrt((diff_vel_planner**2).mean(axis=0)) / max_vel_lin
# RMSE_vel_osqp = np.sqrt((diff_vel_osqp**2).mean(axis=0)) / max_vel_lin

# RMSE_ang_lin = np.sqrt((diff_ang_lin**2).mean(axis=0)) / max_ang_lin
# RMSE_ang_nl = np.sqrt((diff_ang_nl**2).mean(axis=0))  / max_ang_lin
# RMSE_ang_planner = np.sqrt((diff_ang_planner**2).mean(axis=0)) / max_ang_lin
# RMSE_ang_osqp = np.sqrt((diff_ang_osqp**2).mean(axis=0)) / max_ang_lin

# plt.figure()

# bars = []
# if Linear : bars.append("Lin")
# if Non_linear : bars.append("NL")
# if Planner : bars.append("Plan")
# if OSQP : bars.append("OSQP")


# lgd = ["Linear vel X", "Linear vel Y", "Position Z",
#                "Position Roll", "Position Pitch", "Ang vel Yaw"]

# for i in range(6):
#     plt.subplot(3, 2, index6[i])    
#     heights = []

#     if i in [0,1]:        
#         if Linear : heights.append(RMSE_vel_lin[i])
#         if Non_linear : heights.append(RMSE_vel_nl[i])
#         if Planner : heights.append(RMSE_vel_planner[i])
#         if OSQP : heights.append(RMSE_vel_osqp[i])

#     if i == 2:        
#         if Linear : heights.append(RMSE_pos_lin[2])
#         if Non_linear : heights.append(RMSE_pos_nl[2])
#         if Planner : heights.append(RMSE_pos_planner[2])
#         if OSQP : heights.append(RMSE_pos_osqp[2])

#     if i in [3,4]:        
#         if Linear : heights.append(RMSE_rpy_lin[i-3])
#         if Non_linear : heights.append(RMSE_rpy_nl[i-3])
#         if Planner : heights.append(RMSE_rpy_planner[i-3])
#         if OSQP : heights.append(RMSE_rpy_osqp[i-3])
    
#     if i == 5:        
#         if Linear : heights.append(RMSE_ang_lin[2])
#         if Non_linear : heights.append(RMSE_ang_nl[2])
#         if Planner : heights.append(RMSE_ang_planner[2])
#         if OSQP : heights.append(RMSE_ang_osqp[2])

        
#     y_pos = range(len(bars))
#     plt.bar(y_pos, heights)
#     plt.ylim([0., 0.6])
#     # Rotation of the bars names
#     plt.xticks(y_pos, bars, rotation=0)
#     plt.ylabel(lgd[i])
# plt.suptitle("NORMALIZED RMSE : sqrt(  (measures - ref**2).mean() ) / measure_max")

# print("NORMALIZED RMSE : sqrt(  (measures - ref**2).mean() ) / measure_max ")

# print('RMSE Velocity : [Vx, Vy] ')
# print(RMSE_vel_lin[:2], ' : LINEAR')
# print(RMSE_vel_nl[:2], ' : NON LINEAR')
# print(RMSE_vel_planner[:2], ' : PLANNER')
# print(RMSE_vel_osqp[:2], ' : OSQP')

# print("\n\n")
# print('RMSE POSITION : [Z] ')
# print(RMSE_pos_lin[2], ' : LINEAR')
# print(RMSE_pos_nl[2], ' : NON LINEAR')
# print(RMSE_pos_planner[2], ' : PLANNER')
# print(RMSE_pos_osqp[2], ' : OSQP')

# print("\n\n")
# print('RMSE Roll / Pitch : ')
# print(RMSE_rpy_lin[:2], ' : LINEAR')
# print(RMSE_rpy_nl[:2], ' : NON LINEAR')
# print(RMSE_rpy_planner[:2], ' : PLANNER')
# print(RMSE_rpy_osqp[:2], ' : OSQP')

# print("\n\n")
# print('RMSE Yaw Velocity : ')
# print(RMSE_ang_lin[2], ' : LINEAR')
# print(RMSE_ang_nl[2], ' : NON LINEAR')
# print(RMSE_ang_planner[2], ' : PLANNER')
# print(RMSE_ang_osqp[2], ' : OSQP')

# # Display all graphs and wait
plt.show(block=True)