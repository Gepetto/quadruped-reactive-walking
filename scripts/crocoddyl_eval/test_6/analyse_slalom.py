# coding: utf8

import sys
import os 
from sys import argv
sys.path.insert(0, os.getcwd()) # adds current directory to python path

import numpy as np
import matplotlib.pylab as plt
import pinocchio as pin
import libquadruped_reactive_walking as lqrw


######################
# Recover Logged data, mpc lin
######################
file_name = "crocoddyl_eval/logs/vel6/data_lin_vel6.npz"
logs = np.load(file_name)


# Common data

joy_v_ref = logs.get('joy_v_ref')
planner_xref = logs.get("planner_xref")



mocapPosition = logs.get("mocapPosition")
mocapOrientationQuat = logs.get("mocapOrientationQuat")
mocapOrientationMat9 = logs.get("mocapOrientationMat9")
mocapVelocity = logs.get("mocapVelocity")
mocapAngularVelocity = logs.get('mocapAngularVelocity')

params = lqrw.Params()  # Object that holds all controller parameters


N = mocapPosition.shape[0]

mocap_pos_mpc_lin = np.zeros([N, 3])
mocap_h_v_mpc_lin = np.zeros([N, 3])
mocap_b_w_mpc_lin = np.zeros([N, 3])
mocap_RPY_mpc_lin = np.zeros([N, 3])

for i in range(N):
    mocap_RPY_mpc_lin[i] = pin.rpy.matrixToRpy(pin.Quaternion(mocapOrientationQuat[i]).toRotationMatrix())

# Robot world to Mocap initial translationa and rotation
mTo = np.array([mocapPosition[0, 0], mocapPosition[0, 1], 0.02])  
mRo = pin.rpy.rpyToMatrix(0.0, 0.0, mocap_RPY_mpc_lin[0, 2])

for i in range(N):
    oRb = mocapOrientationMat9[i]

    oRh = pin.rpy.rpyToMatrix(0.0, 0.0, mocap_RPY_mpc_lin[i, 2] - mocap_RPY_mpc_lin[0, 2])

    mocap_h_v_mpc_lin[i] = (oRh.transpose() @ mRo.transpose() @ mocapVelocity[i].reshape((3, 1))).ravel()
    mocap_b_w_mpc_lin[i] = (oRb.transpose() @ mocapAngularVelocity[i].reshape((3, 1))).ravel()
    mocap_pos_mpc_lin[i] = (mRo.transpose() @ (mocapPosition[i, :] - mTo).reshape((3, 1))).ravel()


######################
# Recover Logged data, mpc non linear
######################
file_name = "crocoddyl_eval/logs/vel6/data_nl_vel6.npz"
logs = np.load(file_name)

mocapPosition = logs.get("mocapPosition")
mocapOrientationQuat = logs.get("mocapOrientationQuat")
mocapOrientationMat9 = logs.get("mocapOrientationMat9")
mocapVelocity = logs.get("mocapVelocity")
mocapAngularVelocity = logs.get('mocapAngularVelocity')

params = lqrw.Params()  # Object that holds all controller parameters


N = mocapPosition.shape[0]

mocap_pos_mpc_nl = np.zeros([N, 3])
mocap_h_v_mpc_nl = np.zeros([N, 3])
mocap_b_w_mpc_nl = np.zeros([N, 3])
mocap_RPY_mpc_nl = np.zeros([N, 3])

for i in range(N):
    mocap_RPY_mpc_nl[i] = pin.rpy.matrixToRpy(pin.Quaternion(mocapOrientationQuat[i]).toRotationMatrix())

# Robot world to Mocap initial translationa and rotation
mTo = np.array([mocapPosition[0, 0], mocapPosition[0, 1], 0.02])  
mRo = pin.rpy.rpyToMatrix(0.0, 0.0, mocap_RPY_mpc_nl[0, 2])

for i in range(N):
    oRb = mocapOrientationMat9[i]

    oRh = pin.rpy.rpyToMatrix(0.0, 0.0, mocap_RPY_mpc_nl[i, 2] - mocap_RPY_mpc_nl[0, 2])

    mocap_h_v_mpc_nl[i] = (oRh.transpose() @ mRo.transpose() @ mocapVelocity[i].reshape((3, 1))).ravel()
    mocap_b_w_mpc_nl[i] = (oRb.transpose() @ mocapAngularVelocity[i].reshape((3, 1))).ravel()
    mocap_pos_mpc_nl[i] = (mRo.transpose() @ (mocapPosition[i, :] - mTo).reshape((3, 1))).ravel()


######################
# Recover Logged data, mpc planner
######################
file_name = "crocoddyl_eval/logs/vel6/data_planner_vel6.npz"
logs = np.load(file_name)

mocapPosition = logs.get("mocapPosition")
mocapOrientationQuat = logs.get("mocapOrientationQuat")
mocapOrientationMat9 = logs.get("mocapOrientationMat9")
mocapVelocity = logs.get("mocapVelocity")
mocapAngularVelocity = logs.get('mocapAngularVelocity')

params = lqrw.Params()  # Object that holds all controller parameters


N = mocapPosition.shape[0]

mocap_pos_mpc_planner = np.zeros([N, 3])
mocap_h_v_mpc_planner = np.zeros([N, 3])
mocap_b_w_mpc_planner = np.zeros([N, 3])
mocap_RPY_mpc_planner = np.zeros([N, 3])

for i in range(N):
    mocap_RPY_mpc_planner[i] = pin.rpy.matrixToRpy(pin.Quaternion(mocapOrientationQuat[i]).toRotationMatrix())

# Robot world to Mocap initial translationa and rotation
mTo = np.array([mocapPosition[0, 0], mocapPosition[0, 1], 0.02])  
mRo = pin.rpy.rpyToMatrix(0.0, 0.0, mocap_RPY_mpc_planner[0, 2])

for i in range(N):
    oRb = mocapOrientationMat9[i]

    oRh = pin.rpy.rpyToMatrix(0.0, 0.0, mocap_RPY_mpc_planner[i, 2] - mocap_RPY_mpc_planner[0, 2])

    mocap_h_v_mpc_planner[i] = (oRh.transpose() @ mRo.transpose() @ mocapVelocity[i].reshape((3, 1))).ravel()
    mocap_b_w_mpc_planner[i] = (oRb.transpose() @ mocapAngularVelocity[i].reshape((3, 1))).ravel()
    mocap_pos_mpc_planner[i] = (mRo.transpose() @ (mocapPosition[i, :] - mTo).reshape((3, 1))).ravel()



lgd = ["Position X", "Position Y", "Position Z", "Position Roll", "Position Pitch", "Position Yaw"]
index6 = [1, 3, 5, 2, 4, 6]
t_range = np.array([k*params.dt_wbc for k in range(N)])

plt.figure()
for i in range(6):
    if i == 0:
        ax0 = plt.subplot(3, 2, index6[i])
    else:
        plt.subplot(3, 2, index6[i], sharex=ax0)

    if i < 3:
        plt.plot(t_range, mocap_pos_mpc_lin[:, i], "k", linewidth=3)
        plt.plot(t_range, mocap_pos_mpc_nl[:, i], "b", linewidth=3)
        plt.plot(t_range, mocap_pos_mpc_planner[:, i], "r", linewidth=3)
    else:
        plt.plot(t_range, mocap_RPY_mpc_lin[:, i-3], "k", linewidth=3)
        plt.plot(t_range, mocap_RPY_mpc_nl[:, i-3], "b", linewidth=3)
        plt.plot(t_range, mocap_RPY_mpc_planner[:, i-3], "r", linewidth=3)
   
    plt.legend(["MOCAP LIN", "MOCAP NL", "MOCAP PLANNER"], prop={'size': 8})
    plt.ylabel(lgd[i])
plt.suptitle("Measured & Reference position and orientation")

lgd = ["Linear vel X", "Linear vel Y", "Linear vel Z",
               "Andiff_pos_lingular vel Roll", "Angular vel Pitch", "Angular vel Yaw"]
plt.figure()
for i in range(6):
    if i == 0:
        ax0 = plt.subplot(3, 2, index6[i])
    else:
        plt.subplot(3, 2, index6[i], sharex=ax0)
    
    if i < 3:
        plt.plot(t_range, mocap_h_v_mpc_lin[:, i], "k", linewidth=3)
        plt.plot(t_range, mocap_h_v_mpc_nl[:, i], "b", linewidth=3)
        plt.plot(t_range, mocap_h_v_mpc_planner[:, i], "r", linewidth=3)
    else:
        plt.plot(t_range, mocap_b_w_mpc_lin[:, i-3], "k", linewidth=3)
        plt.plot(t_range, mocap_b_w_mpc_nl[:, i-3], "b", linewidth=3)
        plt.plot(t_range, mocap_b_w_mpc_planner[:, i-3], "r", linewidth=3)

        """N = 2000
        y = np.convolve(self.mocap_b_w[:, i-3], np.ones(N)/N, mode='valid')
        plt.plot(t_range[int(N/2)-1:-int(N/2)], y, linewidth=3, linestyle="--")"""

    # plt.plot(t_range, self.log_dq[i, :], "g", linewidth=2)
    # plt.plot(t_range[:-2], self.log_dx_invkin[i, :-2], "g", linewidth=2)
    # plt.plot(t_range[:-2], self.log_dx_ref_invkin[i, :-2], "violet", linewidth=2, linestyle="--")
    plt.legend(["MOCAP LIN", "MOCAP NL", "MOCAP PLANNER"], prop={'size': 8})
    plt.ylabel(lgd[i])
plt.suptitle("Measured & Reference linear and angular velocities")


diff_vel_lin = abs(mocap_h_v_mpc_lin[:,:] - joy_v_ref[:,:3])
diff_vel_nl = abs(mocap_h_v_mpc_nl[:,:] - joy_v_ref[:,:3])
diff_vel_planner = abs(mocap_h_v_mpc_planner[:,:] - joy_v_ref[:,:3])

diff_ang_lin = abs(mocap_b_w_mpc_lin[:,:] - joy_v_ref[:,3:6])
diff_ang_nl = abs(mocap_b_w_mpc_nl[:,:] - joy_v_ref[:,3:6])
diff_ang_planner = abs(mocap_b_w_mpc_planner[:,:] - joy_v_ref[:,3:6])

diff_rpy_lin = abs(mocap_RPY_mpc_lin[:,:] - planner_xref[:, 3:6, 1] )
diff_rpy_nl = abs(mocap_RPY_mpc_nl[:,:] - planner_xref[:, 3:6, 1] )
diff_rpy_planner = abs(mocap_RPY_mpc_planner[:,:] - planner_xref[:, 3:6, 1] )

diff_pos_lin = abs(mocap_pos_mpc_lin[:,:] - planner_xref[:, :3, 1] )
diff_pos_nl = abs(mocap_pos_mpc_nl[:,:] - planner_xref[:, :3, 1] )
diff_pos_planner = abs(mocap_pos_mpc_planner[:,:] - planner_xref[:, :3, 1] )

# print('RMSE Vx [Lin, Nl, PLanner]: ', np.linalg.norm(diff_vel_lin))

lgd = ["Linear vel X", "Linear vel Y", "POsition Z",
               "Position Roll", "Position Pitch", "Ang vel Yaw"]
plt.figure()
for i in range(6):
    if i == 0:
        ax0 = plt.subplot(3, 2, index6[i])
    else:
        plt.subplot(3, 2, index6[i], sharex=ax0)
    
    if i < 2:
        plt.plot(t_range, diff_vel_lin[:, i], "k", linewidth=3)
        plt.plot(t_range, diff_vel_nl[:, i], "b", linewidth=3)
        plt.plot(t_range, diff_vel_planner[:, i], "r", linewidth=3)
    
    elif i == 2 : 
        plt.plot(t_range, diff_pos_lin[:, i], "k", linewidth=3)
        plt.plot(t_range, diff_pos_nl[:, i], "b", linewidth=3)
        plt.plot(t_range, diff_pos_planner[:, i], "r", linewidth=3)


    elif i == 3 or i == 4:
        plt.plot(t_range, diff_rpy_lin[:, i-3], "k", linewidth=3)
        plt.plot(t_range, diff_rpy_nl[:, i-3], "b", linewidth=3)
        plt.plot(t_range, diff_rpy_planner[:, i-3], "r", linewidth=3)
    
    else :

        plt.plot(t_range, diff_ang_lin[:, i-3], "k", linewidth=3)
        plt.plot(t_range, diff_ang_nl[:, i-3], "b", linewidth=3)
        plt.plot(t_range, diff_ang_planner[:, i-3], "r", linewidth=3)

        """N = 2000
        y = np.convolve(self.mocap_b_w[:, i-3], np.ones(N)/N, mode='valid')
        plt.plot(t_range[int(N/2)-1:-int(N/2)], y, linewidth=3, linestyle="--")"""

    # plt.plot(t_range, self.log_dq[i, :], "g", linewidth=2)
    # plt.plot(t_range[:-2], self.log_dx_invkin[i, :-2], "g", linewidth=2)
    # plt.plot(t_range[:-2], self.log_dx_ref_invkin[i, :-2], "violet", linewidth=2, linestyle="--")
    plt.legend(["LIN", "NL", "PLANNER"], prop={'size': 8})
    plt.ylabel(lgd[i])
plt.suptitle("Measured & Reference linear and angular velocities (ABS ERROR) ")



# Diff with reference 
diff_pos_lin = (mocap_pos_mpc_lin[:,:] - planner_xref[:, :3, 1] )
diff_pos_nl = (mocap_pos_mpc_nl[:,:] - planner_xref[:, :3, 1] )
diff_pos_planner = (mocap_pos_mpc_planner[:,:] - planner_xref[:, :3, 1] )

diff_rpy_lin = (mocap_RPY_mpc_lin[:,:] - planner_xref[:, 3:6, 1] )
diff_rpy_nl = (mocap_RPY_mpc_nl[:,:] - planner_xref[:, 3:6, 1] )
diff_rpy_planner = (mocap_RPY_mpc_planner[:,:] - planner_xref[:, 3:6, 1] )

diff_vel_lin = (mocap_h_v_mpc_lin[:,:] - joy_v_ref[:,:3])
diff_vel_nl = (mocap_h_v_mpc_nl[:,:] - joy_v_ref[:,:3])
diff_vel_planner = (mocap_h_v_mpc_planner[:,:] - joy_v_ref[:,:3])

diff_ang_lin = (mocap_b_w_mpc_lin[:,:] - joy_v_ref[:,3:6])
diff_ang_nl = (mocap_b_w_mpc_nl[:,:] - joy_v_ref[:,3:6])
diff_ang_planner = (mocap_b_w_mpc_planner[:,:] - joy_v_ref[:,3:6])


# Max measures 
max_pos_lin = abs(np.max(mocap_pos_mpc_lin, axis = 0))
max_pos_nl = abs(np.max(mocap_pos_mpc_nl, axis = 0))
max_pos_planner = abs(np.max(mocap_pos_mpc_planner, axis = 0))

max_rpy_lin = abs(np.max(mocap_RPY_mpc_lin, axis = 0))
max_rpy_nl = abs(np.max(mocap_RPY_mpc_nl, axis = 0))
max_rpy_planner = abs(np.max(mocap_RPY_mpc_planner, axis = 0))

max_vel_lin = abs(np.max(mocap_h_v_mpc_lin, axis = 0))
max_vel_nl = abs(np.max(mocap_h_v_mpc_nl, axis = 0))
max_vel_planner = abs(np.max(mocap_h_v_mpc_planner, axis = 0))

max_ang_lin = abs(np.max(mocap_b_w_mpc_lin, axis = 0))
max_ang_nl = abs(np.max(mocap_b_w_mpc_nl, axis = 0))
max_ang_planner = abs(np.max(mocap_b_w_mpc_planner, axis = 0))

# RMSE normalized
RMSE_pos_lin = np.sqrt((diff_pos_lin**2).mean(axis=0)) / max_pos_lin
RMSE_pos_nl =  np.sqrt((diff_pos_nl**2).mean(axis=0)) / max_pos_lin
RMSE_pos_planner = np.sqrt((diff_pos_planner**2).mean(axis=0)) / max_pos_lin

RMSE_rpy_lin = np.sqrt((diff_rpy_lin**2).mean(axis=0)) / max_rpy_lin
RMSE_rpy_nl = np.sqrt((diff_rpy_nl**2).mean(axis=0))  / max_rpy_lin
RMSE_rpy_planner = np.sqrt((diff_rpy_planner**2).mean(axis=0)) / max_rpy_lin

RMSE_vel_lin = np.sqrt((diff_vel_lin**2).mean(axis=0))  / max_vel_lin
RMSE_vel_nl = np.sqrt((diff_vel_nl**2).mean(axis=0)) / max_vel_lin
RMSE_vel_planner = np.sqrt((diff_vel_planner**2).mean(axis=0)) / max_vel_lin

RMSE_ang_lin = np.sqrt((diff_ang_lin**2).mean(axis=0)) / max_ang_lin
RMSE_ang_nl = np.sqrt((diff_ang_nl**2).mean(axis=0))  / max_ang_lin
RMSE_ang_planner = np.sqrt((diff_ang_planner**2).mean(axis=0)) / max_ang_lin

print("NORMALIZED RMSE : sqrt(  (measures - ref**2).mean() ) / measure_max ")

print('RMSE Velocity : [Vx, Vy] ')
print(RMSE_vel_lin[:2], ' : LINEAR')
print(RMSE_vel_nl[:2], ' : NON LINEAR')
print(RMSE_vel_planner[:2], ' : PLANNER')

print("\n\n")
print('RMSE POSITION : [Z] ')
print(RMSE_pos_lin[2], ' : LINEAR')
print(RMSE_pos_nl[2], ' : NON LINEAR')
print(RMSE_pos_planner[2], ' : PLANNER')

print("\n\n")
print('RMSE Roll / Pitch : ')
print(RMSE_rpy_lin[:2], ' : LINEAR')
print(RMSE_rpy_nl[:2], ' : NON LINEAR')
print(RMSE_rpy_planner[:2], ' : PLANNER')

print("\n\n")
print('RMSE Yaw Velocity : ')
print(RMSE_ang_lin[2], ' : LINEAR')
print(RMSE_ang_nl[2], ' : NON LINEAR')
print(RMSE_ang_planner[2], ' : PLANNER')





 # Display all graphs and wait
plt.show(block=True)