# coding: utf8

import numpy as np
import matplotlib.pylab as plt
import MPC_Wrapper
import types

####################
#  Initialization  #
####################

# Time step of the MPC
dt_mpc = 0.02

# Period of the MPC
T_mpc = 0.32

# Creation of the MPC object
enable_multiprocessing = False
mpc_wrapper = MPC_Wrapper.MPC_Wrapper(dt_mpc, np.int(T_mpc/dt_mpc), multiprocessing=enable_multiprocessing)

# Joystick object that contains the reference velocity in local frame
joystick = types.SimpleNamespace()  # Empty object
joystick.v_ref = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T

# MpcInterface object that contains information about the current state of the robot
mpc_interface = types.SimpleNamespace()  # Empty object
mpc_interface.lC = np.array([[0.0, 0.0, 0.2]]).T  # CoM centered and at 20 cm above the ground
mpc_interface.abg = np.array([[0.0, 0.0, 0.0]]).T  # horizontal base (roll, pitch, 0.0)
mpc_interface.lV = np.array([[0.0, 0.0, 0.0]]).T  # motionless base (linear velocity)
mpc_interface.lW = np.array([[0.0, 0.0, 0.0]]).T  # motionless base (angular velocity)
mpc_interface.l_feet = np.array([[0.19, 0.19, -0.19, -0.19],
                                 [0.15005, -0.15005, 0.15005, -0.15005],
                                 [0.0, 0.0, 0.0, 0.0]])  # position of feet in local frame

# FootstepPlanner object that contains information about the footsteps
fstep_planner = types.SimpleNamespace()  # Empty object
fstep_planner.x0 = np.vstack((mpc_interface.lC, mpc_interface.abg,
                              mpc_interface.lV, mpc_interface.lW))  # Current state vector
fstep_planner.xref = np.repeat(fstep_planner.x0, np.int(T_mpc/dt_mpc)+1, axis=1)  # Desired future state vectors
fstep_planner.fsteps = np.full((6, 13), np.nan)  # Array that contains information about the gait
fstep_planner.fsteps[:, 0] = np.zeros((6,))
fstep_planner.fsteps[0, 0] = np.int(T_mpc/dt_mpc)
fstep_planner.fsteps[0, 1:] = mpc_interface.l_feet.ravel(order="F")

#############
#  Run MPC  #
#############

# Run the MPC once to initialize internal matrices
mpc_wrapper.run_MPC(dt_mpc, np.int(T_mpc/dt_mpc), 0, T_mpc, T_mpc/2, joystick, fstep_planner, mpc_interface)

# Run the MPC to get the reference forces and the next predicted state
mpc_wrapper.run_MPC(dt_mpc, np.int(T_mpc/dt_mpc), 1, T_mpc, T_mpc/2, joystick, fstep_planner, mpc_interface)

# Output of the MPC
f_applied = mpc_wrapper.get_latest_result(1)

#####################
#  Display results  #
#####################

# enable_multiprocessing has to be "False" or else the mpc is in another process and we cannot retrieve the result easily

# Desired contact forces for the next time step
print("Desired forces: ", f_applied.ravel())

# Retrieve the "contact forces" part of the solution of the QP problem
f_predicted = mpc_wrapper.mpc.x[mpc_wrapper.mpc.xref.shape[0]*(mpc_wrapper.mpc.xref.shape[1]-1):].reshape((mpc_wrapper.mpc.xref.shape[0],
                                                                                                           mpc_wrapper.mpc.xref.shape[1]-1),
                                                                                                           order='F')

# Predicted evolution of state variables
l_t = np.linspace(dt_mpc, T_mpc, np.int(T_mpc/dt_mpc))
l_str = ["X", "Y", "Z", "Roll", "Pitch", "Yaw", "Vx", "Vy", "Vz", "VRoll", "VPitch", "VYaw"]
index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
plt.figure()
for i in range(12):
    plt.subplot(3, 4, index[i])
    plt.plot(l_t, mpc_wrapper.mpc.x_robot[i, :], linewidth=2, marker='x')
    plt.legend([l_str[i]])

# Desired evolution of contact forces
l_t = np.linspace(dt_mpc, T_mpc, np.int(T_mpc/dt_mpc))
l_str = ["FL_X", "FL_Y", "FL_Z", "FR_X", "FR_Y", "FR_Z", "HL_X", "HL_Y", "HL_Z", "HR_X", "HR_Y", "HR_Z"]
index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
plt.figure()
for i in range(12):
    plt.subplot(3, 4, index[i])
    plt.plot(l_t, f_predicted[i, :], linewidth=2, marker='x')
    plt.legend([l_str[i]])
plt.show(block=True)
