# coding: utf8

import sys
import os
from sys import argv

sys.path.insert(0, os.getcwd())  # adds current directory to python path

import numpy as np
import matplotlib.pylab as plt
import quadruped_reactive_walking as qrw
import crocoddyl_class.MPC_crocoddyl as MPC_crocoddyl
import crocoddyl
import utils_mpc

np.set_printoptions(formatter={"float": lambda x: "{0:0.7f}".format(x)}, linewidth=450)

##############
#  Parameters
##############
iteration_mpc = 350  # Control cycle
Relaunch_DDP = True  # Compare a third MPC with != parameters
linear_mpc = True
params = qrw.Params()  # Object that holds all controller parameters

# Default position after calibration
q_init = np.array(params.q_init.tolist())

# Update I_mat, etc...
solo = utils_mpc.init_robot(q_init, params)

######################
# Recover Logged data
######################
file_name = "/home/thomas_cbrs/Desktop/edin/quadruped-reactive-walking/scripts/crocoddyl_eval/logs/explore_weight_acc/data_cost.npz"
logs = np.load(file_name)
planner_gait = logs.get("planner_gait")
planner_xref = logs.get("planner_xref")
planner_fsteps = logs.get("planner_fsteps")
mpc_x_f = logs.get("mpc_x_f")

k = int(
    iteration_mpc * (params.dt_mpc / params.dt_wbc)
)  # simulation iteration corresponding

# OSQP MPC
mpc_osqp = qrw.MPC(params)
mpc_osqp.run(0, planner_xref[0], planner_fsteps[0])  # Initialization of the matrix
# mpc_osqp.run(1, planner_xref[1] , planner_fsteps[1])
mpc_osqp.run(k, planner_xref[k], planner_fsteps[k])

osqp_xs = mpc_osqp.get_latest_result()[
    :12, :
]  # States computed over the whole predicted horizon
osqp_xs = np.vstack(
    [planner_xref[k, :, 0], osqp_xs.transpose()]
).transpose()  # Add current state
osqp_us = mpc_osqp.get_latest_result()[
    12:, :
]  # Forces computed over the whole predicted horizon

# DDP MPC
mpc_ddp = MPC_crocoddyl.MPC_crocoddyl(params, mu=0.9, inner=False, linearModel=True)
# Without warm-start :
# mpc_ddp.warm_start = False
# mpc_ddp.solve(k, planner_xref[k] , planner_fsteps[k] ) # Without warm-start

# Using warm-start logged :
# ( Exactly the function solve from MPC_crocoddyl)
# But cannot be used because there areis no previous value inside ddp.xs and ddp.us
planner_xref_offset = planner_xref[k].copy()
planner_xref_offset[2, :] += mpc_ddp.offset_com
mpc_ddp.updateProblem(planner_fsteps[k], planner_xref_offset)  # Update dynamics


print("\n\n")
print("model 0 :")
print(mpc_ddp.problem.runningModels[0].B)
x_init = []
u_init = []
mpc_ddp.ddp.solve(x_init, u_init, mpc_ddp.max_iteration)

k_previous = int((iteration_mpc - 1) * (params.dt_mpc / params.dt_wbc))

i = 0
while planner_gait[k, i, :].any():
    i += 1

for j in range(mpc_x_f[0, :, :].shape[1] - 1):
    u_init.append(
        mpc_x_f[k_previous, 12:24, j + 1]
    )  # Drag through an iteration (remove first)
u_init.append(
    np.repeat(planner_gait[k, i - 1, :], 3) * np.array(4 * [0.5, 0.5, 5.0])
)  # Add last node with average values, depending on the gait

for j in range(mpc_x_f[0, :, :].shape[1] - 1):
    x_init.append(
        mpc_x_f[k_previous, :12, j + 1]
    )  # Drag through an iteration (remove first)
x_init.append(mpc_x_f[k_previous, :12, -1])  # repeat last term
x_init.insert(
    0, planner_xref_offset[:, 0]
)  # With ddp, first value is the initial value

# Solve problem
mpc_ddp.ddp.solve(x_init, u_init, mpc_ddp.max_iteration)

ddp_xs = mpc_ddp.get_latest_result()[
    :12, :
]  # States computed over the whole predicted horizon
ddp_xs = np.vstack(
    [planner_xref_offset[:, 0], ddp_xs.transpose()]
).transpose()  # Add current state
ddp_us = mpc_ddp.get_latest_result()[
    12:, :
]  # Forces computed over the whole predicted horizon

######################################
# Relaunch DDP to adjust the gains
######################################

mpc_ddp_2 = MPC_crocoddyl.MPC_crocoddyl(
    params, mu=0.9, inner=False, linearModel=True
)  # To modify the linear model if wanted, recreate a list with proper model
mpc_ddp_2.implicit_integration = False
mpc_ddp_2.offset_com = 0.0
# mpc_ddp_2.stateWeights = np.sqrt([2.0, 2.0, 20.0, 0.25, 0.25, 1.0, 0.2, 0.2, 0.2, 0.0, 0.0, 0.3])
# mpc_ddp_2.shoulderWeights = 1.
# Weight Vector : State
# w_x = 0.2
# w_y = 0.2
# w_z = 2
# w_roll = 0.8
# w_pitch = 1.5
# w_yaw = 0.11
# w_vx =  1*np.sqrt(w_x)
# w_vy =  2*np.sqrt(w_y)
# w_vz =  1*np.sqrt(w_z)
# w_vroll =  0.05*np.sqrt(w_roll)
# w_vpitch =  0.05*np.sqrt(w_pitch)
# w_vyaw =  0.03*np.sqrt(w_yaw)
# mpc_ddp.stateWeight = np.array([w_x,w_y,w_z,w_roll,w_pitch,w_yaw,
#                             w_vx,w_vy,w_vz,w_vroll,w_vpitch,w_vyaw])
# OSQP values, in ddp formulation, terms are put in square
# mpc_ddp_2.stateWeight = np.sqrt([2.0, 2.0, 20.0, 0.25, 0.25, 10.0, 0.2, 0.2, 0.2, 0.0, 0.0, 0.3])

# Friction coefficient
# mpc_ddp_2.mu = 0.9
# mpc_ddp_2.max_iteration = 10
# mpc_ddp_2.warm_start = True

# Minimum normal force (N)
# mpc_ddp_2.min_fz = 0.2
# mpc_ddp_2.max_fz = 25

# Integration scheme
# V+ = V + dt*B*u   ; P+ = P + dt*V+ != explicit : P+ = P + dt*V
# mpc_ddp_2.implicit_integration = False

# Weight on the shoulder term :
# mpc_ddp_2.shoulderWeights = 0.
# mpc_ddp_2.shoulder_hlim = 0.27

# Weight Vector : Force Norm
# mpc_ddp.forceWeights = np.array(4*[0.01,0.01,0.01])
# mpc_ddp_2.forceWeights = np.sqrt(4*[0.00005, 0.00005, 0.00005]) # OSQP values

# Weight Vector : Friction cone cost
# mpc_ddp_2.frictionWeights = 0.5

# mpc_ddp_2.relative_forces = False

# Update weights and params inside the models
mpc_ddp_2.updateActionModels()
mpc_ddp_2.problem = crocoddyl.ShootingProblem(
    np.zeros(12), mpc_ddp_2.ListAction, mpc_ddp_2.terminalModel
)
mpc_ddp_2.ddp = crocoddyl.SolverDDP(mpc_ddp_2.problem)
# Run ddp solver
# Update the dynamic depending on the predicted feet position
planner_xref_offset = planner_xref[k]
planner_xref_offset[2, :] += mpc_ddp_2.offset_com

if Relaunch_DDP:
    mpc_ddp_2.updateProblem(planner_fsteps[k], planner_xref_offset)
    mpc_ddp_2.ddp.solve(x_init, u_init, mpc_ddp_2.max_iteration, isFeasible=False)

    ddp_xs_relaunch = mpc_ddp_2.get_latest_result()[
        :12, :
    ]  # States computed over the whole predicted horizon
    ddp_xs_relaunch = np.vstack(
        [planner_xref_offset[:, 0], ddp_xs_relaunch.transpose()]
    ).transpose()  # Add current state
    ddp_us_relaunch = mpc_ddp_2.get_latest_result()[
        12:, :
    ]  # Forces computed over the whole predicted horizon

#############
#  Plot     #
#############

# Predicted evolution of state variables
l_t = np.linspace(0.0, params.T_gait, np.int(params.T_gait / params.dt_mpc) + 1)
l_str = [
    "X_osqp",
    "Y_osqp",
    "Z_osqp",
    "Roll_osqp",
    "Pitch_osqp",
    "Yaw_osqp",
    "Vx_osqp",
    "Vy_osqp",
    "Vz_osqp",
    "VRoll_osqp",
    "VPitch_osqp",
    "VYaw_osqp",
]
l_str2 = [
    "X_ddp",
    "Y_ddp",
    "Z_ddp",
    "Roll_ddp",
    "Pitch_ddp",
    "Yaw_ddp",
    "Vx_ddp",
    "Vy_ddp",
    "Vz_ddp",
    "VRoll_ddp",
    "VPitch_ddp",
    "VYaw_ddp",
]

index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
plt.figure()
for i in range(12):
    plt.subplot(3, 4, index[i])

    (pl1,) = plt.plot(l_t, ddp_xs[i, :], linewidth=2, marker="x")
    (pl2,) = plt.plot(l_t, osqp_xs[i, :], linewidth=2, marker="x")

    if Relaunch_DDP:
        (pl3,) = plt.plot(l_t, ddp_xs_relaunch[i, :], linewidth=2, marker="x")
        plt.legend([pl1, pl2, pl3], [l_str2[i], l_str[i], "ddp_redo"])

    else:
        plt.legend([pl1, pl2], [l_str2[i], l_str[i]])


# Desired evolution of contact forces
l_t = np.linspace(params.dt_mpc, params.T_gait, np.int(params.T_gait / params.dt_mpc))
l_str = [
    "FL_X_osqp",
    "FL_Y_osqp",
    "FL_Z_osqp",
    "FR_X_osqp",
    "FR_Y_osqp",
    "FR_Z_osqp",
    "HL_X_osqp",
    "HL_Y_osqp",
    "HL_Z_osqp",
    "HR_X_osqp",
    "HR_Y_osqp",
    "HR_Z_osqp",
]
l_str2 = [
    "FL_X_ddp",
    "FL_Y_ddp",
    "FL_Z_ddp",
    "FR_X_ddp",
    "FR_Y_ddp",
    "FR_Z_ddp",
    "HL_X_ddp",
    "HL_Y_ddp",
    "HL_Z_ddp",
    "HR_X_ddp",
    "HR_Y_ddp",
    "HR_Z_ddp",
]
index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
plt.figure()
for i in range(12):
    plt.subplot(3, 4, index[i])
    (pl1,) = plt.plot(l_t, ddp_us[i, :], linewidth=2, marker="x")
    (pl2,) = plt.plot(l_t, osqp_us[i, :], linewidth=2, marker="x")

    if Relaunch_DDP:
        (pl3,) = plt.plot(l_t, ddp_us_relaunch[i, :], linewidth=2, marker="x")
        plt.legend([pl1, pl2, pl3], [l_str2[i], l_str[i], "ddp_redo"])

    else:
        plt.legend([pl1, pl2], [l_str2[i], l_str[i]])


plt.show(block=True)
