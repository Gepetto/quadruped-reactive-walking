# coding: utf8

# import sys
# import os

# from sys import argv
# from matplotlib.cbook import ls_mapper

# sys.path.insert(0, os.getcwd())  # adds current directory to python path

import numpy as np
import matplotlib.pylab as plt
import quadruped_reactive_walking as qrw

# import crocoddyl_class.MPC_crocoddyl_planner as MPC_crocoddyl_planner
# import time
import utils_mpc

# import pinocchio as pin
from matplotlib.collections import LineCollection
import matplotlib
import LoggerSensors
from LoggerControl import LoggerControl


class Poly_5:
    """
    Class to evaluate 5th degree polynomial curve for the foot trajectory.
    x(0) = x0
    x'(0) = v0
    x''(0) = a0
    x(Dt) = x1
    Args :
        - x0 (float) : initial position
        - v0 (float) : initial velocity
        - a0 (float) : initial acceleration
        - x1 (float) : final position
        - Dt (float) : Time of the trajectory
    """

    def __init__(self, x0, v0, a0, x1, Dt) -> None:
        self.A0 = x0
        self.A1 = v0
        self.A2 = a0 / 2
        self.A3 = (20 * (x1 - x0) - 12 * Dt * v0 - 3 * a0 * Dt**2) / (2 * Dt**3)
        self.A4 = -(30 * (x1 - x0) - 16 * Dt * v0 - 3 * a0 * Dt**2) / (2 * Dt**4)
        self.A5 = (12 * (x1 - x0) - 6 * Dt * v0 - a0 * Dt**2) / (2 * Dt**5)

    def update_coeff(self, x0, v0, a0, x1, Dt):
        """
        Re-compute the internal coefficients
        Args :
            - x0 (float) : initial position
            - v0 (float) : initial velocity
            - a0 (float) : initial acceleration
            - x1 (float) : final position
            - Dt (float) : Time of the trajectory
        """
        self.A0 = x0
        self.A1 = v0
        self.A2 = a0 / 2
        self.A3 = (20 * (x1 - x0) - 12 * Dt * v0 - 3 * a0 * Dt**2) / (2 * Dt**3)
        self.A4 = -(30 * (x1 - x0) - 16 * Dt * v0 - 3 * a0 * Dt**2) / (2 * Dt**4)
        self.A5 = (12 * (x1 - x0) - 6 * Dt * v0 - a0 * Dt**2) / (2 * Dt**5)

    def compute(self, t, index):
        """
        Evaluate the trajectory : x(index)(t)
        Args :
            - t (float) : time
            - index (int) : Derivative order
        """
        if index == 0:
            return (
                self.A0
                + self.A1 * t
                + self.A2 * t**2
                + self.A3 * t**3
                + self.A4 * t**4
                + self.A5 * t**5
            )

        elif index == 1:
            return (
                self.A1
                + 2 * self.A2 * t
                + 3 * self.A3 * t**2
                + 4 * self.A4 * t**3
                + 5 * self.A5 * t**4
            )

        elif index == 2:
            return (
                2 * self.A2
                + 6 * self.A3 * t
                + 12 * self.A4 * t**2
                + 20 * self.A5 * t**3
            )

        else:
            return 6 * self.A3 + 24 * self.A4 * t + 60 * self.A5 * t**2


################
#  Parameters
################
iteration_mpc = 180  # Control cycle
iteration_init = iteration_mpc
params = qrw.Params()  # Object that holds all controller parameters
vert_time = params.vert_time

# Default position after calibration
q_init = np.array(params.q_init.tolist())

# Update I_mat, etc...
solo = utils_mpc.init_robot(q_init, params)

#######################
# Recover Logged data
#######################
file_name = (
    "/home/thomas_cbrs/Desktop/edin/quadruped-reactive-walking/scripts/"
    "crocoddyl_eval/logs/explore_weight_acc/data_cost.npz"
)

logs = np.load(file_name)
planner_gait = logs.get("planner_gait")
planner_xref = logs.get("planner_xref")
planner_fsteps = logs.get("planner_fsteps")
planner_goals = logs.get("planner_goals")
planner_vgoals = logs.get("planner_vgoals")
planner_agoals = logs.get("planner_agoals")
planner_jgoals = logs.get("planner_jgoals")
loop_o_q = logs.get("loop_o_q")
mpc_x_f = logs.get("mpc_x_f")

# Create loggers and process mocap
loggerSensors = LoggerSensors.LoggerSensors(logSize=20000 - 3)
logger = LoggerControl(0.001, 30, logSize=20000 - 3)
N_logger = logger.tstamps.shape[0]
logger.processMocap(N_logger, loggerSensors)
logger.loadAll(loggerSensors, fileName=file_name)

mocap_RPY = logger.mocap_RPY
mocap_pos = logger.mocap_pos

k_0 = int(
    iteration_mpc * (params.dt_mpc / params.dt_wbc)
)  # initial simulation iteration corresponding
N_mpc = 12  # Number of mpc to launch


t_init = k_0 * params.dt_wbc
t_end = k_0 * params.dt_wbc + N_mpc * params.dt_mpc
T = np.linspace(
    t_init, t_end - params.dt_wbc, int((t_end - t_init) / params.dt_wbc) + 1
)

plt.figure()
k_end = int(k_0 + N_mpc * params.dt_mpc / params.dt_wbc)

gait = planner_gait[k_0, :, :]
flying_feet = np.where(gait[0, :] == 0.0)[0]
foot = flying_feet[0]

lgd = [
    "Position X",
    "Velocity X",
    "Accleration X",
    "Jerk X",
    "Position Y",
    "Velocity Y",
    "Accleration Y",
    "Jerk Y",
]

plt.subplot(2, 4, 1)
plt.plot(T, planner_goals[k_0:k_end, 0, foot], linewidth=4, color="r")
plt.xlabel(lgd[0])
plt.subplot(2, 4, 2)
plt.plot(T, planner_vgoals[k_0:k_end, 0, foot], linewidth=4, color="r")
plt.xlabel(lgd[1])
plt.subplot(2, 4, 3)
plt.plot(T, planner_agoals[k_0:k_end, 0, foot], linewidth=4, color="r")
plt.xlabel(lgd[2])
plt.subplot(2, 4, 4)
plt.plot(T, planner_jgoals[k_0:k_end, 0, foot], linewidth=4, color="r")
plt.xlabel(lgd[3])

plt.subplot(2, 4, 5)
plt.plot(T, planner_goals[k_0:k_end, 1, foot], linewidth=4, color="r")
plt.xlabel(lgd[4])
plt.subplot(2, 4, 6)
plt.plot(T, planner_vgoals[k_0:k_end, 1, foot], linewidth=4, color="r")
plt.xlabel(lgd[5])
plt.subplot(2, 4, 7)
plt.plot(T, planner_agoals[k_0:k_end, 1, foot], linewidth=4, color="r")
plt.xlabel(lgd[6])
plt.subplot(2, 4, 8)
plt.plot(T, planner_jgoals[k_0:k_end, 1, foot], linewidth=4, color="r")
plt.xlabel(lgd[7])


cmap = plt.cm.get_cmap("hsv", N_mpc)

for i in range(N_mpc):

    k = (
        int(iteration_mpc * (params.dt_mpc / params.dt_wbc)) - 1
    )  # simulation iteration corresponding
    gait = planner_gait[k, :, :]
    id = 0
    while gait[id, foot] == 0:
        id += 1
    o_fsteps = mpc_x_f[k, 24 + 2 * foot : 24 + 2 * foot + 2, id + 1]
    print(k)
    x0 = planner_goals[k, 0, foot]
    y0 = planner_goals[k, 1, foot]
    xv0 = planner_vgoals[k, 0, foot]
    yv0 = planner_vgoals[k, 1, foot]
    xa0 = planner_agoals[k, 0, foot]
    ya0 = planner_agoals[k, 1, foot]
    dt_ = 0.24 - params.dt_mpc * i

    # part curve :
    part_curve = 0
    if dt_ >= 0.24 - vert_time:
        dt_ = 0.24 - 2 * vert_time
        part_curve = 0
    elif dt_ >= vert_time:
        dt_ -= vert_time
        part_curve = 1
    else:
        part_curve = 2
    poly_x = Poly_5(x0, xv0, xa0, o_fsteps[0], dt_)
    poly_y = Poly_5(y0, yv0, ya0, o_fsteps[1], dt_)

    T = np.linspace(0, dt_, 50)

    X = [poly_x.compute(t, 0) for t in T]
    VX = [poly_x.compute(t, 1) for t in T]
    AX = [poly_x.compute(t, 2) for t in T]
    JX = [poly_x.compute(t, 3) for t in T]

    Y = [poly_y.compute(t, 0) for t in T]
    VY = [poly_y.compute(t, 1) for t in T]
    AY = [poly_y.compute(t, 2) for t in T]
    JY = [poly_y.compute(t, 3) for t in T]

    if part_curve == 0:
        t0 = t_init + vert_time
        T += t0
    elif part_curve == 1:
        t0 = t_init + params.dt_mpc * i
        T += t0
    else:
        t0 = t_init + params.dt_mpc * i
        T += t0

    plt.subplot(2, 4, 1)
    plt.plot(T, X, color=cmap(i))
    plt.plot(T[0], X[0], marker="o", color=cmap(i))

    plt.subplot(2, 4, 2)
    plt.plot(T, VX, color=cmap(i))
    plt.plot(T[0], VX[0], marker="o", color=cmap(i))
    plt.subplot(2, 4, 3)
    plt.plot(T, AX, color=cmap(i))
    plt.plot(T[0], AX[0], marker="o", color=cmap(i))
    plt.subplot(2, 4, 4)
    plt.plot(T, JX, color=cmap(i))
    plt.plot(T[0], JX[0], marker="o", color=cmap(i))

    plt.subplot(2, 4, 5)
    plt.plot(T, Y, color=cmap(i))
    plt.plot(T[0], Y[0], marker="o", color=cmap(i))

    plt.subplot(2, 4, 6)
    plt.plot(T, VY, color=cmap(i))
    plt.plot(T[0], VY[0], marker="o", color=cmap(i))

    plt.subplot(2, 4, 7)
    plt.plot(T, AY, color=cmap(i))
    plt.plot(T[0], AY[0], marker="o", color=cmap(i))

    plt.subplot(2, 4, 8)
    plt.plot(T, JY, color=cmap(i))
    plt.plot(T[0], JY[0], marker="o", color=cmap(i))

    iteration_mpc += 1


fig_f, ax_f = plt.subplots(2, 4)
iteration_mpc = iteration_init
t_init = k_0 * params.dt_wbc
t_end = k_0 * params.dt_wbc + N_mpc * params.dt_mpc

for i in range(N_mpc):

    k = (
        int(iteration_mpc * (params.dt_mpc / params.dt_wbc)) - 1
    )  # simulation iteration corresponding
    gait = planner_gait[k, :, :]
    id = 0
    while gait[id, foot] == 0:
        id += 1
    o_fsteps = mpc_x_f[k, 24 + 2 * foot : 24 + 2 * foot + 2, id + 1]
    print(k)
    x0 = planner_goals[k, 0, foot]
    y0 = planner_goals[k, 1, foot]
    xv0 = planner_vgoals[k, 0, foot]
    yv0 = planner_vgoals[k, 1, foot]
    xa0 = planner_agoals[k, 0, foot]
    ya0 = planner_agoals[k, 1, foot]
    dt_ = 0.24 - params.dt_mpc * i

    # part curve :
    part_curve = 0
    if dt_ >= 0.24 - vert_time:
        dt_ = 0.24 - 2 * vert_time
        part_curve = 0
    elif dt_ >= vert_time:
        dt_ -= vert_time
        part_curve = 1
    else:
        part_curve = 2
    poly_x = Poly_5(x0, xv0, xa0, o_fsteps[0], dt_)
    poly_y = Poly_5(y0, yv0, ya0, o_fsteps[1], dt_)

    T = np.linspace(0, dt_, 20)

    X = [poly_x.compute(t, 0) for t in T]
    VX = [poly_x.compute(t, 1) for t in T]
    AX = [poly_x.compute(t, 2) for t in T]
    JX = [poly_x.compute(t, 3) for t in T]

    Y = [poly_y.compute(t, 0) for t in T]
    VY = [poly_y.compute(t, 1) for t in T]
    AY = [poly_y.compute(t, 2) for t in T]
    JY = [poly_y.compute(t, 3) for t in T]

    if part_curve == 0:
        t0 = t_init + vert_time
        T += t0
    elif part_curve == 1:
        t0 = t_init + params.dt_mpc * i
        T += t0
    else:
        t0 = t_init + params.dt_mpc * i
        T += t0

    # Set up lists of (x,y) points for predicted state i
    points_j = np.array([T, X]).transpose().reshape(-1, 1, 2)
    # Set up lists of segments
    segs_j = np.concatenate([points_j[:-1], points_j[1:]], axis=1)
    # Make collections segments
    cm = plt.get_cmap("Greys_r")
    lc_q = LineCollection(segs_j, cmap=cm, zorder=-1)
    lc_q.set_array(T)
    # Customize
    lc_q.set_linestyle("-")

    # Scatter to highlight points
    colors = np.r_[np.linspace(0.1, 0.65, 19), 1]
    my_colors = cm(colors)

    ax_f[0, 0].add_collection(lc_q)
    # Scatter to highlight points
    ax_f[0, 0].scatter(T, X, s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys)

    #####

    # Set up lists of (x,y) points for predicted state i
    points_j = np.array([T, VX]).transpose().reshape(-1, 1, 2)
    # Set up lists of segments
    segs_j = np.concatenate([points_j[:-1], points_j[1:]], axis=1)
    # Make collections segments
    cm = plt.get_cmap("Greys_r")
    lc_q = LineCollection(segs_j, cmap=cm, zorder=-1)
    lc_q.set_array(T)
    # Customize
    lc_q.set_linestyle("-")
    ax_f[0, 1].add_collection(lc_q)
    # Scatter to highlight points
    ax_f[0, 1].scatter(T, VX, s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys)

    #####

    # Set up lists of (x,y) points for predicted state i
    points_j = np.array([T, AX]).transpose().reshape(-1, 1, 2)
    # Set up lists of segments
    segs_j = np.concatenate([points_j[:-1], points_j[1:]], axis=1)
    # Make collections segments
    cm = plt.get_cmap("Greys_r")
    lc_q = LineCollection(segs_j, cmap=cm, zorder=-1)
    lc_q.set_array(T)
    # Customize
    lc_q.set_linestyle("-")
    ax_f[0, 2].add_collection(lc_q)
    # Scatter to highlight points
    ax_f[0, 2].scatter(T, AX, s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys)

    #####

    # Set up lists of (x,y) points for predicted state i
    points_j = np.array([T, JX]).transpose().reshape(-1, 1, 2)
    # Set up lists of segments
    segs_j = np.concatenate([points_j[:-1], points_j[1:]], axis=1)
    # Make collections segments
    cm = plt.get_cmap("Greys_r")
    lc_q = LineCollection(segs_j, cmap=cm, zorder=-1)
    lc_q.set_array(T)
    # Customize
    lc_q.set_linestyle("-")
    ax_f[0, 3].add_collection(lc_q)
    # Scatter to highlight points
    ax_f[0, 3].scatter(T, JX, s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys)

    ####################################################################################

    # Set up lists of (x,y) points for predicted state i
    points_j = np.array([T, Y]).transpose().reshape(-1, 1, 2)
    # Set up lists of segments
    segs_j = np.concatenate([points_j[:-1], points_j[1:]], axis=1)
    # Make collections segments
    cm = plt.get_cmap("Greys_r")
    lc_q = LineCollection(segs_j, cmap=cm, zorder=-1)
    lc_q.set_array(T)
    # Customize
    lc_q.set_linestyle("-")

    # Scatter to highlight points
    colors = np.r_[np.linspace(0.1, 0.65, 19), 1]
    my_colors = cm(colors)

    ax_f[1, 0].add_collection(lc_q)
    # Scatter to highlight points
    ax_f[1, 0].scatter(T, Y, s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys)

    #####

    # Set up lists of (x,y) points for predicted state i
    points_j = np.array([T, VY]).transpose().reshape(-1, 1, 2)
    # Set up lists of segments
    segs_j = np.concatenate([points_j[:-1], points_j[1:]], axis=1)
    # Make collections segments
    cm = plt.get_cmap("Greys_r")
    lc_q = LineCollection(segs_j, cmap=cm, zorder=-1)
    lc_q.set_array(T)
    # Customize
    lc_q.set_linestyle("-")
    ax_f[1, 1].add_collection(lc_q)
    # Scatter to highlight points
    ax_f[1, 1].scatter(T, VY, s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys)

    #####

    # Set up lists of (x,y) points for predicted state i
    points_j = np.array([T, AY]).transpose().reshape(-1, 1, 2)
    # Set up lists of segments
    segs_j = np.concatenate([points_j[:-1], points_j[1:]], axis=1)
    # Make collections segments
    cm = plt.get_cmap("Greys_r")
    lc_q = LineCollection(segs_j, cmap=cm, zorder=-1)
    lc_q.set_array(T)
    # Customize
    lc_q.set_linestyle("-")
    ax_f[1, 2].add_collection(lc_q)
    # Scatter to highlight points
    ax_f[1, 2].scatter(T, AY, s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys)

    #####

    # Set up lists of (x,y) points for predicted state i
    points_j = np.array([T, JY]).transpose().reshape(-1, 1, 2)
    # Set up lists of segments
    segs_j = np.concatenate([points_j[:-1], points_j[1:]], axis=1)
    # Make collections segments
    cm = plt.get_cmap("Greys_r")
    lc_q = LineCollection(segs_j, cmap=cm, zorder=-1)
    lc_q.set_array(T)
    # Customize
    lc_q.set_linestyle("-")
    ax_f[1, 3].add_collection(lc_q)
    # Scatter to highlight points
    ax_f[1, 3].scatter(T, JY, s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys)

    # plt.subplot(2,4,1)
    # plt.plot(T,X, color = cmap(i) )
    # plt.plot(T[0], X[0],marker = "o", color = cmap(i) )

    # plt.subplot(2,4,2)
    # plt.plot(T,VX, color = cmap(i) )
    # plt.plot(T[0], VX[0],marker = "o", color = cmap(i) )
    # plt.subplot(2,4,3)
    # plt.plot(T,AX, color = cmap(i) )
    # plt.plot(T[0], AX[0],marker = "o", color = cmap(i) )
    # plt.subplot(2,4,4)
    # plt.plot(T,JX, color = cmap(i) )
    # plt.plot(T[0], JX[0],marker = "o", color = cmap(i) )

    # plt.subplot(2,4,5)
    # plt.plot(T,Y, color = cmap(i) )
    # plt.plot(T[0], Y[0],marker = "o", color = cmap(i) )

    # plt.subplot(2,4,6)
    # plt.plot(T,VY, color = cmap(i) )
    # plt.plot(T[0], VY[0],marker = "o", color = cmap(i) )

    # plt.subplot(2,4,7)
    # plt.plot(T,AY, color = cmap(i) )
    # plt.plot(T[0], AY[0],marker = "o", color = cmap(i) )

    # plt.subplot(2,4,8)
    # plt.plot(T,JY, color = cmap(i) )
    # plt.plot(T[0], JY[0],marker = "o", color = cmap(i) )

    iteration_mpc += 1

t_init = k_0 * params.dt_wbc
t_end = k_0 * params.dt_wbc + N_mpc * params.dt_mpc
T = np.linspace(
    t_init, t_end - params.dt_wbc, int((t_end - t_init) / params.dt_wbc) + 1
)


ax_f[0, 0].plot(T, planner_goals[k_0:k_end, 0, foot], linewidth=2, color="b")
ax_f[0, 0].set(xlabel=lgd[0])
ax_f[0, 1].plot(T, planner_vgoals[k_0:k_end, 0, foot], linewidth=2, color="b")
ax_f[0, 1].set(xlabel=lgd[1])
ax_f[0, 2].plot(T, planner_agoals[k_0:k_end, 0, foot], linewidth=2, color="b")
ax_f[0, 2].set(xlabel=lgd[2])
ax_f[0, 3].plot(T, planner_jgoals[k_0:k_end, 0, foot], linewidth=2, color="b")
ax_f[0, 3].set(xlabel=lgd[3])


ax_f[1, 0].plot(T, planner_goals[k_0:k_end, 1, foot], linewidth=2, color="b")
ax_f[1, 0].set(xlabel=lgd[4])
ax_f[1, 1].plot(T, planner_vgoals[k_0:k_end, 1, foot], linewidth=2, color="b")
ax_f[1, 1].set(xlabel=lgd[5])
ax_f[1, 2].plot(T, planner_agoals[k_0:k_end, 1, foot], linewidth=2, color="b")
ax_f[1, 2].set(xlabel=lgd[6])
ax_f[1, 3].plot(T, planner_jgoals[k_0:k_end, 1, foot], linewidth=2, color="b")
ax_f[1, 3].set(xlabel=lgd[7])


iteration_mpc = iteration_init
t_0 = k_0 * params.dt_wbc
N = mpc_x_f.shape[2]

l_str = [
    "Position X",
    "Position Y",
    "Position Z",
    "Position Roll",
    "Position Pitch",
    "Position Yaw",
    "Linear velocity X",
    "Linear velocity Y",
    "Linear velocity Z",
    "Angular velocity Roll",
    "Angular velocity Pitch",
    "Angular velocity Yaw",
]


index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]

# plt.figure()
# for i in range(N_mpc):
# k = (
# int(iteration_mpc * (params.dt_mpc / params.dt_wbc)) - 1
# )  # simulation iteration corresponding
# xref = planner_xref[k].copy()  # From logged value
# xs = mpc_x_f[k, :12, :]
# xs = np.vstack([xref[:, 0], xs.transpose()]).transpose()

# t_init = t_0 + i * params.dt_mpc
# t_end = t_init + N * params.dt_mpc
# T = np.linspace(t_init, t_end, N + 1)

# for j in range(12):
# plt.subplot(3, 4, index[j])
# plt.plot(T, xs[j, :])

# iteration_mpc += 1


# for j in range(12):
#     plt.subplot(3,4,index[j])
#     plt.xlabel(l_str[j])

iteration_mpc = iteration_init
t_0 = k_0 * params.dt_wbc
N = mpc_x_f.shape[2]

data_xs_mpc = np.zeros((12, 25, N_mpc))
iteration_mpc = iteration_init
# Extract state prediction
for i in range(N_mpc):
    k = (
        int(iteration_mpc * (params.dt_mpc / params.dt_wbc)) - 1
    )  # simulation iteration corresponding
    xref = planner_xref[k].copy()  # From logged value
    xs = mpc_x_f[k, :12, :]
    xs = np.vstack([xref[:, 0], xs.transpose()]).transpose()
    data_xs_mpc[:, :, i] = xs
    iteration_mpc += 1


# index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]

fig_x, ax_x = plt.subplots(3, 4)
com_offset = 0.03
# For each states
for i in range(12):
    # Extract state prediction
    xs_i = data_xs_mpc[i, :, :]
    if i == 2:
        xs_i[1:, :] += com_offset
    # For each planning step in the trajectory
    for j in range(N_mpc):
        # Receding horizon = [j,j+N_h]
        t0_horizon = t_init + j * params.dt_mpc
        t_end = t0_horizon + N * params.dt_mpc
        tspan_x_pred = np.linspace(t0_horizon, t_end, N + 1)

        # Set up lists of (x,y) points for predicted state i
        points_j = np.array([tspan_x_pred, xs_i[:, j]]).transpose().reshape(-1, 1, 2)

        # Set up lists of segments
        segs_j = np.concatenate([points_j[:-1], points_j[1:]], axis=1)

        # Make collections segments
        cm = plt.get_cmap("Greys_r")
        lc_q = LineCollection(segs_j, cmap=cm, zorder=-1)

        lc_q.set_array(tspan_x_pred)

        # Customize
        lc_q.set_linestyle("-")

        # Scatter to highlight points
        colors = np.r_[np.linspace(0.1, 1, N), 1]
        my_colors = cm(colors)

        if i < 3:
            # Plot collections
            ax_x[i, 0].add_collection(lc_q)
            # Scatter to highlight points
            ax_x[i, 0].scatter(
                tspan_x_pred,
                xs_i[:, j],
                s=10,
                zorder=1,
                c=my_colors,
                cmap=matplotlib.cm.Greys,
            )

        elif i < 6:
            # Plot collections
            ax_x[i - 3, 1].add_collection(lc_q)
            # Scatter to highlight points
            ax_x[i - 3, 1].scatter(
                tspan_x_pred,
                xs_i[:, j],
                s=10,
                zorder=1,
                c=my_colors,
                cmap=matplotlib.cm.Greys,
            )

        elif i < 9:
            # Plot collections
            ax_x[i - 6, 2].add_collection(lc_q)
            # Scatter to highlight points
            ax_x[i - 6, 2].scatter(
                tspan_x_pred,
                xs_i[:, j],
                s=10,
                zorder=1,
                c=my_colors,
                cmap=matplotlib.cm.Greys,
            )

        else:
            # Plot collections
            ax_x[i - 9, 3].add_collection(lc_q)
            # Scatter to highlight points
            ax_x[i - 9, 3].scatter(
                tspan_x_pred,
                xs_i[:, j],
                s=10,
                zorder=1,
                c=my_colors,
                cmap=matplotlib.cm.Greys,
            )


t_init = k_0 * params.dt_wbc
t_end = k_0 * params.dt_wbc + N_mpc * params.dt_mpc
k_end = int(k_0 + N_mpc * params.dt_mpc / params.dt_wbc)
T = np.linspace(
    t_init, t_end - params.dt_wbc, int((t_end - t_init) / params.dt_wbc) + 1
)


####
# Measured & Reference position and orientation (ideal world frame)
####

index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
for i in range(12):
    if i < 2:
        pass
    elif i < 3:
        ax_x[i, 0].plot(T, planner_xref[k_0:k_end, i, 0], "b", linewidth=3)
        ax_x[i, 0].legend(["Robot state"], prop={"size": 8})
        ax_x[i, 0].set(xlabel=l_str[i])
    elif i < 6:
        ax_x[i - 3, 1].plot(T, planner_xref[k_0:k_end, i, 0], "b", linewidth=3)
        ax_x[i - 3, 1].legend(["Robot state"], prop={"size": 8})
        ax_x[i - 3, 1].set(xlabel=l_str[i])
    elif i < 9:
        ax_x[i - 6, 2].plot(T, planner_xref[k_0:k_end, i, 0], "b", linewidth=3)
        ax_x[i - 6, 2].legend(["Robot state"], prop={"size": 8})
        ax_x[i - 6, 2].set(xlabel=l_str[i])
    else:
        ax_x[i - 9, 3].plot(T, planner_xref[k_0:k_end, i, 0], "b", linewidth=3)
        ax_x[i - 9, 3].legend(["Robot state"], prop={"size": 8})
        ax_x[i - 9, 3].set(xlabel=l_str[i])


plt.show()


# T = np.linspace()
# for it in range(N_mpc):
# # simulation iteration corresponding
# k_0 = int(iteration_mpc * (params.dt_mpc / params.dt_wbc))
# gait = planner_gait[k_0]
