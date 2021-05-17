
import numpy as np
from matplotlib import pyplot as plt
import pinocchio as pin
import tsid as tsid
from IPython import embed

# For storage
log_feet_pos = []
log_feet_pos_target = []
log_feet_vel_target = []
log_feet_acc_target = []
log_x_cmd = []
log_x = []
log_q = []
log_dq = []
log_x_ref_invkin = []
log_x_invkin = []
log_dx_ref_invkin = []
log_dx_invkin = []
log_tau_ff = []
log_qdes = []
log_vdes = []
log_q_pyb = []
log_v_pyb = []

files = ["push_no_ff.npz", "push_with_ff.npz", "push_pyb_no_ff.npz", "push_pyb_with_ff.npz"]
for file in files:
    # Load data file
    data = np.load(file)
    # Store content of data in variables
    log_feet_pos.append(data["log_feet_pos"])
    log_feet_pos_target.append(data["log_feet_pos_target"])
    log_feet_vel_target.append(data["log_feet_vel_target"])
    log_feet_acc_target.append(data["log_feet_acc_target"])
    log_x_cmd.append(data["log_x_cmd"])
    log_x.append(data["log_x"])
    log_q.append(data["log_q"])
    log_dq.append(data["log_dq"])
    log_x_ref_invkin.append(data["log_x_ref_invkin"])
    log_x_invkin.append(data["log_x_invkin"])
    log_dx_ref_invkin.append(data["log_dx_ref_invkin"])
    log_dx_invkin.append(data["log_dx_invkin"])
    log_tau_ff.append(data["log_tau_ff"])
    log_qdes.append(data["log_qdes"])
    log_vdes.append(data["log_vdes"])
    log_q_pyb.append(data["log_q_pyb"])
    log_v_pyb.append(data["log_v_pyb"])

# Creating time vector
N = log_tau_ff[0].shape[1]
Tend = N * 0.001
t = np.linspace(0, Tend, N+1, endpoint=True)
t_range = t[:-1]

# Parameters
dt = 0.0020
lwdth = 2

##########
# GRAPHS #
##########

k_lgd = ["No FF"]
index6 = [1, 3, 5, 2, 4, 6]
index12 = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]

lgd_X = ["FL", "FR", "HL", "HR"]
lgd_Y = ["Pos X", "Pos Y", "Pos Z"]
plt.figure()
for i in range(12):
    if i == 0:
        ax0 = plt.subplot(3, 4, index12[i])
    else:
        plt.subplot(3, 4, index12[i], sharex=ax0)
    for k in range(len(log_tau_ff)):
        plt.plot(t_range, log_feet_pos[k][i % 3, np.int(i/3), :], color='b', linewidth=3, marker='')
        plt.plot(t_range, log_feet_pos_target[k][i % 3, np.int(i/3), :], color='r', linewidth=3, marker='')
        plt.legend([lgd_Y[i % 3] + " " + lgd_X[np.int(i/3)]+"", lgd_Y[i %
                                                                      3] + " " + lgd_X[np.int(i/3)]+" Ref"])
plt.suptitle("Reference positions of feet (world frame)")

lgd_X = ["FL", "FR", "HL", "HR"]
lgd_Y = ["Vel X", "Vel Y", "Vel Z"]
plt.figure()
for i in range(12):
    if i == 0:
        ax0 = plt.subplot(3, 4, index12[i])
    else:
        plt.subplot(3, 4, index12[i], sharex=ax0)
    for k in range(len(log_tau_ff)):
        plt.plot(t_range, log_feet_vel[k][i % 3, np.int(i/3), :], color='b', linewidth=3, marker='')
        plt.plot(t_range, log_feet_vel_target[k][i % 3, np.int(i/3), :], color='r', linewidth=3, marker='')
        plt.legend([lgd_Y[i % 3] + " " + lgd_X[np.int(i/3)]+" Ref"])
plt.suptitle("Current and reference velocities of feet (world frame)")

lgd_X = ["FL", "FR", "HL", "HR"]
lgd_Y = ["Acc X", "Acc Y", "Acc Z"]
plt.figure()
for i in range(12):
    if i == 0:
        ax0 = plt.subplot(3, 4, index12[i])
    else:
        plt.subplot(3, 4, index12[i], sharex=ax0)
    for k in range(len(log_tau_ff)):
        plt.plot(t_range, log_feet_acc_target[k][i % 3, np.int(i/3), :], color='r', linewidth=3, marker='')
    plt.legend([lgd_Y[i % 3] + " " + lgd_X[np.int(i/3)]+" Ref"])
plt.suptitle("Current and reference accelerations of feet (world frame)")

# LOG_Q
lgd = ["Position X", "Position Y", "Position Z", "Position Roll", "Position Pitch", "Position Yaw"]
plt.figure()
for i in range(6):
    if i == 0:
        ax0 = plt.subplot(3, 2, index6[i])
    else:
        plt.subplot(3, 2, index6[i], sharex=ax0)
    for k in range(len(log_tau_ff)):
        plt.plot(t_range[:-2], log_x[k][i, :-2], "b", linewidth=2)
        plt.plot(t_range[:-2], log_x_cmd[k][i, :-2], "r", linewidth=3)
        # plt.plot(t_range, log_q[i, :], "g", linewidth=2)
        plt.plot(t_range[:-2], log_x_invkin[k][i, :-2], "g", linewidth=2)
        plt.plot(t_range[:-2], log_x_ref_invkin[k][i, :-2], "violet", linewidth=2, linestyle="--")
    plt.legend(["WBC integrated output state", "Robot reference state",
                "Task current state", "Task reference state"])
    plt.ylabel(lgd[i])

# LOG_V
lgd = ["Linear vel X", "Linear vel Y", "Linear vel Z",
       "Angular vel Roll", "Angular vel Pitch", "Angular vel Yaw"]
plt.figure()
for i in range(6):
    if i == 0:
        ax0 = plt.subplot(3, 2, index6[i])
    else:
        plt.subplot(3, 2, index6[i], sharex=ax0)
    for k in range(len(log_tau_ff)):
        plt.plot(t_range[:-2], log_x[k][i+6, :-2], "b", linewidth=2)
        plt.plot(t_range[:-2], log_x_cmd[k][i+6, :-2], "r", linewidth=3)
        # plt.plot(t_range, log_dq[i, :], "g", linewidth=2)
        plt.plot(t_range[:-2], log_dx_invkin[k][i, :-2], "g", linewidth=2)
        plt.plot(t_range[:-2], log_dx_ref_invkin[k][i, :-2], "violet", linewidth=2, linestyle="--")
    plt.legend(["WBC integrated output state", "Robot reference state",
                "Task current state", "Task reference state"])
    plt.ylabel(lgd[i])

plt.figure()
for k in range(len(log_tau_ff)):
    plt.plot(t_range[:-2], log_x[k][6, :-2], "b", linewidth=2)
    plt.plot(t_range[:-2], log_x_cmd[k][6, :-2], "r", linewidth=2)
    plt.plot(t_range[:-2], log_dx_invkin[k][0, :-2], "g", linewidth=2)
    plt.plot(t_range[:-2], log_dx_ref_invkin[k][0, :-2], "violet", linewidth=2)
plt.legend(["WBC integrated output state", "Robot reference state",
            "Task current state", "Task reference state"])

k_c = ["r", "g", "b", "violet"]
plt.figure()
plt.plot(t_range[:-2], log_x_cmd[0][6, :-2], "k", linewidth=2)
for k in range(len(log_tau_ff)):
    plt.plot(t_range[:-2], log_v_pyb[k][1, :-2], color=k_c[k], linewidth=2)  # (1+len(log_tau_ff))-k)
plt.legend(["Reference", "Inv Kin FB no FF", "Inv Kin FB with FF", "PyB FB no FF", "PyB FB with FF"])

k_c = ["r", "g", "b", "violet"]
plt.figure()
plt.plot(t_range[:-2], log_x_cmd[0][1, :-2], "k", linewidth=2)
for k in range(len(log_tau_ff)):
    plt.plot(t_range[:-2], log_q_pyb[k][1, :-2], color=k_c[k], linewidth=2)  # (1+len(log_tau_ff))-k)
plt.xlabel("Time [s]")
plt.ylabel("Position [m]")
plt.legend(["Reference", "Inv Kin FB no FF", "Inv Kin FB with FF", "PyB FB no FF", "PyB FB with FF"])
plt.show(block=True)
