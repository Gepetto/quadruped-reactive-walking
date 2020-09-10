# coding: utf8

# import numpy as np
import matplotlib.pylab as plt
from TSID_Debug_controller_four_legs_fb_vel import dt
from main import run_scenario
# from IPython import embed

################################
# PARAMETERS OF THE SIMULATION #
################################

envID = 0  #  Identifier of the environment to choose in which one the simulation will happen
velID = 4  #  Identifier of the reference velocity profile to choose which one will be sent to the robot

dt_mpc = 0.02  # Time step of the MPC
k_mpc = int(dt_mpc / dt)  # dt is dt_tsid, defined in the TSID controller script
t = 0.0  # Time
n_periods = 1  # Number of periods in the prediction horizon
T_gait = 0.64  # Duration of one gait period
N_SIMULATION = 12000  # number of simulated TSID time steps

# Which MPC solver you want to use
# True to have PA's MPC, to False to have Thomas's MPC
type_MPC = True

# Whether PyBullet feedback is enabled or not
pyb_feedback = True

# Whether we are working with solo8 or not
on_solo8 = False

# If True the ground is flat, otherwise it has bumps
use_flat_plane = True

# If we are using a predefined reference velocity (True) or a joystick (False)
predefined_vel = True

# Enable or disable PyBullet GUI
enable_pyb_GUI = False

#################
# RUN SCENARIOS #
#################

# List to store the logger objects
result_loggers = []

# Run a scenario and retrieve data thanks to the logger
result_logger1 = run_scenario(envID, velID, dt_mpc, k_mpc, t, n_periods, T_gait, N_SIMULATION, type_MPC,
                              pyb_feedback, on_solo8, use_flat_plane, predefined_vel, enable_pyb_GUI)
result_loggers.append(result_logger1)

# Run a scenario and retrieve data thanks to the logger
result_logger2 = run_scenario(envID, velID, dt_mpc, k_mpc, t, n_periods, T_gait, N_SIMULATION, False,
                              pyb_feedback, on_solo8, use_flat_plane, predefined_vel, enable_pyb_GUI)
result_loggers.append(result_logger2)

# Display what has been logged by the loggers
# result_logger1.plot_graphs(enable_multiprocessing=False, show_block=False)
# result_logger2.plot_graphs(enable_multiprocessing=False)

# Only plot some graphs for debug purpose
# result_logger1.plot_state()
# result_logger1.plot_footsteps()
# result_logger1.plot_fstep_planner()
# result_logger1.plot_tracking_foot()
# result_logger1.plot_forces()
# result_logger1.plot_torques()
# result_logger2.plot_state()
# plt.show(block=True)

# embed()

###########
# Results #
###########
# quit()  # Logger 2 is commented so no need to compare the two scenarios
index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
lgd = ["Pos CoM X [m]", "Pos CoM Y [m]", "Pos CoM Z [m]", "Roll [deg]", "Pitch [deg]", "Yaw [deg]",
       "Lin Vel CoM X [m/s]", "Lin Vel CoM Y [m/s]", "Lin Vel CoM Z [m/s]", "Ang Vel Roll [deg/s]",
       "Ang Vel Pitch [deg/s]", "Ang Vel Yaw [deg/s]"]
plt.figure()
for i in range(12):
    if i == 0:
        ax0 = plt.subplot(3, 4, index[i])
    else:
        plt.subplot(3, 4, index[i], sharex=ax0)

    if i < 3:
        for logger in result_loggers:
            plt.plot(logger.t_range, logger.lC_pyb[i, :], linewidth=3)
    elif i < 6:
        for logger in result_loggers:
            plt.plot(logger.t_range, (180/3.1415)*logger.RPY_pyb[i-3, :], linewidth=3)
    elif i < 9:
        for logger in result_loggers:
            plt.plot(logger.t_range, logger.lV_pyb[i-6, :], linewidth=3)
    else:
        for logger in result_loggers:
            plt.plot(logger.t_range, (180/3.1415)*logger.lW_pyb[i-9, :], linewidth=3)

    if i in [2, 3, 4, 6, 7, 8, 9, 10, 11]:
        for logger in result_loggers:
            plt.plot(logger.t_range, logger.state_ref[i, :], "r", linewidth=3, linestyle="--")

    plt.xlabel("Time [s]")
    plt.ylabel(lgd[i])

    lgds = ["Logger " + str(j) for j in range(len(result_loggers))] + ["Reference TSID"]
    plt.legend(lgds)

    if i < 2:
        plt.ylim([-0.07, 0.07])
    elif i == 2:
        plt.ylim([0.16, 0.24])
    elif i < 6:
        plt.ylim([-10, 10])
    elif i == 6:
        plt.ylim([-0.05, 0.7])
    elif i == 7:
        plt.ylim([-0.1, 0.1])
    elif i == 8:
        plt.ylim([-0.2, 0.2])
    else:
        plt.ylim([-80.0, 80.0])

plt.suptitle("State of the robot in TSID Vs PyBullet Vs Reference (local frame)")
plt.show(block=True)
quit()
