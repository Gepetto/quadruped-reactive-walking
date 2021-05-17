# coding: utf8
import Logger
import pybullet as pyb
import MPC_Wrapper
import processing as proc
import ForceMonitor
import EmergencyStop_controller
import Safety_controller
from TSID_Debug_controller_four_legs_fb_vel import controller, dt
import time
import utils
import matplotlib.pylab as plt
import numpy as np
import sys
import os
from sys import argv
sys.path.insert(0, os.getcwd())  # adds current directory to python path


def run_scenario(envID, velID, dt_mpc, k_mpc, t, n_periods, T_gait, N_SIMULATION, type_MPC, pyb_feedback,
                 on_solo8, use_flat_plane, predefined_vel, enable_pyb_GUI):

    ########################################################################
    #                        Parameters definition                         #
    ########################################################################
    """# Time step
    dt_mpc = 0.02
    k_mpc = int(dt_mpc / dt)  # dt is dt_tsid, defined in the TSID controller script
    t = 0.0  # Time
    n_periods = 1  # Number of periods in the prediction horizon
    T_gait = 0.48  # Duration of one gait period
    # Simulation parameters
    N_SIMULATION = 1000  # number of time steps simulated"""

    # Initialize the error for the simulation time
    time_error = False

    # Lists to log the duration of 1 iteration of the MPC/TSID
    t_list_tsid = [0] * int(N_SIMULATION)
    t_list_loop = [0] * int(N_SIMULATION)
    t_list_mpc = [0] * int(N_SIMULATION)

    # List to store the IDs of debug lines
    ID_deb_lines = []

    # Enable/Disable Gepetto viewer
    enable_gepetto_viewer = True

    # Which MPC solver you want to use
    # True to have PA's MPC, to False to have Thomas's MPC
    """type_MPC = True"""

    # Create Joystick, FootstepPlanner, Logger and Interface objects
    joystick, fstep_planner, logger_ddp, interface, estimator = utils.init_objects(
        dt, dt_mpc, N_SIMULATION, k_mpc, n_periods, T_gait, type_MPC, on_solo8,
        predefined_vel)

    # Create a new logger type for the second solver
    logger_osqp = Logger.Logger(N_SIMULATION, dt, dt_mpc, k_mpc, n_periods, T_gait, True)

    # Wrapper that makes the link with the solver that you want to use for the MPC
    # First argument to True to have PA's MPC, to False to have Thomas's MPC
    enable_multiprocessing = True

    # Initialize the two algorithms
    mpc_wrapper_ddp = MPC_Wrapper.MPC_Wrapper(False, dt_mpc, fstep_planner.n_steps,
                                              k_mpc, fstep_planner.T_gait, enable_multiprocessing)

    mpc_wrapper_osqp = MPC_Wrapper.MPC_Wrapper(True, dt_mpc, fstep_planner.n_steps,
                                               k_mpc, fstep_planner.T_gait, enable_multiprocessing)

    # Enable/Disable hybrid control
    enable_hybrid_control = True

    ########################################################################
    #                            Gepetto viewer                            #
    ########################################################################

    # Initialisation of the Gepetto viewer
    solo = utils.init_viewer(enable_gepetto_viewer)

    ########################################################################
    #                              PyBullet                                #
    ########################################################################

    # Initialisation of the PyBullet simulator
    pyb_sim = utils.pybullet_simulator(envID, use_flat_plane, enable_pyb_GUI, dt=dt)

    # Force monitor to display contact forces in PyBullet with red lines
    myForceMonitor = ForceMonitor.ForceMonitor(pyb_sim.robotId, pyb_sim.planeId)

    ########################################################################
    #                             Simulator                                #
    ########################################################################

    # Define the default controller as well as emergency and safety controller
    myController = controller(int(N_SIMULATION), k_mpc, n_periods, T_gait, on_solo8)

    for k in range(int(N_SIMULATION)):
        time_loop = time.time()

        if (k % 1000) == 0:
            print("Iteration: ", k)

        # Process state estimator
        if k == 1:
            estimator.run_filter(k, fstep_planner.gait[0, 1:], pyb_sim.robotId,
                                 myController.invdyn.data(), myController.model)
        else:
            estimator.run_filter(k, fstep_planner.gait[0, 1:], pyb_sim.robotId)

        # Process states update and joystick
        proc.process_states(solo, k, k_mpc, velID, pyb_sim, interface, joystick, myController, estimator, pyb_feedback)

        if np.isnan(interface.lC[2, 0]):
            print("NaN value for the position of the center of mass. Simulation likely crashed. Ending loop.")
            break

        # Process footstep planner
        proc.process_footsteps_planner(k, k_mpc, pyb_sim, interface, joystick, fstep_planner)

        # Process MPC once every k_mpc iterations of TSID
        if (k % k_mpc) == 0:
            time_mpc = time.time()
            # Run both algorithms
            proc.process_mpc(k, k_mpc, interface, joystick, fstep_planner, mpc_wrapper_ddp,
                             dt_mpc, ID_deb_lines)
            proc.process_mpc(k, k_mpc, interface, joystick, fstep_planner, mpc_wrapper_osqp,
                             dt_mpc, ID_deb_lines)
            t_list_mpc[k] = time.time() - time_mpc

        if k == 0:
            f_applied = mpc_wrapper_ddp.get_latest_result()
        # elif (k % k_mpc) == 0:
        else:
            # Output of the MPC (with delay)
            f_applied = mpc_wrapper_ddp.get_latest_result()

        # Process Inverse Dynamics
        time_tsid = time.time()
        jointTorques = proc.process_invdyn(solo, k, f_applied, pyb_sim, interface, fstep_planner,
                                           myController, enable_hybrid_control, enable_gepetto_viewer)
        t_list_tsid[k] = time.time() - time_tsid  # Logging the time spent to run this iteration of inverse dynamics

        # Process PD+ (feedforward torques and feedback torques)
        for i_step in range(1):

            # Process the PD+
            jointTorques = proc.process_pdp(pyb_sim, myController, estimator)

            if myController.error:
                print('NaN value in feedforward torque. Ending loop.')
                break

            # Process PyBullet
            proc.process_pybullet(pyb_sim, k, envID, velID, jointTorques)

        # Call logger object to log various parameters
        logger_ddp.call_log_functions(k, pyb_sim, joystick, fstep_planner, interface, mpc_wrapper_ddp, myController,
                                      False, pyb_sim.robotId, pyb_sim.planeId, solo)

        if (k % k_mpc) == 0:
            logger_ddp.log_fstep_planner(k, fstep_planner)
            # logger_osqp.log_predicted_trajectories(k, mpc_wrapper_osqp)

        t_list_loop[k] = time.time() - time_loop

    ####################
    # END OF MAIN LOOP #
    ####################

    # Close the parallel process if it is running
    if enable_multiprocessing:
        print("Stopping parallel process")
        mpc_wrapper_osqp.stop_parallel_loop()

    print("END")

    pyb.disconnect()

    return logger_ddp, logger_osqp


"""# Display what has been logged by the logger
logger.plot_graphs(enable_multiprocessing=False)

quit()

# Display duration of MPC block and Inverse Dynamics block
plt.figure()
plt.plot(t_list_tsid, 'k+')
plt.title("Time TSID")
plt.show(block=True)"""
