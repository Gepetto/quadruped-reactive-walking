# coding: utf8

import numpy as np
import utils
import time

from TSID_Debug_controller_four_legs_fb_vel import controller, dt
import processing as proc
import MPC_Wrapper
import pybullet as pyb


def run_scenario(envID, velID, dt_mpc, k_mpc, t, n_periods, T_gait, N_SIMULATION, type_MPC, pyb_feedback,
                 on_solo8, use_flat_plane, predefined_vel, enable_pyb_GUI):
    """Function that runs a simulation scenario based on a reference velocity profile, an environment and
    various parameters to define the gait

    Args:
        envID (int): identifier of the current environment to be able to handle different scenarios
        velID (int): identifier of the current velocity profile to be able to handle different scenarios
        dt_mpc (float): time step of the MPC
        k_mpc (int): number of iterations of inverse dynamics for one iteration of the MPC
        t (float): time of the simulation
        n_periods (int): number of gait periods in the prediction horizon of the MPC
        T_gait (float): duration of one gait period in seconds
        N_SIMULATION (int): number of iterations of inverse dynamics during the simulation
        type_mpc (bool): True to have PA's MPC, False to have Thomas's MPC
        pyb_feedback (bool): whether PyBullet feedback is enabled or not
        on_solo8 (bool): whether we are working on solo8 or not
        use_flat_plane (bool): to use either a flat ground or a rough ground
        predefined_vel (bool): to use either a predefined velocity profile or a gamepad
        enable_pyb_GUI (bool): to display PyBullet GUI or not
    """

    ########################################################################
    #                        Parameters definition                         #
    ########################################################################

    # Lists to log the duration of 1 iteration of the MPC/TSID
    t_list_filter = [0] * int(N_SIMULATION)
    t_list_states = [0] * int(N_SIMULATION)
    t_list_fsteps = [0] * int(N_SIMULATION)
    t_list_mpc = [0] * int(N_SIMULATION)
    t_list_tsid = [0] * int(N_SIMULATION)
    t_list_loop = [0] * int(N_SIMULATION)

    # Init joint torques to correct shape
    jointTorques = np.zeros((12, 1))

    # List to store the IDs of debug lines
    ID_deb_lines = []

    # Enable/Disable Gepetto viewer
    enable_gepetto_viewer = False

    # Create Joystick, FootstepPlanner, Logger and Interface objects
    joystick, fstep_planner, logger, interface, estimator = utils.init_objects(
        dt, dt_mpc, N_SIMULATION, k_mpc, n_periods, T_gait, type_MPC, on_solo8,
        predefined_vel)

    # Wrapper that makes the link with the solver that you want to use for the MPC
    # First argument to True to have PA's MPC, to False to have Thomas's MPC
    enable_multiprocessing = True
    mpc_wrapper = MPC_Wrapper.MPC_Wrapper(type_MPC, dt_mpc, fstep_planner.n_steps,
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
    # import ForceMonitor
    # myForceMonitor = ForceMonitor.ForceMonitor(pyb_sim.robotId, pyb_sim.planeId)

    ########################################################################
    #                             Simulator                                #
    ########################################################################

    # Define the default controller
    myController = controller(int(N_SIMULATION), k_mpc, n_periods, T_gait, on_solo8)

    tic = time.time()
    for k in range(int(N_SIMULATION)):

        time_loop = time.time()  # To analyze the time taken by each step

        # Process state estimator
        if k == 1:
            estimator.run_filter(k, fstep_planner.gait[0, 1:], pyb_sim.robotId,
                                 myController.invdyn.data(), myController.model)
        else:
            estimator.run_filter(k, fstep_planner.gait[0, 1:], pyb_sim.robotId)

        t_filter = time.time()  # To analyze the time taken by each step

        # Process state update and joystick
        proc.process_states(solo, k, k_mpc, velID, pyb_sim, interface, joystick, myController, estimator, pyb_feedback)

        t_states = time.time()  # To analyze the time taken by each step

        if np.isnan(interface.lC[2]):
            print("NaN value for the position of the center of mass. Simulation likely crashed. Ending loop.")
            break

        # Process footstep planner
        proc.process_footsteps_planner(k, k_mpc, pyb_sim, interface, joystick, fstep_planner)

        t_fsteps = time.time()  # To analyze the time taken by each step

        # Process MPC once every k_mpc iterations of TSID
        if (k % k_mpc) == 0:
            proc.process_mpc(k, k_mpc, interface, joystick, fstep_planner, mpc_wrapper,
                             dt_mpc, ID_deb_lines)

        # Retrieve reference contact forces
        if enable_multiprocessing or (k == 0):
            # Check if the MPC has outputted a new result
            f_applied = mpc_wrapper.get_latest_result()
        else:
            if (k % k_mpc) == 2:  # Mimic a 4 ms delay
                f_applied = mpc_wrapper.get_latest_result()

        t_mpc = time.time()  # To analyze the time taken by each step

        # Process Inverse Dynamics
        # If nothing wrong happened yet in TSID controller
        if not myController.error:
            proc.process_invdyn(solo, k, f_applied, pyb_sim, interface, fstep_planner,
                                myController, enable_hybrid_control, enable_gepetto_viewer)

            # Process PD+ (feedforward torques and feedback torques)
            jointTorques[:, 0] = proc.process_pdp(pyb_sim, myController, estimator)

        # If something wrong happened in TSID controller we stick to a security controller
        if myController.error:
            # D controller to slow down the legs
            D = 0.5
            jointTorques[:, 0] = D * (- pyb_sim.vmes12[6:, 0])

            # Saturation to limit the maximal torque
            t_max = 1.0
            jointTorques[jointTorques > t_max] = t_max
            jointTorques[jointTorques < -t_max] = -t_max

        t_tsid = time.time()  # To analyze the time taken by each step

        # Compute processing duration of each step
        t_list_filter[k] = t_filter - time_loop
        t_list_states[k] = t_states - t_filter
        t_list_fsteps[k] = t_fsteps - t_states
        t_list_mpc[k] = t_mpc - t_fsteps
        t_list_tsid[k] = t_tsid - t_mpc
        t_list_loop[k] = time.time() - time_loop

        # Process PyBullet
        proc.process_pybullet(pyb_sim, k, envID, velID, jointTorques)

        # Call logger object to log various parameters
        # logger.call_log_functions(k, pyb_sim, joystick, fstep_planner, interface, mpc_wrapper, myController,
        #                          False, pyb_sim.robotId, pyb_sim.planeId, solo)
        logger.log_state(k, pyb_sim, joystick, interface, mpc_wrapper, solo)
        # logger.log_forces(k, interface, myController, pyb_sim.robotId, pyb_sim.planeId)
        # logger.log_footsteps(k, interface, myController)
        # logger.log_fstep_planner(k, fstep_planner)
        # logger.log_tracking_foot(k, myController, solo)

        # Wait a bit to have simulated time = real time
        if k < 640:
            while (time.time() - time_loop) < dt:
                pass
        else:
            while (time.time() - time_loop) < 30*dt:
                pass


    ####################
    # END OF MAIN LOOP #
    ####################

    tac = time.time()
    print("Average computation time of one iteration: ", (tac-tic)/N_SIMULATION)
    print("Computation duration: ", tac-tic)
    print("Simulated duration: ", N_SIMULATION*dt)
    print("Max loop time: ", np.max(t_list_loop[10:]))

    # Close the parallel process if it is running
    if enable_multiprocessing:
        print("Stopping parallel process")
        mpc_wrapper.stop_parallel_loop()

    print("END")

    # Plot processing duration of each step of the control loop
    """
    import matplotlib.pylab as plt
    plt.figure()
    plt.plot(t_list_filter[1:], '+', color="orange")
    plt.plot(t_list_states[1:], 'r+')
    plt.plot(t_list_fsteps[1:], 'g+')
    plt.plot(t_list_mpc[1:], 'b+')
    plt.plot(t_list_tsid[1:], '+', color="violet")
    plt.plot(t_list_loop[1:], 'k+')
    plt.title("Time for state update + footstep planner + MPC communication + Inv Dyn + PD+")
    plt.show(block=True)"""

    # Disconnect the PyBullet server (also close the GUI)
    pyb.disconnect()

    # Plot graphs of the state estimator
    # estimator.plot_graphs()

    return logger
