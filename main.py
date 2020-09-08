# coding: utf8

import numpy as np
import matplotlib.pylab as plt
import utils
import time

from TSID_Debug_controller_four_legs_fb_vel import controller, dt
import Safety_controller
import EmergencyStop_controller
import ForceMonitor
import processing as proc
import MPC_Wrapper
import pybullet as pyb

from loop import Loop


class SimulatorLoop(Loop):
    """
    Class used to call pybullet at a given frequency
    """

    def __init__(self, period, t_max, envID, velID, dt_mpc, k_mpc, t, n_periods, T_gait, N_SIMULATION, type_MPC, pyb_feedback):
        """
        Constructor
        :param period: the time step
        :param t_max: maximum simulation time
        """
        self.t = 0.0
        self.t_max = t_max
        self.period = period

        # Lists to log the duration of 1 iteration of the MPC/TSID
        self.t_list_states = [0] * int(N_SIMULATION)
        self.t_list_fsteps = [0] * int(N_SIMULATION)
        self.t_list_mpc = [0] * int(N_SIMULATION)
        self.t_list_tsid = [0] * int(N_SIMULATION)
        self.t_list_loop = [0] * int(N_SIMULATION)

        # List to store the IDs of debug lines
        self.ID_deb_lines = []

        # Enable/Disable Gepetto viewer
        self.enable_gepetto_viewer = False

        # Which MPC solver you want to use
        # True to have PA's MPC, to False to have Thomas's MPC
        """type_MPC = True"""

        # Create Joystick, FootstepPlanner, Logger and Interface objects
        self.joystick, self.fstep_planner, self.logger, self.interface, self.estimator = utils.init_objects(
            dt, dt_mpc, N_SIMULATION, k_mpc, n_periods, T_gait, type_MPC)

        # Wrapper that makes the link with the solver that you want to use for the MPC
        # First argument to True to have PA's MPC, to False to have Thomas's MPC
        self.enable_multiprocessing = False
        self.mpc_wrapper = MPC_Wrapper.MPC_Wrapper(type_MPC, dt_mpc, self.fstep_planner.n_steps,
                                                   k_mpc, self.fstep_planner.T_gait, self.enable_multiprocessing)

        # Enable/Disable hybrid control
        self.enable_hybrid_control = True

        # Initialisation of the Gepetto viewer
        self.solo = utils.init_viewer(self.enable_gepetto_viewer)

        # Initialisation of the PyBullet simulator
        self.pyb_sim = utils.pybullet_simulator(envID, dt=dt)

        # Force monitor to display contact forces in PyBullet with red lines
        self.myForceMonitor = ForceMonitor.ForceMonitor(self.pyb_sim.robotId, self.pyb_sim.planeId)

        # Define the default controller as well as emergency and safety controller
        self.myController = controller(int(N_SIMULATION), k_mpc, n_periods, T_gait)

        self.envID = envID
        self.velID = velID
        self.dt_mpc = dt_mpc
        self.k_mpc = k_mpc
        self.n_periods = n_periods
        self.T_gait = T_gait
        self.N_SIMULATION = N_SIMULATION
        self.type_MPC = type_MPC
        self.pyb_feedback = pyb_feedback
        self.k = -1

    def trigger(self):
        super().__init__(self.period)

    def loop(self, signum, frame):
        self.t += self.period
        if self.t > self.t_max:
            self.stop()

        self.k += 1

        self.time_loop = time.time()

        """if (k % 1000) == 0:
            print("Iteration: ", k)"""

        # Process states update and joystick
        proc.process_states(self.solo, self.k, self.k_mpc, self.velID, self.pyb_sim, self.interface, self.joystick,
                            self.myController, self.pyb_feedback)

        self.t_states = time.time()

        if np.isnan(self.interface.lC[2]):
            print("NaN value for the position of the center of mass. Simulation likely crashed. Ending loop.")
            self.stop()

        # Process footstep planner
        proc.process_footsteps_planner(self.k, self.k_mpc, self.pyb_sim, self.interface, self.joystick,
                                       self.fstep_planner)

        self.t_fsteps = time.time()

        # Process MPC once every k_mpc iterations of TSID
        if (self.k % self.k_mpc) == 0:
            # time_mpc = time.time()
            proc.process_mpc(self.k, self.k_mpc, self.interface, self.joystick, self.fstep_planner, self.mpc_wrapper,
                             self.dt_mpc, self.ID_deb_lines)
            # t_list_mpc[k] = time.time() - time_mpc

        if self.k == 0:
            self.f_applied = self.mpc_wrapper.get_latest_result()
        # elif (k % k_mpc) == 0:
        else:
            # Output of the MPC (with delay)
            self.f_applied = self.mpc_wrapper.get_latest_result()

        self.t_mpc = time.time()

        # Process Inverse Dynamics
        # time_tsid = time.time()
        # If nothing wrong happened yet in TSID controller
        if not self.myController.error:
            self.jointTorques = proc.process_invdyn(self.solo, self.k, self.f_applied, self.pyb_sim, self.interface,
                                                    self.fstep_planner, self.myController, self.enable_hybrid_control)
            # t_list_tsid[k] = time.time() - time_tsid  # Logging the time spent to run this iteration of inverse dynamics

            # Process PD+ (feedforward torques and feedback torques)
            self.jointTorques = proc.process_pdp(self.pyb_sim, self.myController)

        # If something wrong happened in TSID controller we stick to a security controller
        if self.myController.error:
            # D controller to slow down the legs
            self.D = 0.1
            self.jointTorques = self.D * (- self.pyb_sim.vmes12[6:, 0])

            # Saturation to limit the maximal torque
            self.tau_max = 1.0
            self.jointTorques[self.jointTorques > self.tau_max] = self.tau_max
            self.jointTorques[self.jointTorques < -self.tau_max] = -self.tau_max

        self.t_tsid = time.time()

        self.t_list_states[self.k] = self.t_states - self.time_loop
        self.t_list_fsteps[self.k] = self.t_fsteps - self.t_states
        self.t_list_mpc[self.k] = self.t_mpc - self.t_fsteps
        self.t_list_tsid[self.k] = self.t_tsid - self.t_mpc
        self.t_list_loop[self.k] = time.time() - self.time_loop

        # Process PyBullet
        proc.process_pybullet(self.pyb_sim, self.k, self.envID, self.jointTorques)

        # Call logger object to log various parameters
        # logger.call_log_functions(k, pyb_sim, joystick, fstep_planner, interface, mpc_wrapper, myController,
        #                         False, pyb_sim.robotId, pyb_sim.planeId, solo)


def run_scenarioo(envID, velID, dt_mpc, k_mpc, t, n_periods, T_gait, N_SIMULATION, type_MPC, pyb_feedback):

    # Start the control loop:
    sim_loop = SimulatorLoop(0.005, 5.0, envID, velID, dt_mpc, k_mpc, t, n_periods, T_gait, N_SIMULATION,
                             type_MPC, pyb_feedback)

    tic = time.time()
    sim_loop.loop(0, 0)

    sim_loop.trigger()
    tac = time.time()
    print("-- FINAL --")
    print("Average: ", (tac-tic)/sim_loop.k)
    # print("Computation duration: ", time.time()-tic)
    print("Simulated duration: ", N_SIMULATION*dt)
    print("Max loop time: ", np.max(sim_loop.t_list_loop[10:]))
    print("END")

    plt.figure()
    plt.plot(sim_loop.t_list_states[1:], 'r+')
    plt.plot(sim_loop.t_list_fsteps[1:], 'g+')
    plt.plot(sim_loop.t_list_mpc[1:], 'b+')
    plt.plot(sim_loop.t_list_tsid[1:], '+', color="violet")
    plt.plot(sim_loop.t_list_loop[1:], 'k+')
    plt.title("Time for state update + footstep planner + MPC communication + Inv Dyn + PD+")
    plt.show(block=True)

    pyb.disconnect()

    return sim_loop.logger


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

    # Which MPC solver you want to use
    # True to have PA's MPC, to False to have Thomas's MPC
    """type_MPC = True"""

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
    myForceMonitor = ForceMonitor.ForceMonitor(pyb_sim.robotId, pyb_sim.planeId)

    ########################################################################
    #                             Simulator                                #
    ########################################################################

    # Define the default controller
    myController = controller(int(N_SIMULATION), k_mpc, n_periods, T_gait, on_solo8)

    tic = time.time()
    for k in range(int(N_SIMULATION)):

        time_loop = time.time()

        """if (k % 1000) == 0:
            print("Iteration: ", k)"""
        # print("###")

        # Process estimator
        if k == 1:
            estimator.run_filter(k, fstep_planner.gait[0, 1:], pyb_sim.robotId,
                                 myController.invdyn.data(), myController.model)
        else:
            estimator.run_filter(k, fstep_planner.gait[0, 1:], pyb_sim.robotId)

        t_filter = time.time()

        # Process states update and joystick
        proc.process_states(solo, k, k_mpc, velID, pyb_sim, interface, joystick, myController, estimator, pyb_feedback)

        t_states = time.time()

        if np.isnan(interface.lC[2]):
            print("NaN value for the position of the center of mass. Simulation likely crashed. Ending loop.")
            break

        # Process footstep planner
        proc.process_footsteps_planner(k, k_mpc, pyb_sim, interface, joystick, fstep_planner)

        t_fsteps = time.time()

        # Process MPC once every k_mpc iterations of TSID
        if (k % k_mpc) == 0:
            # time_mpc = time.time()
            proc.process_mpc(k, k_mpc, interface, joystick, fstep_planner, mpc_wrapper,
                             dt_mpc, ID_deb_lines)
            # t_list_mpc[k] = time.time() - time_mpc

        if k == 0:
            f_applied = mpc_wrapper.get_latest_result()
        # elif (k % k_mpc) == 0:
        else:
            # Output of the MPC (with delay)
            f_applied = mpc_wrapper.get_latest_result()

        t_mpc = time.time()

        # print(f_applied.ravel())

        # Process Inverse Dynamics
        # time_tsid = time.time()
        # If nothing wrong happened yet in TSID controller
        if not myController.error:
            proc.process_invdyn(solo, k, f_applied, pyb_sim, interface, fstep_planner,
                                myController, enable_hybrid_control)
            # t_list_tsid[k] = time.time() - time_tsid  # Logging the time spent to run this iteration of inverse dynamics

            # Process PD+ (feedforward torques and feedback torques)
            jointTorques[:, 0] = proc.process_pdp(pyb_sim, myController)

        # If something wrong happened in TSID controller we stick to a security controller
        if myController.error:
            # D controller to slow down the legs
            D = 0.1
            jointTorques[:, 0] = D * (- pyb_sim.vmes12[6:, 0])

            # Saturation to limit the maximal torque
            t_max = 1.0
            jointTorques[jointTorques > t_max] = t_max
            jointTorques[jointTorques < -t_max] = -t_max

            """if np.max(np.abs(pyb_sim.vmes12[6:, 0])) < 0.1:
                print("Trigger get back to default pos")
                # pyb_sim.get_to_default_position(pyb_sim.straight_standing)
                pyb_sim.get_to_default_position(np.array([[0.0, np.pi/2, np.pi,
                                                           0.0, np.pi/2, np.pi,
                                                           0.0, -np.pi/2, np.pi,
                                                           0.0, -np.pi/2, np.pi]]).transpose())"""
            """pyb_sim.get_to_default_position(np.array([[0, 0.8, -1.6,
                                                       0, 0.8, -1.6,
                                                       0, -0.8, 1.6,
                                                       0, -0.8, 1.6]]).transpose())"""

        # print(jointTorques.ravel())
        """if myController.error:
            pyb.disconnect()
            quit()"""

        t_tsid = time.time()

        t_list_filter[k] = t_filter - time_loop
        t_list_states[k] = t_states - t_filter
        t_list_fsteps[k] = t_fsteps - t_states
        t_list_mpc[k] = t_mpc - t_fsteps
        t_list_tsid[k] = t_tsid - t_mpc
        t_list_loop[k] = time.time() - time_loop

        # Process PyBullet
        proc.process_pybullet(pyb_sim, k, envID, jointTorques)

        # Call logger object to log various parameters
        # logger.call_log_functions(k, pyb_sim, joystick, fstep_planner, interface, mpc_wrapper, myController,
        #                          False, pyb_sim.robotId, pyb_sim.planeId, solo)
        # logger.log_state(k, pyb_sim, joystick, interface, mpc_wrapper, solo)
        # logger.log_footsteps(k, interface, myController)
        # logger.log_fstep_planner(k, fstep_planner)
        # logger.log_tracking_foot(k, myController, solo)
        """if k >= 1000:
            time.sleep(0.1)"""

        while (time.time() - time_loop) < 0.002:
            pass

    ####################
    # END OF MAIN LOOP #
    ####################

    tac = time.time()
    print("Average: ", (tac-tic)/N_SIMULATION)
    print("Computation duration: ", tac-tic)
    print("Simulated duration: ", N_SIMULATION*dt)
    print("Max loop time: ", np.max(t_list_loop[10:]))

    if enable_multiprocessing:
        print("Stopping parallel process")
        mpc_wrapper.stop_parallel_loop()

    print("END")

    """plt.figure()
    plt.plot(t_list_filter[1:], '+', color="orange")
    plt.plot(t_list_states[1:], 'r+')
    plt.plot(t_list_fsteps[1:], 'g+')
    plt.plot(t_list_mpc[1:], 'b+')
    plt.plot(t_list_tsid[1:], '+', color="violet")
    plt.plot(t_list_loop[1:], 'k+')
    plt.title("Time for state update + footstep planner + MPC communication + Inv Dyn + PD+")
    plt.show(block=True)"""

    pyb.disconnect()

    NN = estimator.log_v_est.shape[2]
    avg = np.zeros((3, NN))
    for m in range(NN):
        tmp_cpt = 0
        tmp_sum = np.zeros((3, 1))
        for j in range(4):
            if np.any(np.abs(estimator.log_v_est[:, j, m]) > 1e-2):
                tmp_cpt += 1
                tmp_sum[:, 0] = tmp_sum[:, 0] + estimator.log_v_est[:, j, m].ravel()
        if tmp_cpt > 0:
            avg[:, m:(m+1)] = tmp_sum / tmp_cpt

    plt.figure()
    for i in range(3):
        plt.subplot(3, 1, i+1)
        for j in range(4):
            plt.plot(estimator.log_v_est[i, j, :], linewidth=3)
            # plt.plot(-myController.log_Fv1F[i, j, :], linewidth=3, linestyle="--")
        plt.plot(avg[i, :], color="rebeccapurple", linewidth=3, linestyle="--")
        plt.plot(estimator.log_v_truth[i, :], "k", linewidth=3, linestyle="--")
        plt.plot(estimator.log_filt_lin_vel[i, :], color="darkgoldenrod", linewidth=3, linestyle="--")
        plt.legend(["FL", "FR", "HL", "HR", "Avg", "Truth", "Filtered"])
        plt.xlim([2000, 8000])
    plt.suptitle("Estimation of the linear velocity of the trunk (in base frame)")

    """plt.figure()
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(estimator.log_filt_lin_vel[i, :], color="red", linewidth=3)
        plt.plot(estimator.log_filt_lin_vel_bis[i, :], color="forestgreen", linewidth=3)
        plt.plot(estimator.rotated_FK[i, :], color="blue", linewidth=3)
        plt.legend(["alpha = 1.0", "alpha = 450/500"])
    plt.suptitle("Estimation of the velocity of the base")"""

    """plt.figure()
    for i in range(3):
        plt.subplot(3, 1, i+1)
        for j in range(4):
            plt.plot(logger.feet_vel[i, j, :], linewidth=3)
    plt.suptitle("Velocity of feet over time")"""
    plt.show(block=True)

    return logger


"""# Display what has been logged by the logger
logger.plot_graphs(enable_multiprocessing=False)

quit()

# Display duration of MPC block and Inverse Dynamics block
plt.figure()
plt.plot(t_list_tsid, 'k+')
plt.title("Time TSID")
plt.show(block=True)"""
