# coding: utf8

import numpy as np
import utils_mpc
import time

from TSID_Debug_controller_four_legs_fb_vel import controller
import processing as proc
import MPC_Wrapper
import pybullet as pyb


class dummyDevice:

    def __init__(self):

        pass


class Controller:

    def __init__(self, q_init, envID, velID, dt_tsid, dt_mpc, k_mpc, t, n_periods, T_gait, N_SIMULATION, type_MPC,
                 pyb_feedback, on_solo8, use_flat_plane, predefined_vel, enable_pyb_GUI):
        """Function that runs a simulation scenario based on a reference velocity profile, an environment and
        various parameters to define the gait

        Args:
            envID (int): identifier of the current environment to be able to handle different scenarios
            velID (int): identifier of the current velocity profile to be able to handle different scenarios
            dt_tsid (float): time step of TSID
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
        self.t_list_filter = [0] * int(N_SIMULATION)
        self.t_list_states = [0] * int(N_SIMULATION)
        self.t_list_fsteps = [0] * int(N_SIMULATION)
        self.t_list_mpc = [0] * int(N_SIMULATION)
        self.t_list_tsid = [0] * int(N_SIMULATION)
        self.t_list_loop = [0] * int(N_SIMULATION)

        # Init joint torques to correct shape
        self.jointTorques = np.zeros((12, 1))

        # List to store the IDs of debug lines
        self.ID_deb_lines = []

        # Enable/Disable Gepetto viewer
        self.enable_gepetto_viewer = False

        # Create Joystick, FootstepPlanner, Logger and Interface objects
        self.joystick, self.fstep_planner, self.logger, self.interface, self.estimator = utils_mpc.init_objects(
            dt_tsid, dt_mpc, N_SIMULATION, k_mpc, n_periods, T_gait, type_MPC, on_solo8,
            predefined_vel)

        # Wrapper that makes the link with the solver that you want to use for the MPC
        # First argument to True to have PA's MPC, to False to have Thomas's MPC
        self.enable_multiprocessing = True
        self.mpc_wrapper = MPC_Wrapper.MPC_Wrapper(type_MPC, dt_mpc, self.fstep_planner.n_steps,
                                                   k_mpc, self.fstep_planner.T_gait, self.enable_multiprocessing)

        # Enable/Disable hybrid control
        self.enable_hybrid_control = True

        ########################################################################
        #                            Gepetto viewer                            #
        ########################################################################

        # Initialisation of the Gepetto viewer
        self.solo = utils_mpc.init_viewer(self.enable_gepetto_viewer)

        ########################################################################
        #                              PyBullet                                #
        ########################################################################

        # Initialisation of the PyBullet simulator
        # self.pyb_sim = utils_mpc.pybullet_simulator(envID, use_flat_plane, enable_pyb_GUI, dt=dt_tsid)

        # Force monitor to display contact forces in PyBullet with red lines
        # import ForceMonitor
        # myForceMonitor = ForceMonitor.ForceMonitor(pyb_sim.robotId, pyb_sim.planeId)

        ########################################################################
        #                             Simulator                                #
        ########################################################################

        # Define the default controller
        self.myController = controller(int(N_SIMULATION), dt_tsid, k_mpc, n_periods, T_gait, on_solo8)

        self.envID = envID
        self.velID = velID
        self.dt_tsid = dt_tsid
        self.dt_mpc = dt_mpc
        self.k_mpc = k_mpc
        self.t = t
        self.n_periods = n_periods
        self.T_gait = T_gait
        self.N_SIMULATION = N_SIMULATION
        self.type_MPC = type_MPC
        self.pyb_feedback = pyb_feedback
        self.on_solo8 = on_solo8
        self.use_flat_plane = use_flat_plane
        self.predefined_vel = predefined_vel
        self.enable_pyb_GUI = enable_pyb_GUI

        self.k = 0

        self.qmes12 = np.zeros((19, 1))
        self.vmes12 = np.zeros((18, 1))

        # Interface with the PD+ on the control board
        self.result.P = 0.0
        self.result.D = 0.0
        self.result.q_des = np.zeros(12)
        self.result.v_des = np.zeros(12)
        self.result.tau_ff = np.zeros(12)

        # Run the control loop once with a dummy device for initialization
        dDevice = dummyDevice()
        dDevice.q_mes = q_init
        dDevice.v_mes = np.zeros(12)
        dDevice.baseLinearAcceleration = np.zeros(3)
        dDevice.baseAngularVelocity = np.zeros(3)
        dDevice.baseOrientation = np.array([0.0, 0.0, 0.0, 1.0])
        self.compute(dDevice)

    def compute(self, device):

        # tic = time.time()
        # for k in range(int(self.N_SIMULATION)):

        # time_loop = time.time()  # To analyze the time taken by each step

        # Process state estimator
        if self.k == 1:
            self.estimator.run_filter(self.k, self.fstep_planner.gait[0, 1:], device,
                                      self.myController.invdyn.data(), self.myController.model)
        else:
            self.estimator.run_filter(self.k, self.fstep_planner.gait[0, 1:], device)

        # t_filter = time.time()  # To analyze the time taken by each step

        # Process state update and joystick
        proc.process_states(self.solo, self.k, self.k_mpc, self.velID, self.interface,
                            self.joystick, self.myController, self.estimator, self.pyb_feedback)

        # t_states = time.time()  # To analyze the time taken by each step

        if np.isnan(self.interface.lC[2]):
            print("NaN value for the position of the center of mass. Simulation likely crashed. Ending loop.")
            return np.zeros(12)

        # Process footstep planner
        proc.process_footsteps_planner(self.k, self.k_mpc, self.interface,
                                       self.joystick, self.fstep_planner)

        # t_fsteps = time.time()  # To analyze the time taken by each step

        # Process MPC once every k_mpc iterations of TSID
        if (self.k % self.k_mpc) == 0:
            proc.process_mpc(self.k, self.k_mpc, self.interface, self.joystick, self.fstep_planner, self.mpc_wrapper,
                             self.dt_mpc, self.ID_deb_lines)

        # Retrieve reference contact forces
        if self.enable_multiprocessing or (self.k == 0):
            # Check if the MPC has outputted a new result
            self.f_applied = self.mpc_wrapper.get_latest_result()
        else:
            if (self.k % self.k_mpc) == 2:  # Mimic a 4 ms delay
                self.f_applied = self.mpc_wrapper.get_latest_result()

        # t_mpc = time.time()  # To analyze the time taken by each step

        # Process Inverse Dynamics
        # If nothing wrong happened yet in TSID controller
        if (not self.myController.error) and (not self.joystick.stop):
            proc.process_invdyn(self.solo, self.k, self.f_applied, self.estimator, self.interface, self.fstep_planner,
                                self.myController, self.enable_hybrid_control, self.enable_gepetto_viewer)

            # Quantities sent to the control board
            self.result.P = 2.0 * np.ones(12)
            self.result.D = 0.1 * \
                np.array([1.0, 0.3, 0.3, 1.0, 0.3, 0.3,
                          1.0, 0.3, 0.3, 1.0, 0.3, 0.3])
            self.result.q_des[:] = self.myController.qdes[7:]
            self.result.v_des[:] = self.myController.vdes[6:]
            self.result.tau_ff[:] = self.myController.tau_ff

            # Process PD+ (feedforward torques and feedback torques)
            self.jointTorques[:, 0] = proc.process_pdp(self.myController, self.estimator)

        # If something wrong happened in TSID controller we stick to a security controller
        if self.myController.error or self.joystick.stop:

            # Quantities sent to the control board
            self.result.P = np.zeros(12)
            self.result.D = 0.05 * np.ones(12)
            self.result.q_des[:] = np.zeros(12)
            self.result.v_des[:] = np.zeros(12)
            self.result.tau_ff[:] = np.zeros(12)

            # D controller to slow down the legs
            D = 0.05
            self.jointTorques[:, 0] = D * (- self.estimator.v_filt[6:, 0])

            # Saturation to limit the maximal torque
            t_max = 1.0
            self.jointTorques[self.jointTorques > t_max] = t_max
            self.jointTorques[self.jointTorques < -t_max] = -t_max

        # t_tsid = time.time()  # To analyze the time taken by each step

        # Compute processing duration of each step
        """self.t_list_filter[self.k] = t_filter - time_loop
        self.t_list_states[self.k] = t_states - t_filter
        self.t_list_fsteps[self.k] = t_fsteps - t_states
        self.t_list_mpc[self.k] = t_mpc - t_fsteps
        self.t_list_tsid[self.k] = t_tsid - t_mpc
        self.t_list_loop[self.k] = time.time() - time_loop"""

        # Increment loop counter
        self.k += 1

        return self.jointTorques

    ####################
    # END OF MAIN LOOP #
    ####################

    def launch_simu(self, device):
        """# Default position after calibration
        q_init = np.array([0.0, 0.8, -1.6, 0, 0.8, -1.6, 0, -0.8, 1.6, 0, -0.8, 1.6])

        class dummyDevice:

            def __init__(self):

                pass

        dDevice = dummyDevice()
        dDevice.q_mes = q_init
        dDevice.v_mes = np.zeros(12)
        dDevice.baseLinearAcceleration = np.zeros(3)
        dDevice.baseAngularVelocity = np.zeros(3)
        dDevice.baseOrientation = np.array([0.0, 0.0, 0.0, 1.0])
        tau = self.compute(dDevice)"""

        tic = time.time()

        for k in range(1, int(self.N_SIMULATION)):

            device.UpdateMeasurment()

            tau = self.compute(device)

            device.SetDesiredJointTorque(tau)

            device.SendCommand(WaitEndOfCycle=True)

            print("###")
            print("TSID: ", self.myController.qtsid.ravel())
            print("TAU: ", tau.ravel())

            if self.enable_pyb_GUI:
                # Update the PyBullet camera on the robot position to do as if it was attached to the robot
                pyb.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=45, cameraPitch=-39.9,
                                               cameraTargetPosition=[self.estimator.q_filt[0, 0],
                                                                     self.estimator.q_filt[1, 0], 0.0])

            # Process PyBullet
            # proc.process_pybullet(self.pyb_sim, self.k, self.envID, self.velID, tau)

            # Call logger object to log various parameters
            # logger.call_log_functions(k, pyb_sim, joystick, fstep_planner, interface, mpc_wrapper, myController,
            #                          False, pyb_sim.robotId, pyb_sim.planeId, solo)
            # logger.log_state(k, pyb_sim, joystick, interface, mpc_wrapper, solo)
            # logger.log_forces(k, interface, myController, pyb_sim.robotId, pyb_sim.planeId)
            # logger.log_footsteps(k, interface, myController)
            # logger.log_fstep_planner(k, fstep_planner)
            # logger.log_tracking_foot(k, myController, solo)

        tac = time.time()
        print("Average computation time of one iteration: ", (tac-tic)/self.N_SIMULATION)
        print("Computation duration: ", tac-tic)
        print("Simulated duration: ", self.N_SIMULATION*self.dt_tsid)
        print("Max loop time: ", np.max(self.t_list_loop[10:]))

        # Close the parallel process if it is running
        if self.enable_multiprocessing:
            print("Stopping parallel process")
            self.mpc_wrapper.stop_parallel_loop()

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

        return self.logger
