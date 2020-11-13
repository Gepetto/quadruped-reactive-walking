# coding: utf8

import numpy as np
import utils_mpc
import time

# from TSID_Debug_controller_four_legs_fb_vel import controller
from QP_WBC import controller
import processing as proc
import MPC_Wrapper
import pybullet as pyb
from Planner import Planner
import pinocchio as pin
from Planner import EulerToQuaternion


class Result:

    def __init__(self):

        self.P = 0.0
        self.D = 0.0
        self.q_des = np.zeros(12)
        self.v_des = np.zeros(12)
        self.tau_ff = np.zeros(12)


class dummyHardware:

    def __init__(self):

        pass

    def imu_data_attitude(self, i):

        return 0.0


class dummyDevice:

    def __init__(self):

        self.hardware = dummyHardware()


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
        self.t_list_planner = [0] * int(N_SIMULATION)
        self.t_list_mpc = [0] * int(N_SIMULATION)
        self.t_list_wbc = [0] * int(N_SIMULATION)
        self.t_list_loop = [0] * int(N_SIMULATION)

        self.t_list_InvKin = [0] * int(N_SIMULATION)
        self.t_list_QPWBC = [0] * int(N_SIMULATION)
        self.t_list_intlog = [0] * int(N_SIMULATION)

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

        # Enable/Disable hybrid control
        self.enable_hybrid_control = True

        h_ref = 0.22294615
        self.q = np.zeros((19, 1))
        self.q[0:7, 0] = np.array([0.0, 0.0, h_ref, 0.0, 0.0, 0.0, 1.0])
        self.q[7:, 0] = q_init
        self.v = np.zeros((18, 1))
        self.b_v = np.zeros((18, 1))
        self.o_v_filt = np.zeros((18, 1))
        self.planner = Planner(dt_mpc, dt_tsid, n_periods,
                               T_gait, k_mpc, on_solo8, h_ref)

        # Wrapper that makes the link with the solver that you want to use for the MPC
        # First argument to True to have PA's MPC, to False to have Thomas's MPC
        self.enable_multiprocessing = True
        self.mpc_wrapper = MPC_Wrapper.MPC_Wrapper(type_MPC, dt_mpc, self.fstep_planner.n_steps,
                                                   k_mpc, self.fstep_planner.T_gait, self.q, self.enable_multiprocessing)

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
        # self.myController = controller(q_init, int(N_SIMULATION), dt_tsid, k_mpc, n_periods, T_gait, on_solo8)
        self.myController = controller(dt_tsid, int(N_SIMULATION))

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

        self.error_flag = 0
        self.q_security = np.array([np.pi*0.4, np.pi*80/180, np.pi] * 4)

        self.log_tmp1 = [0] * int(N_SIMULATION)
        self.log_tmp2 = [0] * int(N_SIMULATION)
        self.log_tmp3 = [0] * int(N_SIMULATION)
        self.log_tmp4 = [0] * int(N_SIMULATION)

        # Interface with the PD+ on the control board
        self.result = Result()

        # Run the control loop once with a dummy device for initialization
        dDevice = dummyDevice()
        dDevice.q_mes = q_init
        dDevice.v_mes = np.zeros(12)
        dDevice.baseLinearAcceleration = np.zeros(3)
        dDevice.baseAngularVelocity = np.zeros(3)
        dDevice.baseOrientation = np.array([0.0, 0.0, 0.0, 1.0])
        self.compute(dDevice)

    def compute(self, device):

        tic = time.time()

        # Update the reference velocity coming from the gamepad
        self.joystick.update_v_ref(self.k, self.velID)

        # Process state estimator
        self.estimator.run_filter(self.k, self.planner.gait[0, 1:], device, self.planner.goals, self.planner.gait[0, 0])

        t_filter = time.time()
        self.t_list_filter[self.k] = t_filter - tic

        # Update state for the next iteration of the whole loop
        if self.k > 1:
            self.q[:, 0] = self.estimator.q_filt[:, 0]
            oMb = pin.SE3(pin.Quaternion(self.q[3:7, 0:1]), self.q[0:3, 0:1])
            self.v[0:3, 0:1] = oMb.rotation @ self.estimator.v_filt[0:3, 0:1]
            self.v[3:6, 0:1] = oMb.rotation @ self.estimator.v_filt[3:6, 0:1]
            self.v[6:, 0] = self.estimator.v_filt[6:, 0]           

            # Update estimated position of the robot
            self.v_estim[0:3, 0:1] = oMb.rotation.transpose() @ self.joystick.v_ref[0:3, 0:1]
            self.v_estim[3:6, 0:1] = oMb.rotation.transpose() @ self.joystick.v_ref[3:6, 0:1]
            if not self.planner.is_static:
                self.q_estim[:, 0] = pin.integrate(self.solo.model,
                                                self.q, self.v_estim * self.myController.dt)
                self.yaw_estim = (utils_mpc.quaternionToRPY(self.q_estim[3:7, 0]))[2, 0]
            else:
                self.planner.q_static[:, 0] = pin.integrate(self.solo.model,
                                                self.planner.q_static, self.v_estim * self.myController.dt)
                self.planner.RPY_static[:, 0:1] = utils_mpc.quaternionToRPY(self.planner.q_static[3:7, 0])
        else:
            self.yaw_estim = 0.0
            self.q_estim = self.q.copy()
            oMb = pin.SE3(pin.Quaternion(self.q[3:7, 0:1]), self.q[0:3, 0:1])
            self.v_estim = self.v.copy()

        # Update position of PyBullet camera on the robot position to do as if it was attached to the robot
        if self.k > 10 and self.enable_pyb_GUI:
            # pyb.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=45, cameraPitch=-30,
            #                                cameraTargetPosition=[1.0, 0.3, 0.25])
            pyb.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=45, cameraPitch=-39.9,
                                           cameraTargetPosition=[device.dummyHeight[0], device.dummyHeight[1], 0.0])

        # Update reference height
        """if self.k < 1000:
            self.planner.h_ref += 0.00005"""

        # Run planner
        self.planner.run_planner(self.k, self.k_mpc, self.q[0:7, 0:1],
                                 self.v[0:6, 0:1].copy(), self.joystick.v_ref, self.q_estim[2, 0], 0.0, self.joystick)

        t_planner = time.time()
        self.t_list_planner[self.k] = t_planner - t_filter

        # Process MPC once every k_mpc iterations of TSID
        if (self.k % self.k_mpc) == 0:
            try:
                self.mpc_wrapper.solve(self.k, self.planner)
            except ValueError:
                print("MPC Problem")

        # Retrieve reference contact forces
        if self.enable_multiprocessing or (self.k == 0):
            # Check if the MPC has outputted a new result
            self.x_f_mpc = self.mpc_wrapper.get_latest_result()
        else:
            print("TODO: Check non multiprocessing mode.")
            if (self.k % self.k_mpc) == 2:  # Mimic a 4 ms delay
                self.f_applied = self.mpc_wrapper.get_latest_result()

        t_mpc = time.time()
        self.t_list_mpc[self.k] = t_mpc - t_planner

        # Target state for the inverse kinematics
        if not self.planner.is_static:
            self.x_f_mpc[0] = self.q_estim[0, 0]
            self.x_f_mpc[1] = self.q_estim[1, 0]
            self.x_f_mpc[2] = self.planner.h_ref
            self.x_f_mpc[3] = 0.0
            self.x_f_mpc[4] = 0.0
            self.x_f_mpc[5] = self.yaw_estim
        else:
            self.x_f_mpc[0:3] = self.planner.q_static[0:3, 0]
            self.x_f_mpc[3:6] = self.planner.RPY_static[:, 0]
        self.x_f_mpc[6:12] = self.planner.xref[6:, 1]

        self.estimator.x_f_mpc = self.x_f_mpc.copy()  # For logging

        # Whole Body Control
        # If nothing wrong happened yet in the WBC controller
        if (not self.myController.error) and (not self.joystick.stop):

            # Get velocity in base frame for pinocchio
            self.b_v[0:3, 0:1] = oMb.rotation.transpose() @ self.v[0:3, 0:1]
            self.b_v[3:6, 0:1] = oMb.rotation.transpose() @ self.v[3:6, 0:1]
            self.b_v[6:, 0] = self.v[6:, 0]

            # Run InvKin + WBC QP
            self.myController.compute(self.q, self.b_v, self.v, self.x_f_mpc[:12],
                                      self.x_f_mpc[12:], self.planner.gait[0, 1:], self.planner)

            # Quantities sent to the control board
            self.result.P = 3.0 * np.ones(12)
            self.result.D = 0.2 * np.ones(12)
            self.result.q_des[:] = self.myController.qdes[7:]
            self.result.v_des[:] = self.myController.vdes[6:, 0]
            self.result.tau_ff[:] = self.myController.tau_ff

            """if self.k % 5 == 0:
                self.solo.display(self.q)"""

        # Security check
        if (self.error_flag == 0) and (not self.myController.error) and (not self.joystick.stop):
            if np.any(np.abs(self.estimator.q_filt[7:, 0]) > self.q_security):
                self.myController.error = True
                self.error_flag = 1
                self.error_value = self.estimator.q_filt[7:, 0] * 180 / 3.1415
            if np.any(np.abs(self.estimator.v_secu) > 50):
                self.myController.error = True
                self.error_flag = 2
                self.error_value = self.estimator.v_secu
            if np.any(np.abs(self.myController.tau_ff) > 8):
                self.myController.error = True
                self.error_flag = 3
                self.error_value = self.myController.tau_ff

        # If something wrong happened in TSID controller we stick to a security controller
        if self.myController.error or self.joystick.stop:

            # Quantities sent to the control board
            self.result.P = np.zeros(12)
            self.result.D = 0.1 * np.ones(12)
            self.result.q_des[:] = np.zeros(12)
            self.result.v_des[:] = np.zeros(12)
            self.result.tau_ff[:] = np.zeros(12)

        # Log joystick command
        if self.joystick is not None:
            self.estimator.v_ref = self.joystick.v_ref

        self.t_list_wbc[self.k] = time.time() - t_mpc
        self.t_list_loop[self.k] = time.time() - tic
        self.t_list_InvKin[self.k] = self.myController.tac - \
            self.myController.tic
        self.t_list_QPWBC[self.k] = self.myController.toc - \
            self.myController.tac
        self.t_list_intlog[self.k] = self.myController.tuc - \
            self.myController.toc

        # Increment loop counter
        self.k += 1

        return 0.0

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

            # device.SetDesiredJointTorque(tau)
            device.SetKp(self.result.P)
            device.SetKd(self.result.D)
            device.SetQdes(self.result.q_des)
            device.SetVdes(self.result.v_des)
            # device.SetTauFF(self.result.tau_ff)

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
        print("Average computation time of one iteration: ",
              (tac-tic)/self.N_SIMULATION)
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
