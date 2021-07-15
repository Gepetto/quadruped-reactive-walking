# coding: utf8

import numpy as np
import utils_mpc
import time
import math

from QP_WBC import wbc_controller
import MPC_Wrapper
import pybullet as pyb
import pinocchio as pin
from solopython.utils.viewerClient import viewerClient, NonBlockingViewerFromRobot
import libquadruped_reactive_walking as lqrw
from example_robot_data.robots_loader import Solo12Loader

class Result:
    """Object to store the result of the control loop
    It contains what is sent to the robot (gains, desired positions and velocities,
    feedforward torques)"""

    def __init__(self):

        self.P = 0.0
        self.D = 0.0
        self.q_des = np.zeros(12)
        self.v_des = np.zeros(12)
        self.tau_ff = np.zeros(12)


class dummyHardware:
    """Fake hardware for initialisation purpose"""

    def __init__(self):

        pass

    def imu_data_attitude(self, i):

        return 0.0


class dummyDevice:
    """Fake device for initialisation purpose"""

    def __init__(self):

        self.hardware = dummyHardware()


class Controller:

    def __init__(self, params, q_init, t):  # , envID, velID, dt_wbc, dt_mpc, k_mpc, t, T_gait, T_mpc, N_SIMULATION, type_MPC, use_flat_plane, predefined_vel, enable_pyb_GUI, kf_enabled, N_gait, isSimulation, params):
        """Function that runs a simulation scenario based on a reference velocity profile, an environment and
        various parameters to define the gait

        Args:
            envID (int): identifier of the current environment to be able to handle different scenarios
            velID (int): identifier of the current velocity profile to be able to handle different scenarios
            dt_wbc (float): time step of the whole body control
            dt_mpc (float): time step of the MPC
            k_mpc (int): number of iterations of inverse dynamics for one iteration of the MPC
            t (float): time of the simulation
            T_gait (float): duration of one gait period in seconds
            T_mpc (float): duration of mpc prediction horizon
            N_SIMULATION (int): number of iterations of inverse dynamics during the simulation
            type_mpc (int): 0 for OSQP MPC, 1, 2, 3 for Crocoddyl MPCs
            use_flat_plane (bool): to use either a flat ground or a rough ground
            predefined_vel (bool): to use either a predefined velocity profile or a gamepad
            enable_pyb_GUI (bool): to display PyBullet GUI or not
            kf_enabled (bool): complementary filter (False) or kalman filter (True)
            N_gait (int): number of spare lines in the gait matrix
            isSimulation (bool): if we are in simulation mode
            params (Params object): store parameters
        """

        ########################################################################
        #                        Parameters definition                         #
        ########################################################################

        # Lists to log the duration of 1 iteration of the MPC/TSID
        self.t_list_filter = [0] * int(params.N_SIMULATION)
        self.t_list_planner = [0] * int(params.N_SIMULATION)
        self.t_list_mpc = [0] * int(params.N_SIMULATION)
        self.t_list_wbc = [0] * int(params.N_SIMULATION)
        self.t_list_loop = [0] * int(params.N_SIMULATION)

        self.t_list_InvKin = [0] * int(params.N_SIMULATION)
        self.t_list_QPWBC = [0] * int(params.N_SIMULATION)

        # Init joint torques to correct shape
        self.jointTorques = np.zeros((12, 1))

        # List to store the IDs of debug lines
        self.ID_deb_lines = []

        # Enable/Disable Gepetto viewer
        self.enable_gepetto_viewer = False
        '''if self.enable_gepetto_viewer:
            self.view = viewerClient()'''

        # Disable perfect estimator if we are not in simulation
        if not params.SIMULATION:
            params.perfectEstimator = False  # Cannot use perfect estimator if we are running on real robot

        # Initialisation of the solo model/data and of the Gepetto viewer
        self.solo = utils_mpc.init_robot(q_init, params, self.enable_gepetto_viewer)

        # Create Joystick, FootstepPlanner, Logger and Interface objects
        self.joystick, self.estimator = utils_mpc.init_objects(params)

        # Enable/Disable hybrid control
        self.enable_hybrid_control = True

        self.h_ref = params.h_ref
        self.q = np.zeros((19, 1))
        self.q[0:7, 0] = np.array([0.0, 0.0, self.h_ref, 0.0, 0.0, 0.0, 1.0])
        self.q[7:, 0] = q_init
        self.v = np.zeros((18, 1))
        self.b_v = np.zeros((18, 1))
        self.o_v_filt = np.zeros((18, 1))

        self.statePlanner = lqrw.StatePlanner()
        self.statePlanner.initialize(params)

        self.gait = lqrw.Gait()
        self.gait.initialize(params)

        self.footstepPlanner = lqrw.FootstepPlanner()
        self.footstepPlanner.initialize(params, self.gait)

        self.footTrajectoryGenerator = lqrw.FootTrajectoryGenerator()
        self.footTrajectoryGenerator.initialize(params, self.gait)

        # Wrapper that makes the link with the solver that you want to use for the MPC
        self.mpc_wrapper = MPC_Wrapper.MPC_Wrapper(params, self.q)

        # ForceMonitor to display contact forces in PyBullet with red lines
        # import ForceMonitor
        # myForceMonitor = ForceMonitor.ForceMonitor(pyb_sim.robotId, pyb_sim.planeId)

        # Define the default controller
        self.myController = wbc_controller(params)
        self.myController.qdes[7:] = q_init.ravel()

        self.envID = params.envID
        self.velID = params.velID
        self.dt_wbc = params.dt_wbc
        self.dt_mpc = params.dt_mpc
        self.k_mpc = int(params.dt_mpc / params.dt_wbc)
        self.t = t
        self.T_gait = params.T_gait
        self.T_mpc = params.T_mpc
        self.N_SIMULATION = params.N_SIMULATION
        self.type_MPC = params.type_MPC
        self.use_flat_plane = params.use_flat_plane
        self.predefined_vel = params.predefined_vel
        self.enable_pyb_GUI = params.enable_pyb_GUI

        self.k = 0

        self.qmes12 = np.zeros((19, 1))
        self.vmes12 = np.zeros((18, 1))

        self.v_ref = np.zeros((18, 1))
        self.h_v = np.zeros((18, 1))
        self.yaw_estim = 0.0
        self.RPY_filt = np.zeros(3)

        self.feet_a_cmd = np.zeros((3, 4))
        self.feet_v_cmd = np.zeros((3, 4))
        self.feet_p_cmd = np.zeros((3, 4))

        self.error_flag = 0
        self.q_security = np.array([np.pi*0.4, np.pi*80/180, np.pi] * 4)

        # Interface with the PD+ on the control board
        self.result = Result()

        # Run the control loop once with a dummy device for initialization
        dDevice = dummyDevice()
        dDevice.q_mes = q_init
        dDevice.v_mes = np.zeros(12)
        dDevice.baseLinearAcceleration = np.zeros(3)
        dDevice.baseAngularVelocity = np.zeros(3)
        dDevice.baseOrientation = np.array([0.0, 0.0, 0.0, 1.0])
        dDevice.dummyPos = np.array([0.0, 0.0, q_init[2]])
        dDevice.b_baseVel = np.zeros(3)
        self.compute(dDevice)

    def compute(self, device):
        """Run one iteration of the main control loop

        Args:
            device (object): Interface with the masterboard or the simulation
        """

        t_start = time.time()

        # Update the reference velocity coming from the gamepad
        self.joystick.update_v_ref(self.k, self.velID)

        # Process state estimator
        self.estimator.run_filter(self.k, self.gait.getCurrentGait(),
                                  device, self.footTrajectoryGenerator.getFootPosition())

        t_filter = time.time()

        # Update state vectors of the robot (q and v) + transformation matrices between world and horizontal frames
        oRh, oTh = self.updateState()

        # Update gait
        self.gait.updateGait(self.k, self.k_mpc, self.q[0:7, 0:1], self.joystick.joystick_code)

        # Compute target footstep based on current and reference velocities
        o_targetFootstep = self.footstepPlanner.updateFootsteps(self.k % self.k_mpc == 0 and self.k != 0,
                                                                int(self.k_mpc - self.k % self.k_mpc),
                                                                self.q[0:7, 0:1],
                                                                self.h_v[0:6, 0:1].copy(),
                                                                self.v_ref[0:6, 0])

        # Run state planner (outputs the reference trajectory of the base)
        self.statePlanner.computeReferenceStates(self.q[0:7, 0:1], self.h_v[0:6, 0:1].copy(),
                                                 self.v_ref[0:6, 0:1], 0.0)

        # Result can be retrieved with self.statePlanner.getReferenceStates()
        xref = self.statePlanner.getReferenceStates()
        fsteps = self.footstepPlanner.getFootsteps()
        cgait = self.gait.getCurrentGait()

        t_planner = time.time()

        # Solve MPC problem once every k_mpc iterations of the main loop
        if (self.k % self.k_mpc) == 0:
            try:
                if self.type_MPC == 3 :
                    # Compute the target foostep in local frame, to stop the optimisation around it when t_lock overpass
                    l_targetFootstep = self.footstepPlanner.getRz().transpose() @ self.footTrajectoryGenerator.getFootPosition() - self.q[0:3,0:1]
                    self.mpc_wrapper.solve(self.k, xref, fsteps, cgait, l_targetFootstep)
                else :
                    self.mpc_wrapper.solve(self.k, xref, fsteps, cgait, np.zeros((3,4)))

            except ValueError:
                print("MPC Problem")

        # Retrieve reference contact forces in horizontal frame
        self.x_f_mpc = self.mpc_wrapper.get_latest_result()
        """if self.k == 0:
            self.x_save = self.x_f_mpc[12:, :].copy()
        else:
            self.x_f_mpc[12:, :] = self.x_save.copy()"""

        """from IPython import embed
        embed()"""

        t_mpc = time.time()

        # If the MPC optimizes footsteps positions then we use them
        if self.k > 100 and self.type_MPC == 3 :
            for foot in range(4):
                id = 0
                while cgait[id,foot] == 0 :
                    id += 1
                o_targetFootstep[:2,foot] = np.array(self.footstepPlanner.getRz()[:2, :2]) @ self.x_f_mpc[24 +  2*foot:24+2*foot+2, id] + np.array([self.q[0, 0] , self.q[1,0] ])

        # Update pos, vel and acc references for feet
        self.footTrajectoryGenerator.update(self.k, o_targetFootstep)

        # Target state for the whole body control
        self.x_f_wbc = (self.x_f_mpc[:24, 0]).copy()
        if not self.gait.getIsStatic():
            self.x_f_wbc[0] = self.myController.dt * xref[6, 1]
            self.x_f_wbc[1] = self.myController.dt * xref[7, 1]
            self.x_f_wbc[2] = self.h_ref
            self.x_f_wbc[3] = 0.0
            self.x_f_wbc[4] = 0.0
            self.x_f_wbc[5] = self.myController.dt * xref[11, 1]
        else:  # Sort of position control to avoid slow drift
            self.x_f_wbc[0:3] = self.planner.q_static[0:3, 0]  # TODO: Adapt to new code
            self.x_f_wbc[3:6] = self.planner.RPY_static[:, 0]
        self.x_f_wbc[6:12] = xref[6:, 1]

        # Whole Body Control
        # If nothing wrong happened yet in the WBC controller
        if (not self.myController.error) and (not self.joystick.stop):

            self.q_wbc = np.zeros((19, 1))
            self.q_wbc[2, 0] = self.h_ref  # at position (0.0, 0.0, h_ref)
            self.q_wbc[6, 0] = 1.0  # with orientation (0.0, 0.0, 0.0)
            self.q_wbc[7:, 0] = self.myController.qdes[7:]  # with reference angular positions of previous loop

            # Get velocity in base frame for Pinocchio (not current base frame but desired base frame)
            self.b_v = self.v.copy()
            self.b_v[:6, 0] = self.v_ref[:6, 0]  # Base at reference velocity (TODO: add hRb once v_ref is considered in base frame)
            self.b_v[6:, 0] = self.myController.vdes[6:, 0]  # with reference angular velocities of previous loop

            # Feet command acceleration in base frame
            self.feet_a_cmd = oRh.transpose() @ self.footTrajectoryGenerator.getFootAcceleration() \
                - self.cross12(self.v_ref[3:6, 0:1], self.cross12(self.v_ref[3:6, 0:1], self.feet_p_cmd)) \
                - 2 * self.cross12(self.v_ref[3:6, 0:1], self.feet_v_cmd)

            # Feet command velocity in base frame
            self.feet_v_cmd = oRh.transpose() @ self.footTrajectoryGenerator.getFootVelocity()
            self.feet_v_cmd = self.feet_v_cmd - self.v_ref[0:3, 0:1] - self.cross12(self.v_ref[3:6, 0:1], self.feet_p_cmd)

            # Feet command position in base frame
            self.feet_p_cmd = oRh.transpose() @ (self.footTrajectoryGenerator.getFootPosition()
                                                 - np.array([[0.0], [0.0], [self.h_ref]]) - oTh)

            # Run InvKin + WBC QP
            self.myController.compute(self.q_wbc, self.b_v,
                                      self.x_f_wbc[12:], cgait[0, :],
                                      self.feet_p_cmd,
                                      self.feet_v_cmd,
                                      self.feet_a_cmd)

            # Quantities sent to the control board
            self.result.P = 6.0 * np.ones(12)
            self.result.D = 0.3 * np.ones(12)
            self.result.q_des[:] = self.myController.qdes[7:]
            self.result.v_des[:] = self.myController.vdes[6:, 0]
            self.result.tau_ff[:] = 0.8 * self.myController.tau_ff

            # Display robot in Gepetto corba viewer
            """if self.k % 5 == 0:
                self.solo.display(self.q)"""

        t_wbc = time.time()

        # Security check
        self.security_check()

        # Update PyBullet camera
        self.pyb_camera(device, 0.0)  # to have yaw update in simu: utils_mpc.quaternionToRPY(self.estimator.q_filt[3:7, 0])[2, 0]

        # Logs
        self.log_misc(t_start, t_filter, t_planner, t_mpc, t_wbc)

        # Increment loop counter
        self.k += 1

        return 0.0

    def pyb_camera(self, device, yaw):

        # Update position of PyBullet camera on the robot position to do as if it was attached to the robot
        if self.k > 10 and self.enable_pyb_GUI:
            # pyb.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=45, cameraPitch=-30,
            #                                cameraTargetPosition=[1.0, 0.3, 0.25])
            pyb.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=45, cameraPitch=-39.9,
                                           cameraTargetPosition=[device.dummyHeight[0], device.dummyHeight[1], 0.0])

    def security_check(self):

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
                print(self.result.P)
                print(self.result.D)
                print(self.result.q_des)
                print(self.result.v_des)
                print(self.myController.tau_ff)
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

    def log_misc(self, tic, t_filter, t_planner, t_mpc, t_wbc):

        # Log joystick command
        if self.joystick is not None:
            self.estimator.v_ref = self.joystick.v_ref

        self.t_list_filter[self.k] = t_filter - tic
        self.t_list_planner[self.k] = t_planner - t_filter
        self.t_list_mpc[self.k] = t_mpc - t_planner
        self.t_list_wbc[self.k] = t_wbc - t_mpc
        self.t_list_loop[self.k] = time.time() - tic
        self.t_list_InvKin[self.k] = self.myController.tac - self.myController.tic
        self.t_list_QPWBC[self.k] = self.myController.toc - self.myController.tac

    def updateState(self):

        # Update reference velocity vector
        self.v_ref[0:3, 0] = self.joystick.v_ref[0:3, 0]  # TODO: Joystick velocity given in base frame and not
        self.v_ref[3:6, 0] = self.joystick.v_ref[3:6, 0]  # in horizontal frame (case of non flat ground)
        self.v_ref[6:, 0] = 0.0

        # Update position and velocity state vectors
        if not self.gait.getIsStatic():
            # Integration to get evolution of perfect x, y and yaw
            Ryaw = np.array([[math.cos(self.yaw_estim), -math.sin(self.yaw_estim)],
                             [math.sin(self.yaw_estim), math.cos(self.yaw_estim)]])

            self.q[0:2, 0:1] = self.q[0:2, 0:1] + Ryaw @ self.v_ref[0:2, 0:1] * self.myController.dt

            # Mix perfect x and y with height measurement
            self.q[2, 0] = self.estimator.q_filt[2, 0]

            # Mix perfect yaw with pitch and roll measurements
            self.yaw_estim += self.v_ref[5, 0:1] * self.myController.dt
            self.q[3:7, 0] = self.estimator.EulerToQuaternion([self.estimator.RPY[0], self.estimator.RPY[1], self.yaw_estim])

            # Actuators measurements
            self.q[7:, 0] = self.estimator.q_filt[7:, 0]

            # Velocities are the one estimated by the estimator
            self.v = self.estimator.v_filt.copy()
            hRb = utils_mpc.EulerToRotation(self.estimator.RPY[0], self.estimator.RPY[1], 0.0)
            self.h_v[0:3, 0:1] = hRb @ self.v[0:3, 0:1]
            self.h_v[3:6, 0:1] = hRb @ self.v[3:6, 0:1]

            # self.v[:6, 0] = self.joystick.v_ref[:6, 0]
        else:
            quat = np.array([[0.0, 0.0, 0.0, 1.0]]).transpose()
            hRb = np.eye(3)
            pass
            # TODO: Adapt static mode to new version of the code

        # Transformation matrices between world and horizontal frames
        oRh = np.eye(3)
        c = math.cos(self.yaw_estim)
        s = math.sin(self.yaw_estim)
        oRh[0:2, 0:2] = np.array([[c, -s], [s, c]])
        oTh = np.array([[self.q[0, 0]], [self.q[1, 0]], [0.0]])

        return oRh, oTh

    def cross12(self, left, right):
        """Numpy is inefficient for this

        Args:
            left (3x1 array): left term of the cross product
            right (3x4 array): right term of the cross product
        """
        res = np.zeros((3, 4))
        for i in range(4):
            res[:, i] = np.array([left[1, 0] * right[2, i] - left[2, 0] * right[1, i],
                                  left[2, 0] * right[0, i] - left[0, 0] * right[2, i],
                                  left[0, 0] * right[1, i] - left[1, 0] * right[0, i]])
        return res
