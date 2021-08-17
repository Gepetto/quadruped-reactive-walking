# coding: utf8

import numpy as np
import utils_mpc
import time
import math

import MPC_Wrapper
import Joystick
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


class dummyIMU:
    """Fake IMU for initialisation purpose"""

    def __init__(self):

        self.linear_acceleration = np.zeros(3)
        self.gyroscope = np.zeros(3)
        self.attitude_euler = np.zeros(3)
        self.attitude_quaternion = np.zeros(4)


class dummyJoints:
    """Fake joints for initialisation purpose"""

    def __init__(self):

        self.positions = np.zeros(12)
        self.velocities = np.zeros(12)


class dummyDevice:
    """Fake device for initialisation purpose"""

    def __init__(self):

        self.hardware = dummyHardware()
        self.imu = dummyIMU()
        self.joints = dummyJoints()


class Controller:

    def __init__(self, params, q_init, t):
        """Function that runs a simulation scenario based on a reference velocity profile, an environment and
        various parameters to define the gait

        Args:
            params (Params object): store parameters
            q_init (array): initial position of actuators
            t (float): time of the simulation
        """

        ########################################################################
        #                        Parameters definition                         #
        ########################################################################

        # Init joint torques to correct shape
        self.jointTorques = np.zeros((12, 1))

        # List to store the IDs of debug lines
        self.ID_deb_lines = []

        # Disable perfect estimator if we are not in simulation
        if not params.SIMULATION:
            params.perfectEstimator = False  # Cannot use perfect estimator if we are running on real robot

        # Initialisation of the solo model/data and of the Gepetto viewer
        self.solo = utils_mpc.init_robot(q_init, params)

        # Create Joystick object
        self.joystick = Joystick.Joystick(params)

        # Enable/Disable hybrid control
        self.enable_hybrid_control = True

        self.h_ref = params.h_ref
        self.q = np.zeros((18, 1))  # Orientation part is in roll pitch yaw
        self.q[0:6, 0] = np.array([0.0, 0.0, self.h_ref, 0.0, 0.0, 0.0])
        self.q[6:, 0] = q_init
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

        self.estimator = lqrw.Estimator()
        self.estimator.initialize(params)

        self.wbcWrapper = lqrw.WbcWrapper()
        self.wbcWrapper.initialize(params)

        # Wrapper that makes the link with the solver that you want to use for the MPC
        self.mpc_wrapper = MPC_Wrapper.MPC_Wrapper(params, self.q)

        # ForceMonitor to display contact forces in PyBullet with red lines
        # import ForceMonitor
        # myForceMonitor = ForceMonitor.ForceMonitor(pyb_sim.robotId, pyb_sim.planeId)

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
        self.enable_corba_viewer = params.enable_corba_viewer
        self.Kp_main = params.Kp_main
        self.Kd_main = params.Kd_main
        self.Kff_main = params.Kff_main

        self.k = 0

        self.qmes12 = np.zeros((19, 1))
        self.vmes12 = np.zeros((18, 1))

        self.q_display = np.zeros((19, 1))
        self.v_ref = np.zeros((18, 1))
        self.h_v = np.zeros((18, 1))
        self.h_v_bis = np.zeros((6, 1))
        self.yaw_estim = 0.0
        self.RPY_filt = np.zeros(3)

        self.feet_a_cmd = np.zeros((3, 4))
        self.feet_v_cmd = np.zeros((3, 4))
        self.feet_p_cmd = np.zeros((3, 4))

        self.error = False  # True if something wrong happens in the controller
        self.error_flag = 0
        self.q_security = np.array([np.pi*0.4, np.pi*80/180, np.pi] * 4)

        self.q_filt_mpc = np.zeros((18, 1))
        self.h_v_filt_mpc = np.zeros((6, 1))
        self.h_v_bis_filt_mpc = np.zeros((6, 1))
        self.vref_filt_mpc = np.zeros((6, 1))
        self.filter_mpc_q = lqrw.Filter()
        self.filter_mpc_q.initialize(params)
        self.filter_mpc_v = lqrw.Filter()
        self.filter_mpc_v.initialize(params)
        self.filter_mpc_v_bis = lqrw.Filter()
        self.filter_mpc_v_bis.initialize(params)
        self.filter_mpc_vref = lqrw.Filter()
        self.filter_mpc_vref.initialize(params)

        # Interface with the PD+ on the control board
        self.result = Result()

        # Run the control loop once with a dummy device for initialization
        dDevice = dummyDevice()
        dDevice.joints.positions = q_init
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
        self.estimator.run_filter(self.gait.getCurrentGait(),
                                  self.footTrajectoryGenerator.getFootPosition(),
                                  device.imu.linear_acceleration.reshape((-1, 1)),
                                  device.imu.gyroscope.reshape((-1, 1)),
                                  device.imu.attitude_euler.reshape((-1, 1)),
                                  device.joints.positions.reshape((-1, 1)),
                                  device.joints.velocities.reshape((-1, 1)),
                                  np.zeros((3, 1)),  # device.dummyPos.reshape((-1, 1)),  # TODO: Case of real device
                                  np.zeros((3, 1)))  # device.b_baseVel.reshape((-1, 1)))

        # Update state vectors of the robot (q and v) + transformation matrices between world and horizontal frames
        self.estimator.updateState(self.joystick.v_ref, self.gait)
        oRh = self.estimator.getoRh()
        oTh = self.estimator.getoTh().reshape((3, 1))
        self.v_ref[0:6, 0] = self.estimator.getVRef()
        self.h_v[0:6, 0] = self.estimator.getHV()
        self.h_v_bis[0:6, 0] = self.estimator.getHVBis()
        self.q[:, 0] = self.estimator.getQUpdated()
        self.yaw_estim = self.estimator.getYawEstim()
        # TODO: Understand why using Python or C++ h_v leads to a slightly different result since the 
        # difference between them at each time step is 1e-16 at max (butterfly effect?)

        t_filter = time.time()

        # Update gait
        self.gait.updateGait(self.k, self.k_mpc, self.joystick.joystick_code)

        # Quantities go through a 1st order low pass filter with fc = 15 Hz (avoid >25Hz foldback)
        self.q_filt_mpc[:6, 0] = self.filter_mpc_q.filter(self.q[:6, 0:1], True)
        self.q_filt_mpc[6:, 0] = self.q[6:, 0].copy()
        self.h_v_filt_mpc[:, 0] = self.filter_mpc_v.filter(self.h_v[:6, 0:1], False)
        self.h_v_bis_filt_mpc[:, 0] = self.filter_mpc_v_bis.filter(self.h_v_bis[:6, 0:1], False)
        self.vref_filt_mpc[:, 0] = self.filter_mpc_vref.filter(self.v_ref[:6, 0:1], False)

        # Compute target footstep based on current and reference velocities
        o_targetFootstep = self.footstepPlanner.updateFootsteps(self.k % self.k_mpc == 0 and self.k != 0,
                                                                int(self.k_mpc - self.k % self.k_mpc),
                                                                self.q_filt_mpc[:, 0],
                                                                self.h_v_bis_filt_mpc[0:6, 0:1].copy(),
                                                                self.vref_filt_mpc[0:6, 0])

        # Run state planner (outputs the reference trajectory of the base)
        self.statePlanner.computeReferenceStates(self.q[0:6, 0:1], self.h_v_filt_mpc[0:6, 0:1].copy(),
                                                 self.vref_filt_mpc[0:6, 0:1], 0.0)

        # Result can be retrieved with self.statePlanner.getReferenceStates()
        xref = self.statePlanner.getReferenceStates()
        fsteps = self.footstepPlanner.getFootsteps()
        cgait = self.gait.getCurrentGait()

        t_planner = time.time()

        # TODO: Add 25Hz filter for the inputs of the MPC

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

        # Whole Body Control
        # If nothing wrong happened yet in the WBC controller
        if (not self.error) and (not self.joystick.stop):

            self.q_wbc = np.zeros((19, 1))
            self.q_wbc[2, 0] = self.h_ref  # at position (0.0, 0.0, h_ref)
            self.q_wbc[6, 0] = 1.0  # with orientation (0.0, 0.0, 0.0)
            self.q_wbc[7:, 0] = self.wbcWrapper.qdes[:]  # with reference angular positions of previous loop

            # Get velocity in base frame for Pinocchio (not current base frame but desired base frame)
            self.b_v = np.zeros((18, 1))
            self.b_v[:6, 0] = self.v_ref[:6, 0]  # Base at reference velocity (TODO: add hRb once v_ref is considered in base frame)
            self.b_v[6:, 0] = self.wbcWrapper.vdes[:]  # with reference angular velocities of previous loop

            # Feet command position, velocity and acceleration in base frame
            self.feet_a_cmd = self.footTrajectoryGenerator.getFootAccelerationBaseFrame(oRh.transpose(), self.v_ref[3:6, 0:1])
            self.feet_v_cmd = self.footTrajectoryGenerator.getFootVelocityBaseFrame(oRh.transpose(), self.v_ref[0:3, 0:1], self.v_ref[3:6, 0:1])
            self.feet_p_cmd = self.footTrajectoryGenerator.getFootPositionBaseFrame(oRh.transpose(), np.array([[0.0], [0.0], [self.h_ref]]) + oTh)

            # Run InvKin + WBC QP
            self.wbcWrapper.compute(self.q_wbc, self.b_v,
                                    (self.x_f_mpc[12:24, 0]).copy(), np.array([cgait[0, :]]),
                                    self.feet_p_cmd,
                                    self.feet_v_cmd,
                                    self.feet_a_cmd)

            # Quantities sent to the control board
            self.result.P = self.Kp_main * np.ones(12)
            self.result.D = self.Kd_main * np.ones(12)
            self.result.q_des[:] = self.wbcWrapper.qdes[:]
            self.result.v_des[:] = self.wbcWrapper.vdes[:]
            self.result.tau_ff[:] = self.Kff_main * self.wbcWrapper.tau_ff

            # Display robot in Gepetto corba viewer
            if self.enable_corba_viewer and (self.k % 5 == 0):
                self.q_display[:3, 0] = self.q[:3, 0]
                self.q_display[3:7, 0] = pin.Quaternion(pin.rpy.rpyToMatrix(self.q[3:6, 0])).coeffs()
                self.q_display[7:, 0] = self.q[6:, 0]
                self.solo.display(self.q_display)

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

        if (self.error_flag == 0) and (not self.error) and (not self.joystick.stop):
            self.error_flag = self.estimator.security_check(self.wbcWrapper.tau_ff)
            if (self.error_flag != 0):
                self.error = True
                if (self.error_flag == 1):
                    self.error_value = self.estimator.getQFilt()[7:] * 180 / 3.1415
                elif (self.error_flag == 2):
                    self.error_value = self.estimator.getVSecu()
                else:
                    self.error_value = self.wbcWrapper.tau_ff

        # If something wrong happened in the controller we stick to a security controller
        if self.error or self.joystick.stop:

            # Quantities sent to the control board
            self.result.P = np.zeros(12)
            self.result.D = 0.1 * np.ones(12)
            self.result.q_des[:] = np.zeros(12)
            self.result.v_des[:] = np.zeros(12)
            self.result.tau_ff[:] = np.zeros(12)

    def log_misc(self, tic, t_filter, t_planner, t_mpc, t_wbc):

        self.t_filter = t_filter - tic
        self.t_planner = t_planner - t_filter
        self.t_mpc = t_mpc - t_planner
        self.t_wbc = t_wbc - t_mpc
        self.t_loop = time.time() - tic
