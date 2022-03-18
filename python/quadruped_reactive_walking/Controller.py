import time

import numpy as np
import pinocchio as pin
import pybullet as pyb

from . import MPC_Wrapper, quadruped_reactive_walking as qrw
from .solo3D.utils import quaternionToRPY
from .tools.utils_mpc import init_robot


class Result:
    """
    Object to store the result of the control loop
    It contains what is sent to the robot (gains, desired positions and velocities,
    feedforward torques)
    """

    def __init__(self, params):
        self.P = np.array(params.Kp_main.tolist() * 4)
        self.D = np.array(params.Kd_main.tolist() * 4)
        self.FF = params.Kff_main * np.ones(12)
        self.q_des = np.zeros(12)
        self.v_des = np.zeros(12)
        self.tau_ff = np.zeros(12)


class DummyDevice:
    def __init__(self):
        self.imu = self.IMU()
        self.joints = self.Joints()
        self.base_position = np.zeros(3)
        self.base_position[2] = 0.1944
        self.b_base_velocity = np.zeros(3)

    class IMU:
        def __init__(self):
            self.linear_acceleration = np.zeros(3)
            self.gyroscope = np.zeros(3)
            self.attitude_euler = np.zeros(3)
            self.attitude_quaternion = np.zeros(4)

    class Joints:
        def __init__(self):
            self.positions = np.zeros(12)
            self.velocities = np.zeros(12)


class Controller:
    def __init__(self, params, q_init, t):
        """Function that runs a simulation scenario based on a reference velocity profile, an environment and
        various parameters to define the gait

        Args:
            params (Params object): store parameters
            q_init (array): initial position of actuators
            t (float): time of the simulation
        """
        self.DEMONSTRATION = params.DEMONSTRATION
        self.SIMULATION = params.SIMULATION
        self.solo3D = params.solo3D
        self.k_mpc = int(params.dt_mpc / params.dt_wbc)
        self.type_MPC = params.type_MPC
        self.enable_pyb_GUI = params.enable_pyb_GUI
        self.enable_corba_viewer = params.enable_corba_viewer

        self.q_security = np.array([1.2, 2.1, 3.14] * 4)
        self.solo = init_robot(q_init, params)

        self.joystick = qrw.Joystick()
        self.joystick.initialize(params)

        self.gait = qrw.Gait()
        self.gait.initialize(params)

        self.estimator = qrw.Estimator()
        self.estimator.initialize(params)

        self.wbcWrapper = qrw.WbcWrapper()
        self.wbcWrapper.initialize(params)

        self.h_ref = params.h_ref
        self.q_init = np.hstack((np.zeros(6), q_init.copy()))
        self.q_init[2] = params.h_ref
        self.q_display = np.zeros(19)

        self.mpc_wrapper = MPC_Wrapper.MPC_Wrapper(params, self.q_init)
        self.o_targetFootstep = np.zeros((3, 4))

        self.enable_multiprocessing_mip = params.enable_multiprocessing_mip
        if params.solo3D:
            from .solo3D.SurfacePlannerWrapper import Surface_planner_wrapper

            self.surfacePlanner = Surface_planner_wrapper(params)

            self.statePlanner = qrw.StatePlanner3D()
            self.statePlanner.initialize(params)

            self.footstepPlanner = qrw.FootstepPlannerQP()
            self.footstepPlanner.initialize(
                params, self.gait, self.surfacePlanner.floor_surface
            )

            self.footTrajectoryGenerator = qrw.FootTrajectoryGeneratorBezier()
            self.footTrajectoryGenerator.initialize(
                params, self.gait, self.surfacePlanner.floor_surface
            )
            if self.SIMULATION:
                from .solo3D.pyb_environment_3D import PybEnvironment3D

                self.pybEnvironment3D = PybEnvironment3D(
                    params,
                    self.gait,
                    self.statePlanner,
                    self.footstepPlanner,
                    self.footTrajectoryGenerator,
                )
            self.update_mip = False
        else:
            self.statePlanner = qrw.StatePlanner()
            self.statePlanner.initialize(params)

            self.footstepPlanner = qrw.FootstepPlanner()
            self.footstepPlanner.initialize(params, self.gait)

            self.footTrajectoryGenerator = qrw.FootTrajectoryGenerator()
            self.footTrajectoryGenerator.initialize(params, self.gait)

        # ForceMonitor to display contact forces in PyBullet with red lines
        # import ForceMonitor
        # myForceMonitor = ForceMonitor.ForceMonitor(pyb_sim.robotId, pyb_sim.planeId)

        self.t = t

        self.k = 0

        self.q = np.zeros(18)
        self.q_wbc = np.zeros(18)
        self.dq_wbc = np.zeros(18)
        self.xgoals = np.zeros(12)

        self.filter_q = qrw.Filter()
        self.filter_q.initialize(params)
        self.filter_h_v = qrw.Filter()
        self.filter_h_v.initialize(params)
        self.filter_vref = qrw.Filter()
        self.filter_vref.initialize(params)

        self.error = False
        self.last_q_perfect = np.zeros(6)
        self.last_b_vel = np.zeros(3)
        self.n_nan = 0
        self.result = Result(params)

        device = DummyDevice()
        device.joints.positions = q_init
        self.compute(device)

    def compute(self, device, qc=None):
        """Run one iteration of the main control loop

        Args:
            device (object): Interface with the masterboard or the simulation
        """
        t_start = time.time()

        self.joystick.update_v_ref(self.k, self.gait.is_static())

        q_perfect, b_baseVel_perfect = self.get_perfect_data(qc, device)

        oRh, hRb, oTh = self.run_estimator(device, q_perfect, b_baseVel_perfect)

        t_filter = time.time()
        self.t_filter = t_filter - t_start

        self.gait.update(self.k, self.k_mpc, self.joystick.get_joystick_code())
        gait_matrix = self.gait.matrix

        if self.solo3D:
            self.retrieve_surfaces()

        self.o_targetFootstep = self.footstepPlanner.update_footsteps(
            self.k % self.k_mpc == 0 and self.k != 0,
            int(self.k_mpc - self.k % self.k_mpc),
            self.q_filtered,
            self.h_v_windowed,
            self.v_ref,
        )
        footsteps = self.footstepPlanner.get_footsteps()

        self.statePlanner.compute_reference_states(
            self.q_filtered[:6], self.h_v_filtered, self.vref_filtered
        )
        reference_state = self.statePlanner.get_reference_states()

        if self.solo3D and self.update_mip:
            self.call_planner()

        t_planner = time.time()
        self.t_planner = t_planner - t_filter

        self.solve_MPC(reference_state, footsteps, oRh, oTh)

        t_mpc = time.time()
        self.t_mpc = t_mpc - t_planner

        if self.solo3D:
            self.footTrajectoryGenerator.update(
                self.k,
                self.o_targetFootstep,
                self.surfacePlanner.selected_surfaces,
                self.q_filtered,
            )
        else:
            self.footTrajectoryGenerator.update(self.k, self.o_targetFootstep)

        if not self.error and not self.joystick.get_stop():
            if self.DEMONSTRATION and self.gait.is_static():
                hRb = np.eye(3)

            # Desired position, orientation and velocities of the base
            self.xgoals[:6] = np.zeros(6)
            if self.DEMONSTRATION and self.joystick.get_l1() and self.gait.is_static():
                p_ref = self.joystick.get_p_ref()
                self.xgoals[[3, 4]] = p_ref[[3, 4]]
                self.h_ref = p_ref[2]
                hRb = pin.rpy.rpyToMatrix(0.0, 0.0, self.p_ref[5])
            else:
                self.xgoals[[3, 4]] = reference_state[[3, 4], 1]
                self.h_ref = self.q_init[2]

            # If the four feet are in contact then we do not listen to MPC
            if self.DEMONSTRATION and self.gait.is_static():
                self.mpc_result[12:24, 0] = [0.0, 0.0, 9.81 * 2.5 / 4.0] * 4

            # Wbc uses filtered roll and pitch and reference angular positions of previous loop
            self.q_wbc[3:5] = self.q_filtered[3:5]
            self.q_wbc[6:] = self.wbcWrapper.qdes
            self.dq_wbc[:6] = self.estimator.get_v_estimate()[:6]
            self.dq_wbc[6:] = self.wbcWrapper.vdes

            # Feet command position, velocity and acceleration in base frame
            if self.solo3D:
                oTh_3d = np.zeros((3, 1))
                oTh_3d[:2, 0] = self.q_filtered[:2]
                oRh_3d = pin.rpy.rpyToMatrix(0.0, 0.0, self.q_filtered[5])
                self.feet_a_cmd = (
                    self.footTrajectoryGenerator.get_foot_acceleration_base_frame(
                        oRh_3d.transpose(), np.zeros((3, 1)), np.zeros((3, 1))
                    )
                )
                self.feet_v_cmd = (
                    self.footTrajectoryGenerator.get_foot_velocity_base_frame(
                        oRh_3d.transpose(), np.zeros((3, 1)), np.zeros((3, 1))
                    )
                )
                self.feet_p_cmd = (
                    self.footTrajectoryGenerator.get_foot_position_base_frame(
                        oRh_3d.transpose(),
                        oTh_3d + np.array([[0.0], [0.0], [reference_state[2, 1]]]),
                    )
                )
            else:  # Use ideal base frame
                self.feet_a_cmd = (
                    self.footTrajectoryGenerator.get_foot_acceleration_base_frame(
                        hRb @ oRh.transpose(), np.zeros((3, 1)), np.zeros((3, 1))
                    )
                )
                self.feet_v_cmd = (
                    self.footTrajectoryGenerator.get_foot_velocity_base_frame(
                        hRb @ oRh.transpose(), np.zeros((3, 1)), np.zeros((3, 1))
                    )
                )
                self.feet_p_cmd = (
                    self.footTrajectoryGenerator.get_foot_position_base_frame(
                        hRb @ oRh.transpose(),
                        oTh + np.array([[0.0], [0.0], [self.h_ref]]),
                    )
                )

            self.xgoals[6:] = self.vref_filtered

            # Run InvKin + WBC QP
            self.wbcWrapper.compute(
                self.q_wbc,
                self.dq_wbc,
                (self.mpc_result[12:24, 0:1]).copy(),
                np.array([gait_matrix[0, :]]),
                self.feet_p_cmd,
                self.feet_v_cmd,
                self.feet_a_cmd,
                self.xgoals,
            )

            self.result.q_des[:] = self.wbcWrapper.qdes[:]
            self.result.v_des[:] = self.wbcWrapper.vdes[:]
            self.result.tau_ff[:] = self.wbcWrapper.tau_ff

            self.clamp_result(device)

            # Display robot in Gepetto corba viewer
            if self.enable_corba_viewer and (self.k % 5 == 0):
                self.q_display[:3, 0] = self.q_wbc[:3]
                self.q_display[3:7, 0] = pin.Quaternion(
                    pin.rpy.rpyToMatrix(self.q_wbc[3:6])
                ).coeffs()
                self.q_display[7:, 0] = self.q_wbc[6:]
                self.solo.display(self.q_display)

        self.t_wbc = time.time() - t_mpc

        self.security_check()
        if self.error or self.joystick.get_stop():
            self.set_null_control()

        if not self.solo3D:
            self.pyb_camera(device)
        else:
            if self.SIMULATION:
                self.pybEnvironment3D.update(self.k)

        self.pyb_debug(device, footsteps, gait_matrix, reference_state)

        self.t_loop = time.time() - t_start
        self.k += 1

        return self.error

    def pyb_camera(self, device):
        """
        Update position of PyBullet camera on the robot position to do as if it was
        attached to the robot
        """
        if self.k > 10 and self.enable_pyb_GUI:
            pyb.resetDebugVisualizerCamera(
                cameraDistance=0.6,
                cameraYaw=45,
                cameraPitch=-39.9,
                cameraTargetPosition=[
                    device.dummyHeight[0],
                    device.dummyHeight[1],
                    0.0,
                ],
            )

    def pyb_debug(self, device, footsteps, gait_matrix, xref):

        if self.k > 1 and self.enable_pyb_GUI:
            # Display desired feet positions in WBC as green spheres
            oTh_pyb = device.base_position.reshape((-1, 1))
            oRh_pyb = pin.rpy.rpyToMatrix(0.0, 0.0, device.imu.attitude_euler[2])
            for i in range(4):
                if not self.solo3D:
                    pos = oRh_pyb @ self.feet_p_cmd[:, i : (i + 1)] + oTh_pyb
                    pyb.resetBasePositionAndOrientation(
                        device.pyb_sim.ftps_Ids_deb[i], pos[:, 0].tolist(), [0, 0, 0, 1]
                    )
                else:
                    pos = self.o_targetFootstep[:, i]
                    pyb.resetBasePositionAndOrientation(
                        device.pyb_sim.ftps_Ids_deb[i], pos, [0, 0, 0, 1]
                    )

            # Display desired footstep positions as blue spheres
            for i in range(4):
                j = 0
                cpt = 1
                status = gait_matrix[0, i]
                while (
                    cpt < gait_matrix.shape[0] and j < device.pyb_sim.ftps_Ids.shape[1]
                ):
                    while cpt < gait_matrix.shape[0] and gait_matrix[cpt, i] == status:
                        cpt += 1
                    if cpt < gait_matrix.shape[0]:
                        status = gait_matrix[cpt, i]
                        if status:
                            pos = (
                                oRh_pyb
                                @ footsteps[cpt, (3 * i) : (3 * (i + 1))].reshape(
                                    (-1, 1)
                                )
                                + oTh_pyb
                                - np.array([[0.0], [0.0], [oTh_pyb[2, 0]]])
                            )
                            pyb.resetBasePositionAndOrientation(
                                device.pyb_sim.ftps_Ids[i, j],
                                pos[:, 0].tolist(),
                                [0, 0, 0, 1],
                            )
                        else:
                            pyb.resetBasePositionAndOrientation(
                                device.pyb_sim.ftps_Ids[i, j],
                                [0.0, 0.0, -0.1],
                                [0, 0, 0, 1],
                            )
                        j += 1

                # Hide unused spheres underground
                for k in range(j, device.pyb_sim.ftps_Ids.shape[1]):
                    pyb.resetBasePositionAndOrientation(
                        device.pyb_sim.ftps_Ids[i, k], [0.0, 0.0, -0.1], [0, 0, 0, 1]
                    )

            # Display reference trajectory
            xref_rot = np.zeros((3, xref.shape[1]))
            for i in range(xref.shape[1]):
                xref_rot[:, i : (i + 1)] = (
                    oRh_pyb @ xref[:3, i : (i + 1)]
                    + oTh_pyb
                    + np.array([[0.0], [0.0], [0.05 - self.h_ref]])
                )

            if len(device.pyb_sim.lineId_red) == 0:
                for i in range(xref.shape[1] - 1):
                    device.pyb_sim.lineId_red.append(
                        pyb.addUserDebugLine(
                            xref_rot[:3, i].tolist(),
                            xref_rot[:3, i + 1].tolist(),
                            lineColorRGB=[1.0, 0.0, 0.0],
                            lineWidth=8,
                        )
                    )
            else:
                for i in range(xref.shape[1] - 1):
                    device.pyb_sim.lineId_red[i] = pyb.addUserDebugLine(
                        xref_rot[:3, i].tolist(),
                        xref_rot[:3, i + 1].tolist(),
                        lineColorRGB=[1.0, 0.0, 0.0],
                        lineWidth=8,
                        replaceItemUniqueId=device.pyb_sim.lineId_red[i],
                    )

            # Display predicted trajectory
            mpc_result_rot = np.zeros((3, self.mpc_result.shape[1]))
            for i in range(self.mpc_result.shape[1]):
                mpc_result_rot[:, i : (i + 1)] = (
                    oRh_pyb @ self.mpc_result[:3, i : (i + 1)]
                    + oTh_pyb
                    + np.array([[0.0], [0.0], [0.05 - self.h_ref]])
                )

            if len(device.pyb_sim.lineId_blue) == 0:
                for i in range(self.mpc_result.shape[1] - 1):
                    device.pyb_sim.lineId_blue.append(
                        pyb.addUserDebugLine(
                            mpc_result_rot[:3, i].tolist(),
                            mpc_result_rot[:3, i + 1].tolist(),
                            lineColorRGB=[0.0, 0.0, 1.0],
                            lineWidth=8,
                        )
                    )
            else:
                for i in range(self.mpc_result.shape[1] - 1):
                    device.pyb_sim.lineId_blue[i] = pyb.addUserDebugLine(
                        mpc_result_rot[:3, i].tolist(),
                        mpc_result_rot[:3, i + 1].tolist(),
                        lineColorRGB=[0.0, 0.0, 1.0],
                        lineWidth=8,
                        replaceItemUniqueId=device.pyb_sim.lineId_blue[i],
                    )

    def get_perfect_data(self, qc, device):
        """
        Retrieve perfect data from motion capture or simulator
        Check if nan and send error if more than 5 nans in a row

        @params qc qualisys client for motion capture
        @params device device structure holding simulation data
        @return q_perfect 6D perfect position of the base in world frame
        @return v_baseVel_perfect 3D perfect linear velocity of the base in base frame
        """
        q_perfect = np.zeros(6)
        b_baseVel_perfect = np.zeros(3)
        if self.solo3D and qc == None:
            q_perfect[:3] = device.base_position
            q_perfect[3:] = device.imu.attitude_euler
            b_baseVel_perfect = device.b_base_velocity
        elif self.solo3D and qc != None:
            if self.k <= 1:
                self.initial_pos = [0.0, 0.0, -0.046]
            q_perfect[:3] = qc.getPosition() - self.initial_pos
            q_perfect[3:] = quaternionToRPY(qc.getOrientationQuat())
            b_baseVel_perfect = (
                qc.getOrientationMat9().transpose() @ qc.getVelocity().reshape((3, 1))
            ).ravel()

        if np.isnan(np.sum(q_perfect)):
            print("Error: nan values in perfect position of the robot")
            q_perfect = self.last_q_perfect
            self.n_nan += 1
            if not np.any(self.last_q_perfect) or self.n_nan >= 5:
                self.error = True
        elif np.isnan(np.sum(b_baseVel_perfect)):
            print("Error: nan values in perfect velocity of the robot")
            b_baseVel_perfect = self.last_b_vel
            self.n_nan += 1
            if not np.any(self.last_b_vel) or self.n_nan >= 5:
                self.error = True
        else:
            self.last_q_perfect = q_perfect
            self.last_b_vel = b_baseVel_perfect
            self.n_nan = 0

        return q_perfect, b_baseVel_perfect

    def run_estimator(self, device, q_perfect, b_baseVel_perfect):
        """
        Call the estimator and retrieve the reference and estimated quantities.
        Run a filter on q, h_v and v_ref.

        @params device device structure holding simulation data
        @params q_perfect 6D perfect position of the base in world frame
        @params v_baseVel_perfect 3D perfect linear velocity of the base in base frame
        """

        self.estimator.run(
            self.gait.matrix,
            self.footTrajectoryGenerator.get_foot_position(),
            device.imu.linear_acceleration,
            device.imu.gyroscope,
            device.imu.attitude_euler,
            device.joints.positions,
            device.joints.velocities,
            q_perfect,
            b_baseVel_perfect,
        )

        self.estimator.update_reference_state(self.joystick.get_v_ref())

        oRh = self.estimator.get_oRh()
        hRb = self.estimator.get_hRb()
        oTh = self.estimator.get_oTh().reshape((3, 1))

        self.v_ref = self.estimator.get_base_vel_ref()
        self.h_v = self.estimator.get_h_v()
        self.h_v_windowed = self.estimator.get_h_v_filtered()
        if self.solo3D:
            self.q[:3] = self.estimator.get_q_estimate()[:3]
            self.q[6:] = self.estimator.get_q_estimate()[7:]
            self.q[3:6] = quaternionToRPY(self.estimator.get_q_estimate()[3:7]).ravel()
        else:
            self.q = self.estimator.get_q_reference()
        self.v = self.estimator.get_v_reference()

        # Filter quantities
        self.q_filtered = self.q.copy()
        self.q_filtered[:6] = self.filter_q.filter(self.q[:6], True)
        self.h_v_filtered = self.filter_h_v.filter(self.h_v, False)
        self.vref_filtered = self.filter_vref.filter(self.v_ref, False)

        return oRh, hRb, oTh

    def retrieve_surfaces(self):
        """
        Get last surfaces computed by the planner and send them to the footstep adapter
        """
        self.update_mip = self.k % self.k_mpc == 0 and self.gait.is_new_step()
        if self.update_mip:
            self.statePlanner.update_surface(self.q_filtered[:6], self.vref_filtered)
            if self.surfacePlanner.initialized:
                self.error = self.surfacePlanner.get_latest_results()

        self.footstepPlanner.update_surfaces(
            self.surfacePlanner.potential_surfaces,
            self.surfacePlanner.selected_surfaces,
            self.surfacePlanner.mip_success,
            self.surfacePlanner.mip_iteration,
        )

    def call_planner(self):
        """
        Call the planner and show the result in simulation
        """
        configs = self.statePlanner.get_configurations().transpose()
        self.surfacePlanner.run(
            configs, self.gait.matrix, self.o_targetFootstep, self.vref_filtered[:3]
        )
        self.surfacePlanner.initialized = True
        if not self.enable_multiprocessing_mip and self.SIMULATION:
            self.pybEnvironment3D.update_target_SL1M(
                self.surfacePlanner.all_feet_pos_syn
            )

    def solve_MPC(self, reference_state, footsteps, oRh, oTh):
        if (self.k % self.k_mpc) == 0:
            try:
                if self.type_MPC == 3:
                    l_targetFootstep = oRh.transpose() @ (self.o_targetFootstep - oTh)
                    self.mpc_wrapper.solve(
                        self.k,
                        reference_state,
                        footsteps,
                        self.gait.matrix,
                        l_targetFootstep,
                        oRh,
                        oTh,
                        self.footTrajectoryGenerator.get_foot_position(),
                        self.footTrajectoryGenerator.get_foot_velocity(),
                        self.footTrajectoryGenerator.get_foot_acceleration(),
                        self.footTrajectoryGenerator.get_foot_jerk(),
                        self.footTrajectoryGenerator.get_phase_durations()
                        - self.footTrajectoryGenerator.get_elapsed_durations(),
                    )
                else:
                    self.mpc_wrapper.solve(
                        self.k,
                        reference_state,
                        footsteps,
                        self.gait.matrix,
                        np.zeros((3, 4)),
                    )
            except ValueError:
                print("MPC Problem")
        self.mpc_result, self.mpc_cost = self.mpc_wrapper.get_latest_result()

        if self.k > 100 and self.type_MPC == 3:
            for foot in range(4):
                if self.gait.matrix[0, foot] == 0:
                    id = 0
                    while self.gait.matrix[id, foot] == 0:
                        id += 1
                    self.o_targetFootstep[:2, foot] = self.mpc_result[
                        24 + 2 * foot : 24 + 2 * foot + 2, id + 1
                    ]

    def security_check(self):
        """
        Check if the command is fine and set the command to zero in case of error
        """

        if not (self.error or self.joystick.get_stop()):
            if (np.abs(self.estimator.get_q_estimate()[7:]) > self.q_security).any():
                print("-- POSITION LIMIT ERROR --")
                print(self.estimator.get_q_estimate()[7:])
                print(np.abs(self.estimator.get_q_estimate()[7:]) > self.q_security)
                self.error = True
            elif (np.abs(self.estimator.get_v_security()) > 100.0).any():
                print("-- VELOCITY TOO HIGH ERROR --")
                print(self.estimator.get_v_security())
                print(np.abs(self.estimator.get_v_security()) > 100.0)
                self.error = True
            elif (np.abs(self.wbcWrapper.tau_ff) > 8.0).any():
                print("-- FEEDFORWARD TORQUES TOO HIGH ERROR --")
                print(self.wbcWrapper.tau_ff)
                print(np.abs(self.wbcWrapper.tau_ff) > 8.0)
                self.error = True

    def clamp(self, num, min_value=None, max_value=None):
        clamped = False
        if min_value is not None and num <= min_value:
            num = min_value
            clamped = True
        if max_value is not None and num >= max_value:
            num = max_value
            clamped = True
        return clamped

    def clamp_result(self, device, set_error=False):
        """
        Clamp the result
        """
        hip_max = 120.0 * np.pi / 180.0
        knee_min = 5.0 * np.pi / 180.0
        for i in range(4):
            if self.clamp(self.result.q_des[3 * i + 1], -hip_max, hip_max):
                print("Clamping hip n " + str(i))
                self.error = set_error
            if self.q_init[6 + 3 * i + 2] >= 0.0 and self.clamp(
                self.result.q_des[3 * i + 2], knee_min
            ):
                print("Clamping knee n " + str(i))
                self.error = set_error
            elif self.q_init[6 + 3 * i + 2] <= 0.0 and self.clamp(
                self.result.q_des[3 * i + 2], max_value=-knee_min
            ):
                print("Clamping knee n " + str(i))
                self.error = set_error

        for i in range(12):
            if self.clamp(
                self.result.q_des[i],
                device.joints.positions[i] - 4.0,
                device.joints.positions[i] + 4.0,
            ):
                print("Clamping position difference of motor n " + str(i))
                self.error = set_error

            if self.clamp(
                self.result.v_des[i],
                device.joints.velocities[i] - 100.0,
                device.joints.velocities[i] + 100.0,
            ):
                print("Clamping velocity of motor n " + str(i))
                self.error = set_error

            if self.clamp(self.result.tau_ff[i], -8.0, 8.0):
                print("Clamping torque of motor n " + str(i))
                self.error = set_error

    def set_null_control(self):
        """
        Send default null values to the robot
        """
        self.result.P = np.zeros(12)
        self.result.D = 0.1 * np.ones(12)
        self.result.q_des[:] = np.zeros(12)
        self.result.v_des[:] = np.zeros(12)
        self.result.FF = np.zeros(12)
        self.result.tau_ff[:] = np.zeros(12)
