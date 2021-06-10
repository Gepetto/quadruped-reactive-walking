# coding: utf8

import numpy as np
import pinocchio as pin
from solo12InvKin import Solo12InvKin
from time import perf_counter as clock
from time import time
import libquadruped_reactive_walking as lrw
from example_robot_data.robots_loader import Solo12Loader

class wbc_controller():
    """Whole body controller which contains an Inverse Kinematics step and a BoxQP step

    Args:
        dt (float): time step of the whole body control
    """

    def __init__(self, dt, N_SIMULATION):

        Solo12Loader.free_flyer = True
        self.robot = Solo12Loader().robot

        self.dt = dt  # Time step

        self.invKin = Solo12InvKin(dt)  # Inverse Kinematics object
        self.box_qp = lrw.QPWBC()  # Box Quadratic Programming solver

        self.M = np.zeros((18, 18))
        self.Jc = np.zeros((12, 18))

        self.error = False  # Set to True when an error happens in the controller

        self.k_since_contact = np.zeros((1, 4))

        # Logging
        self.k_log = 0
        self.log_feet_pos = np.zeros((3, 4, N_SIMULATION))
        self.log_feet_err = np.zeros((3, 4, N_SIMULATION))
        self.log_feet_vel = np.zeros((3, 4, N_SIMULATION))
        self.log_feet_pos_target = np.zeros((3, 4, N_SIMULATION))
        self.log_feet_vel_target = np.zeros((3, 4, N_SIMULATION))
        self.log_feet_acc_target = np.zeros((3, 4, N_SIMULATION))

        # Arrays to store results (for solo12)
        self.qdes = np.zeros((19, ))
        self.vdes = np.zeros((18, 1))
        self.tau_ff = np.zeros(12)

        # Indexes of feet frames in this order: [FL, FR, HL, HR]
        self.indexes = [10, 18, 26, 34]

    def compute(self, q, dq, f_cmd, contacts, pgoals, vgoals, agoals):
        """ Call Inverse Kinematics to get an acceleration command then
        solve a QP problem to get the feedforward torques

        Args:
            q (19x1): Current state of the base
            dq (18x1): Current velocity of the base (in base frame)
            f_cmd (1x12): Contact forces references from the mpc
            contacts (1x4): Contact status of feet
            planner (object): Object that contains the pos, vel and acc references for feet
        """

        # Update nb of iterations since contact
        self.k_since_contact += contacts  # Increment feet in stance phase
        self.k_since_contact *= contacts  # Reset feet in swing phase

        self.tic = time()

        # Compute Inverse Kinematics
        ddq_cmd = np.array([self.invKin.refreshAndCompute(q[7:, 0:1], dq[6:, 0:1], contacts, pgoals, vgoals, agoals)]).T

        for i in range(4):
            self.log_feet_pos[:, i, self.k_log] = self.invKin.robot.data.oMf[self.indexes[i]].translation
            self.log_feet_err[:, i, self.k_log] = pgoals[:, i] - self.invKin.robot.data.oMf[self.indexes[i]].translation # self.invKin.pfeet_err[i]
            self.log_feet_vel[:, i, self.k_log] = pin.getFrameVelocity(self.invKin.robot.model, self.invKin.robot.data,
                                                                       self.indexes[i], pin.LOCAL_WORLD_ALIGNED).linear
        self.feet_pos = self.log_feet_pos[:, :, self.k_log]
        self.feet_err = self.log_feet_err[:, :, self.k_log]
        self.feet_vel = self.log_feet_vel[:, :, self.k_log]

        self.log_feet_pos_target[:, :, self.k_log] = pgoals[:, :]
        self.log_feet_vel_target[:, :, self.k_log] = vgoals[:, :]
        self.log_feet_acc_target[:, :, self.k_log] = agoals[:, :]

        self.tac = time()

        # Compute the joint space inertia matrix M by using the Composite Rigid Body Algorithm
        q_tmp = np.zeros((19, 1))
        q_tmp[6, 0] = 1.0
        self.M = pin.crba(self.robot.model, self.robot.data, q_tmp)

        self.M[:6, :6] = self.M[:6, :6] * (np.eye(6) == 1)  # (self.M[:6, :6] > 1e-3)

        # Compute Jacobian of contact points
        pin.computeJointJacobians(self.robot.model, self.robot.data, q)
        self.Jc = np.zeros((12, 18))
        for i_ee in range(4):
            if contacts[i_ee]:
                idx = int(self.invKin.foot_ids[i_ee])
                self.Jc[(3*i_ee):(3*(i_ee+1)), :] = pin.getFrameJacobian(self.robot.model, self.robot.data, idx, pin.LOCAL_WORLD_ALIGNED)[:3]

        # Compute joint torques according to the current state of the system and the desired joint accelerations
        RNEA = pin.rnea(self.robot.model, self.robot.data, q, dq, ddq_cmd)[:6]

        # Solve the QP problem with C++ bindings
        self.box_qp.run(self.M, self.Jc, f_cmd.reshape((-1, 1)), RNEA.reshape((-1, 1)), self.k_since_contact)

        # Add deltas found by the QP problem to reference quantities
        deltaddq = self.box_qp.get_ddq_res()
        self.f_with_delta = self.box_qp.get_f_res().reshape((-1, 1))
        ddq_with_delta = ddq_cmd.copy()
        ddq_with_delta[:6, 0] += deltaddq

        # Compute joint torques from contact forces and desired accelerations
        RNEA_delta = pin.rnea(self.robot.model, self.robot.data, q, dq, ddq_with_delta)[6:]
        self.tau_ff[:] = RNEA_delta - ((self.Jc[:, 6:].transpose()) @ self.f_with_delta).ravel()

        # Retrieve desired positions and velocities
        self.vdes[:, 0] = self.invKin.dq_cmd
        self.qdes[:] = self.invKin.q_cmd

        self.toc = time()

        """self.tic = 0.0
        self.tac = 0.0
        self.toc = 0.0"""

        self.k_log += 1

        return 0
