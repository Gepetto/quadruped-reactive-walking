# coding: utf8

import numpy as np
import pinocchio as pin
from solo12InvKin import Solo12InvKin
from time import perf_counter as clock
from time import time
import libquadruped_reactive_walking as lrw


class wbc_controller():
    """Whole body controller which contains an Inverse Kinematics step and a BoxQP step

    Args:
        dt (float): time step of the whole body control
    """

    def __init__(self, dt, N_SIMULATION):

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

    def compute(self, q, dq, x_cmd, f_cmd, contacts, planner):
        """ Call Inverse Kinematics to get an acceleration command then
        solve a QP problem to get the feedforward torques

        Args:
            q (19x1): Current state of the base
            dq (18x1): Current velocity of the base (in base frame)
            x_cmd (1x12): Position and velocity references from the mpc
            f_cmd (1x12): Contact forces references from the mpc
            contacts (1x4): Contact status of feet
            planner (object): Object that contains the pos, vel and acc references for feet
        """

        # Update nb of iterations since contact
        self.k_since_contact += contacts  # Increment feet in stance phase
        self.k_since_contact *= contacts  # Reset feet in swing phase

        self.tic = time()

        # Compute Inverse Kinematics
        ddq_cmd = np.array([self.invKin.refreshAndCompute(q.copy(), dq.copy(), x_cmd, contacts, planner)]).T

        for i in range(4):
            self.log_feet_pos[:, i, self.k_log] = self.invKin.rdata.oMf[self.indexes[i]].translation
            self.log_feet_err[:, i, self.k_log] = self.invKin.feet_position_ref[i] - self.invKin.rdata.oMf[self.indexes[i]].translation # self.invKin.pfeet_err[i]
            self.log_feet_vel[:, i, self.k_log] = pin.getFrameVelocity(self.invKin.rmodel, self.invKin.rdata,
                                                                       self.indexes[i], pin.LOCAL_WORLD_ALIGNED).linear
        self.feet_pos = self.log_feet_pos[:, :, self.k_log]
        self.feet_err = self.log_feet_err[:, :, self.k_log]
        self.feet_vel = self.log_feet_vel[:, :, self.k_log]

        self.log_feet_pos_target[:, :, self.k_log] = planner.goals[:, :]
        self.log_feet_vel_target[:, :, self.k_log] = planner.vgoals[:, :]
        self.log_feet_acc_target[:, :, self.k_log] = planner.agoals[:, :]

        self.tac = time()

        # Compute the joint space inertia matrix M by using the Composite Rigid Body Algorithm
        self.M = pin.crba(self.invKin.rmodel, self.invKin.rdata, q)

        # Compute Jacobian of contact points
        #print("##")
        self.Jc = np.zeros((12, 18))
        for i in range(4):
            if contacts[i]:
                # Feet Jacobian were already retrieved in InvKin so no need to call getFrameJacobian
                self.Jc[(3*i):(3*(i+1)), :] = (self.invKin.cpp_Jf[(3*i):(3*(i+1)), :]).copy()

        # Compute joint torques according to the current state of the system and the desired joint accelerations
        RNEA = pin.rnea(self.invKin.rmodel, self.invKin.rdata, q, dq, ddq_cmd)[:6]

        # Solve the QP problem with C++ bindings
        self.box_qp.run(self.M, self.Jc, f_cmd.reshape((-1, 1)), RNEA.reshape((-1, 1)), self.k_since_contact)

        # Add deltas found by the QP problem to reference quantities
        deltaddq = self.box_qp.get_ddq_res()
        self.f_with_delta = self.box_qp.get_f_res().reshape((-1, 1))
        ddq_with_delta = ddq_cmd.copy()
        ddq_with_delta[:6, 0] += deltaddq

        # Compute joint torques from contact forces and desired accelerations
        RNEA_delta = pin.rnea(self.invKin.rmodel, self.invKin.rdata, q, dq, ddq_with_delta)[6:]
        self.tau_ff[:] = RNEA_delta - ((self.Jc[:, 6:].transpose()) @ self.f_with_delta).ravel()

        # Retrieve desired positions and velocities
        self.vdes[:, 0] = self.invKin.dq_cmd
        self.qdes[:] = self.invKin.q_cmd

        self.toc = time()

        """self.tic = 0.0
        self.tac = 0.0
        self.toc = 0.0"""

        return 0
