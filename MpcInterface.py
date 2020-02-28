# coding: utf8

import numpy as np
import pybullet as pyb
import pinocchio as pin
from IPython import embed


class MpcInterface:

    def __init__(self):

        self.qmes12 = np.zeros((19, 1))
        self.vmes12 = np.zeros((18, 1))

        self.oMb = pin.SE3.Identity()
        self.oMl = pin.SE3.Identity()

    def update(self, pyb_sim, solo):

        ###############################
        # Retrieve data from PyBullet #
        ###############################

        jointStates = pyb.getJointStates(pyb_sim.robotId, pyb_sim.revoluteJointIndices)  # State of all joints
        baseState = pyb.getBasePositionAndOrientation(pyb_sim.robotId)  # Position and orientation of the trunk
        baseVel = pyb.getBaseVelocity(pyb_sim.robotId)  # Velocity of the trunk

        # Joints configuration and velocity vector for free-flyer + 12 actuators
        self.qmes12 = np.vstack((np.array([baseState[0]]).T, np.array([baseState[1]]).T,
                                 np.array([[jointStates[i_joint][0] for i_joint in range(len(jointStates))]]).T))
        self.vmes12 = np.vstack((np.array([baseVel[0]]).T, np.array([baseVel[1]]).T,
                                 np.array([[jointStates[i_joint][1] for i_joint in range(len(jointStates))]]).T))

        ################
        # Process data #
        ################

        # Get SE3 object from world frame to base frame
        self.oMb = pin.SE3(self.qmes12[3:7], self.qmes12[0:3])
        self.RPY = pin.matrixToRpy(self.oMb.rotation)

        # Get SE3 object from world frame to local frame
        self.oMl = pin.SE3(pin.utils.rotate('z', self.RPY[2]), np.array([self.qmes12[0], self.qmes12[1], 0.0]))

        # Get center of mass from Pinocchio
        pin.centerOfMass(solo.model, solo.data, self.qmes12, self.vmes12)

        # Store result
        oC = solo.data.com[0]
        oV = solo.data.vcom[0]

        # Get position and velocity in local frame
        lC = oMl.inverse() * oC
        lV = oMl.rotation.transpose() @ oV

        return 0


test = MpcInterface()
embed()
