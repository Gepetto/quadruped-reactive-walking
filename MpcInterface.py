# coding: utf8

import numpy as np
import pybullet as pyb
import pinocchio as pin
from IPython import embed
import utils


class MpcInterface:

    def __init__(self):

        # PyBullet data
        self.qmes12 = np.zeros((19, 1))  # position
        self.vmes12 = np.zeros((18, 1))  # velocity

        # Initialisation of matrices
        self.oMb = pin.SE3.Identity()  # transform from world to base frame ("1")
        self.oMl = pin.SE3.Identity()  #  transform from world to local frame ("L")
        self.RPY = np.zeros((3, 1))  # roll, pitch, yaw of the base in world frame
        self.oC = np.zeros((3, ))  #  position of the CoM in world frame
        self.oV = np.zeros((3, ))  #  linear velocity of the CoM in world frame
        self.oW = np.zeros((3, 1))  # angular velocity of the CoM in world frame
        self.lC = np.zeros((3, ))  #  position of the CoM in local frame
        self.lV = np.zeros((3, ))  #  linear velocity of the CoM in local frame
        self.lW = np.zeros((3, 1))  #  angular velocity of the CoM in local frame
        self.lRb = np.eye(3)  # rotation matrix from the local frame to the base frame
        self.abg = np.zeros((3, 1))  # roll, pitch, yaw of the base in local frame
        self.l_feet = np.zeros((3, 4))  # position of feet in local frame

        # Indexes of feet frames
        self.indexes = [10, 18, 26, 34]

        # Average height of feet in local frame
        self.mean_feet_z = 0.0

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
        self.oMb = pin.SE3(pin.Quaternion(self.qmes12[3:7]), self.qmes12[0:3])
        self.RPY = pin.rpy.matrixToRpy(self.oMb.rotation)

        # Get SE3 object from world frame to local frame
        self.oMl = pin.SE3(pin.utils.rotate('z', self.RPY[2, 0]),
                           np.array([self.qmes12[0, 0], self.qmes12[1, 0], self.mean_feet_z]))

        # Get center of mass from Pinocchio
        pin.centerOfMass(solo.model, solo.data, self.qmes12, self.vmes12)

        # Store position, linear velocity and angular velocity in global frame
        self.oC = solo.data.com[0]
        self.oV = solo.data.vcom[0]
        self.oW = self.vmes12[3:6]

        # Get position, linear velocity and angular velocity in local frame
        self.lC = self.oMl.inverse() * self.oC
        self.lV = self.oMl.rotation.transpose() @ self.oV
        self.lW = self.oMl.rotation.transpose() @ self.oW

        # Check that the base frame and the local frame have the same yaw orientation
        self.lRb = self.oMl.rotation.transpose() @ self.oMb.rotation
        self.abg = pin.rpy.matrixToRpy(self.lRb)

        # Compute position of frames
        pin.forwardKinematics(solo.model, solo.data, self.qmes12)
        pin.updateFramePlacements(solo.model, solo.data)

        # Position of feet in local frame
        for i, j in enumerate(self.indexes):
            self.l_feet[:, i] = self.oMl.inverse() * solo.data.oMf[j].translation

        # Update average height of feet
        self.mean_feet_z = np.mean(self.l_feet[2, :])  #  average in local frame or in world frame?

        return 0


test = MpcInterface()

solo = utils.init_viewer()
pyb_sim = utils.pybullet_simulator(0.005)

test.update(pyb_sim, solo)  #  To initialize the average height
test.update(pyb_sim, solo)

print("Feet position: ")
print(test.l_feet)

embed()
