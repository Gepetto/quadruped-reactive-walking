# coding: utf8

import numpy as np
import pinocchio as pin

class MpcInterface:
    """Interface between the simulation and the FootstepPlanner/MPC/TSID. Retrieve information from the simulator
       (positions, orientations, velocities) and compute different useful quantities (transforms, roll-pitch-yaw)
    """

    def __init__(self):

        # Initialisation of matrices
        self.oRb = np.eye(3)  # rotation matrix from the world frame to the base frame
        self.vmes12_base = np.zeros((18, 1))  # pinocchio needs the linear and angular vel in base frame
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
        self.o_feet = np.zeros((3, 4))  # position of feet in world frame
        self.lv_feet = np.zeros((3, 4))  # velocity of feet in local frame
        self.ov_feet = np.zeros((3, 4))  # velocity of feet in world frame
        self.la_feet = np.zeros((3, 4))  # acceleration of feet in local frame
        self.oa_feet = np.zeros((3, 4))  # acceleration of feet in world frame

        # Indexes of feet frames in this order: [FL, FR, HL, HR]
        self.indexes = [10, 18, 26, 34]

        # Average height of feet in local frame
        self.mean_feet_z = 0.0

        # Projection of shoulders on the ground in local and world frame
        self.l_shoulders = np.array([[0.19, 0.19, -0.19, -0.19], [0.15005, -0.15005,
                                                                  0.15005, -0.15005], [0.0, 0.0, 0.0, 0.0]])
        self.o_shoulders = np.zeros((3, 4))

    def update(self, solo, qmes12, vmes12):
        """Update the quantities of the MpcInterface based on the last measurements from the simulation

        Args:
            solo (object): Pinocchio wrapper for the quadruped
            qmes12 (19x1 array): the position/orientation of the trunk and angular position of actuators
            vmes12 (18x1 array): the linear/angular velocity of the trunk and angular velocity of actuators

        """

        # Rotation matrix from the world frame to the base frame
        self.oRb = pin.Quaternion(qmes12[3:7]).matrix()

        # Linear and angular velocity in base frame
        self.vmes12_base = vmes12.copy()
        self.vmes12_base[0:3, 0:1] = self.oRb.transpose() @ self.vmes12_base[0:3, 0:1]
        self.vmes12_base[3:6, 0:1] = self.oRb.transpose() @ self.vmes12_base[3:6, 0:1]

        # Update Kinematics (required or automatically done by other functions?)
        pin.forwardKinematics(solo.model, solo.data, qmes12, self.vmes12_base)
        pin.framesForwardKinematics(solo.model, solo.data, qmes12)

        # Get center of mass from Pinocchio
        pin.centerOfMass(solo.model, solo.data, qmes12, self.vmes12_base)

        # Update position/orientation of frames
        pin.updateFramePlacements(solo.model, solo.data)

        # Update minimum height of feet
        # TODO: Rename mean_feet_z into min_feet_z
        self.mean_feet_z = solo.data.oMf[self.indexes[0]].translation[2, 0]
        """for i in self.indexes:
            self.mean_feet_z += solo.data.oMf[i].translation[2, 0]
        self.mean_feet_z *= 0.25"""
        for i in self.indexes[1:]:
            self.mean_feet_z = np.min((self.mean_feet_z, solo.data.oMf[i].translation[2, 0]))

        # Store position, linear velocity and angular velocity in global frame
        self.oC = solo.data.com[0]
        self.oV = solo.data.vcom[0]
        self.oW = vmes12[3:6]

        # Get SE3 object from world frame to base frame
        self.oMb = pin.SE3(pin.Quaternion(qmes12[3:7]), self.oC)
        self.RPY = pin.rpy.matrixToRpy(self.oMb.rotation)

        # Get SE3 object from world frame to local frame
        self.oMl = pin.SE3(pin.utils.rotate('z', self.RPY[2, 0]),
                           np.array([qmes12[0, 0], qmes12[1, 0], self.mean_feet_z]))

        # Get position, linear velocity and angular velocity in local frame
        self.lC = self.oMl.inverse() * self.oC
        self.lV = self.oMl.rotation.transpose() @ self.oV
        self.lW = self.oMl.rotation.transpose() @ self.oW

        # Pos, vel and acc of feet
        for i, j in enumerate(self.indexes):
            # Position of feet in local frame
            self.o_feet[:, i:(i+1)] = solo.data.oMf[j].translation
            self.l_feet[:, i:(i+1)] = self.oMl.inverse() * solo.data.oMf[j].translation

            # getFrameVelocity output is in the frame of the foot so a transform is required
            self.ov_feet[:, i:(i+1)] = solo.data.oMf[j].rotation @ pin.getFrameVelocity(solo.model,
                                                                                        solo.data, j).vector[0:3, 0:1]
            self.lv_feet[:, i:(i+1)] = self.oMl.rotation.transpose() @ self.ov_feet[:, i:(i+1)]

            # getFrameAcceleration output is in the frame of the foot so a transform is required
            self.oa_feet[:, i:(i+1)] = solo.data.oMf[j].rotation @ pin.getFrameAcceleration(solo.model,
                                                                                            solo.data, j).vector[0:3,
                                                                                                                 0:1]
            self.la_feet[:, i:(i+1)] = self.oMl.rotation.transpose() @ self.oa_feet[:, i:(i+1)]

        # Orientation of the base in local frame
        # Base and local frames have the same yaw orientation in world frame
        self.abg[0:2] = self.RPY[0:2]

        # Position of shoulders in world frame
        for i in range(4):
            self.o_shoulders[:, i:(i+1)] = self.oMl * self.l_shoulders[:, i]

        return 0
