# coding: utf8

import numpy as np
import pybullet as pyb
import pinocchio as pin
from IPython import embed


class MpcInterface:

    def __init__(self):

        self.oMs = pin.SE3.Identity()

    def update(self):

        ###############################
        # Retrieve data from PyBullet #
        ###############################

        jointStates = pyb.getJointStates(pyb_sim.robotId, pyb_sim.revoluteJointIndices)  # State of all joints
        baseState = pyb.getBasePositionAndOrientation(pyb_sim.robotId)  # Position and orientation of the trunk
        baseVel = pyb.getBaseVelocity(pyb_sim.robotId)  # Velocity of the trunk

        # Joints configuration and velocity vector for free-flyer + 12 actuators
        qmes12 = np.vstack((np.array([baseState[0]]).T, np.array([baseState[1]]).T,
                            np.array([[jointStates[i_joint][0] for i_joint in range(len(jointStates))]]).T))
        vmes12 = np.vstack((np.array([baseVel[0]]).T, np.array([baseVel[1]]).T,
                            np.array([[jointStates[i_joint][1] for i_joint in range(len(jointStates))]]).T))

        return 0


test = MpcInterface()
embed()
