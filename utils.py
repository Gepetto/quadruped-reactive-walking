import math
import numpy as np

import robots_loader  # Gepetto viewer

import Joystick
import ContactSequencer
import FootstepPlanner
import FootTrajectoryGenerator
import MPC
import Logger
import Interface

import pybullet as pyb  # Pybullet server
import pybullet_data
import pinocchio as pin

##########################
# ROTATION MATRIX TO RPY #
##########################

# Taken from https://www.learnopencv.com/rotation-matrix-to-euler-angles/

# Checks if a matrix is a valid rotation matrix.


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):

    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

###########################################################
#  Roll Pitch Yaw (3 x 1) to Quaternion function (4 x 1)  #
###########################################################


def getQuaternion(rpy):
    c = np.cos(rpy*0.5)
    s = np.sin(rpy*0.5)
    cy = c[2, 0]
    sy = s[2, 0]
    cp = c[1, 0]
    sp = s[1, 0]
    cr = c[0, 0]
    sr = s[0, 0]

    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr

    return np.array([[x, y, z, w]]).transpose()

#################################
# Quarternion to Roll Pitch Yaw #
#################################


def quaternionToRPY(quat):
    qx = quat[0]
    qy = quat[1]
    qz = quat[2]
    qw = quat[3]

    rotateXa0 = 2.0*(qy*qz + qw*qx)
    rotateXa1 = qw*qw - qx*qx - qy*qy + qz*qz
    rotateX = 0.0

    if (rotateXa0 != 0.0) and (rotateXa1 != 0.0):
        rotateX = np.arctan2(rotateXa0, rotateXa1)

    rotateYa0 = -2.0*(qx*qz - qw*qy)
    rotateY = 0.0
    if (rotateYa0 >= 1.0):
        rotateY = np.pi/2.0
    elif (rotateYa0 <= -1.0):
        rotateY = -np.pi/2.0
    else:
        rotateY = np.arcsin(rotateYa0)

    rotateZa0 = 2.0*(qx*qy + qw*qz)
    rotateZa1 = qw*qw + qx*qx - qy*qy - qz*qz
    rotateZ = 0.0
    if (rotateZa0 != 0.0) and (rotateZa1 != 0.0):
        rotateZ = np.arctan2(rotateZa0, rotateZa1)

    return np.array([[rotateX], [rotateY], [rotateZ]])


##################
# Initialisation #
##################


def init_viewer(enable_viewer):
    # loadSolo(False) to load solo12
    # loadSolo(True) to load solo8
    solo = robots_loader.loadSolo(False)

    if enable_viewer:
        solo.initDisplay(loadModel=True)
        if ('viewer' in solo.viz.__dict__):
            solo.viewer.gui.addFloor('world/floor')
            solo.viewer.gui.setRefreshIsSynchronous(False)
        """offset = np.zeros((19, 1))
        offset[5, 0] = 0.7071067811865475
        offset[6, 0] = 0.7071067811865475 - 1.0
        temp = solo.q0 + offset"""
        solo.display(solo.q0)

        pin.centerOfMass(solo.model, solo.data, solo.q0, np.zeros((18, 1)))
        pin.updateFramePlacements(solo.model, solo.data)
        pin.crba(solo.model, solo.data, solo.q0)

    return solo


def init_objects(dt_tsid, dt_mpc, k_max_loop, k_mpc, n_periods):

    # Create Joystick object
    joystick = Joystick.Joystick(k_mpc)

    # Create footstep planner object
    fstep_planner = FootstepPlanner.FootstepPlanner(dt_mpc, n_periods)

    # Create logger object
    logger = Logger.Logger(k_max_loop, dt_tsid, dt_mpc, k_mpc, n_periods)

    # Create Interface object
    interface = Interface.Interface()

    return joystick, fstep_planner, logger, interface


def display_all(solo, k, sequencer, fstep_planner, ftraj_gen, mpc):
    # Display non-locked target footholds with green spheres (gepetto gui)
    fstep_planner.update_viewer(solo.viewer, (k == 0))

    # Display locked target footholds with red spheres (gepetto gui)
    # Display desired 3D position of feet with magenta spheres (gepetto gui)
    ftraj_gen.update_viewer(solo.viewer, (k == 0))

    # Display reference trajectory, predicted trajectory, desired contact forces, current velocity
    # mpc.update_viewer(solo.viewer, (k == 0), sequencer)
    # mpc.plot_graphs(sequencer)

    qu_pinocchio = np.array(solo.q0).flatten()
    qu_pinocchio[0:3] = mpc.q_w[0:3, 0]
    qu_pinocchio[3:7] = getQuaternion(np.matrix([mpc.q_w[3:6, 0]]).T).flatten()
    # Refresh the gepetto viewer display
    solo.display(qu_pinocchio)


def getSkew(a):
    """Returns the skew matrix of a 3 by 1 column vector

    Keyword arguments:
    a -- the column vector
    """
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=a.dtype)

########################################################################
#                              PyBullet                                #
########################################################################


class pybullet_simulator:

    def __init__(self, dt=0.001):

        # Start the client for PyBullet
        physicsClient = pyb.connect(pyb.DIRECT)
        # p.GUI for graphical version
        # p.DIRECT for non-graphical version

        # Load horizontal plane
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.planeId = pyb.loadURDF("plane.urdf")
        #self.stairsId = pyb.loadURDF("../../../../../Documents/Git-Repositories/mpc-tsid/bauzil_stairs.urdf")#,
                                #basePosition=[-1.25, 3.5, -0.1],
                                #baseOrientation=pyb.getQuaternionFromEuler([0.0, 0.0, 3.1415]))
        #pyb.changeDynamics(self.stairsId, -1, lateralFriction=1.0)

        mesh_scale = [1.0, 0.1, 0.02]
        visualShapeId = pyb.createVisualShape(shapeType=pyb.GEOM_MESH,
                                              fileName="cube.obj",
                                              halfExtents=[0.5, 0.5, 0.1],
                                              rgbaColor=[1.0, 0.0, 0.0, 1.0],
                                              specularColor=[0.4, .4, 0],
                                              visualFramePosition=[0.0, 0.0, 0.0],
                                              meshScale=mesh_scale)

        collisionShapeId = pyb.createCollisionShape(shapeType=pyb.GEOM_MESH,
                                                    fileName="cube.obj",
                                                    collisionFramePosition=[0.0, 0.0, 0.0],
                                                    meshScale=mesh_scale)
        for i in range(4):
            tmpId = pyb.createMultiBody(baseMass=0.0,
                                baseInertialFramePosition=[0, 0, 0],
                                baseCollisionShapeIndex=collisionShapeId,
                                baseVisualShapeIndex=visualShapeId,
                                basePosition=[0.0, 0.5+0.2*i, 0.01],
                                useMaximalCoordinates=True)
            pyb.changeDynamics(tmpId, -1, lateralFriction=1.0)

        tmpId = pyb.createMultiBody(baseMass=0.0,
                            baseInertialFramePosition=[0, 0, 0],
                            baseCollisionShapeIndex=collisionShapeId,
                            baseVisualShapeIndex=visualShapeId,
                            basePosition=[0.5, 0.5+0.2*4, 0.01],
                            useMaximalCoordinates=True)
        pyb.changeDynamics(tmpId, -1, lateralFriction=1.0)

        tmpId = pyb.createMultiBody(baseMass=0.0,
                            baseInertialFramePosition=[0, 0, 0],
                            baseCollisionShapeIndex=collisionShapeId,
                            baseVisualShapeIndex=visualShapeId,
                            basePosition=[0.5, 0.5+0.2*5, 0.01],
                            useMaximalCoordinates=True)
        pyb.changeDynamics(tmpId, -1, lateralFriction=1.0)

        mesh_scale = [0.2, 0.1, 0.01]
        visualShapeId = pyb.createVisualShape(shapeType=pyb.GEOM_MESH,
                                              fileName="cube.obj",
                                              halfExtents=[0.5, 0.5, 0.1],
                                              rgbaColor=[0.0, 1.0, 0.0, 1.0],
                                              specularColor=[0.4, .4, 0],
                                              visualFramePosition=[0.0, 0.0, 0.0],
                                              meshScale=mesh_scale)

        collisionShapeId = pyb.createCollisionShape(shapeType=pyb.GEOM_MESH,
                                                    fileName="cube.obj",
                                                    collisionFramePosition=[0.0, 0.0, 0.0],
                                                    meshScale=mesh_scale)

        for i in range(3):
            tmpId = pyb.createMultiBody(baseMass=0.0,
                                baseInertialFramePosition=[0, 0, 0],
                                baseCollisionShapeIndex=collisionShapeId,
                                baseVisualShapeIndex=visualShapeId,
                                basePosition=[0.15 * (-1)**i, 0.9+0.2*i, 0.025],
                                useMaximalCoordinates=True)
            pyb.changeDynamics(tmpId, -1, lateralFriction=1.0)

        mesh_scale = [0.05, 0.05, 0.05]
        visualShapeId = pyb.createVisualShape(shapeType=pyb.GEOM_MESH,
                                              fileName="sphere_smooth.obj",
                                              halfExtents=[0.5, 0.5, 0.1],
                                              rgbaColor=[1.0, 0.0, 0.0, 1.0],
                                              specularColor=[0.4, .4, 0],
                                              visualFramePosition=[0.0, 0.0, 0.0],
                                              meshScale=mesh_scale)

        collisionShapeId = pyb.createCollisionShape(shapeType=pyb.GEOM_MESH,
                                                    fileName="sphere_smooth.obj",
                                                    collisionFramePosition=[0.0, 0.0, 0.0],
                                                    meshScale=mesh_scale)

        self.sphereId1 = pyb.createMultiBody(baseMass=0.3,
                                             baseInertialFramePosition=[0, 0, 0],
                                             baseCollisionShapeIndex=collisionShapeId,
                                             baseVisualShapeIndex=visualShapeId,
                                             basePosition=[-0.6, 0.9, 0.05],
                                             useMaximalCoordinates=True)

        self.sphereId2 = pyb.createMultiBody(baseMass=0.3,
                                             baseInertialFramePosition=[0, 0, 0],
                                             baseCollisionShapeIndex=collisionShapeId,
                                             baseVisualShapeIndex=visualShapeId,
                                             basePosition=[0.6, 1.1, 0.05],
                                             useMaximalCoordinates=True)

        mesh_scale = [0.015, 0.015, 0.015]
        visualShapeId = pyb.createVisualShape(shapeType=pyb.GEOM_MESH,
                                              fileName="sphere_smooth.obj",
                                              halfExtents=[0.5, 0.5, 0.1],
                                              rgbaColor=[0.0, 0.0, 1.0, 1.0],
                                              specularColor=[0.4, .4, 0],
                                              visualFramePosition=[0.0, 0.0, 0.0],
                                              meshScale=mesh_scale)

        # Flag to launch the two spheres in the environment toward the robot
        self.flag_sphere1 = True
        self.flag_sphere2 = True

        self.ftps_Ids = np.zeros((4, 5), dtype=np.int)
        for i in range(4):
            for j in range(5):
                self.ftps_Ids[i, j] = pyb.createMultiBody(baseMass=0.0,
                                                          baseInertialFramePosition=[0, 0, 0],
                                                          baseVisualShapeIndex=visualShapeId,
                                                          basePosition=[0.0, 0.0, -0.1],
                                                          useMaximalCoordinates=True)

        visualShapeId = pyb.createVisualShape(shapeType=pyb.GEOM_MESH,
                                              fileName="sphere_smooth.obj",
                                              halfExtents=[0.5, 0.5, 0.1],
                                              rgbaColor=[0.0, 1.0, 0.0, 1.0],
                                              specularColor=[0.4, .4, 0],
                                              visualFramePosition=[0.0, 0.0, 0.0],
                                              meshScale=mesh_scale)
        self.ftps_Ids_deb = [0] * 4
        for i in range(4):
            self.ftps_Ids_deb[i] = pyb.createMultiBody(baseMass=0.0,
                                                          baseInertialFramePosition=[0, 0, 0],
                                                          baseVisualShapeIndex=visualShapeId,
                                                          basePosition=[0.0, 0.0, -0.1],
                                                          useMaximalCoordinates=True)

        """cubeStartPos = [0.0, 0.45, 0.0]
        cubeStartOrientation = pyb.getQuaternionFromEuler([0, 0, 0])
        self.cubeId = pyb.loadURDF("cube_small.urdf",
                                   cubeStartPos, cubeStartOrientation)
        pyb.changeDynamics(self.cubeId, -1, mass=0.5)
        print("Mass of cube:", pyb.getDynamicsInfo(self.cubeId, -1)[0])"""

        # Set the gravity
        pyb.setGravity(0, 0, -9.81)

        # Load Quadruped robot
        robotStartPos = [0, 0, 0.235+0.0045]
        robotStartOrientation = pyb.getQuaternionFromEuler([0.0, 0.0, 0.0]) # -np.pi/2
        pyb.setAdditionalSearchPath("/opt/openrobots/share/example-robot-data/robots/solo_description/robots")
        self.robotId = pyb.loadURDF("solo12.urdf", robotStartPos, robotStartOrientation)

        # Disable default motor control for revolute joints
        self.revoluteJointIndices = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
        pyb.setJointMotorControlArray(self.robotId, jointIndices=self.revoluteJointIndices, controlMode=pyb.VELOCITY_CONTROL,
                                      targetVelocities=[0.0 for m in self.revoluteJointIndices],
                                      forces=[0.0 for m in self.revoluteJointIndices])

        # Initialize the robot in a specific configuration
        straight_standing = np.array([[0, 0.8, -1.6, 0, 0.8, -1.6, 0, -0.8, 1.6, 0, -0.8, 1.6]]).transpose()
        pyb.resetJointStatesMultiDof(self.robotId, self.revoluteJointIndices, straight_standing)  # q0[7:])

        # Enable torque control for revolute joints
        jointTorques = [0.0 for m in self.revoluteJointIndices]
        pyb.setJointMotorControlArray(self.robotId, self.revoluteJointIndices,
                                      controlMode=pyb.TORQUE_CONTROL, forces=jointTorques)

        # Fix the base in the world frame
        # p.createConstraint(robotId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0.34])

        # Set time step for the simulation
        pyb.setTimeStep(dt)

        # Change camera position
        pyb.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=-50, cameraPitch=-35,
                                       cameraTargetPosition=[0.0, 0.6, 0.0])

    def check_pyb_env(self, qmes12):
        """Check the state of the robot to trigger events and update camera

        Args:
            qmes12 (19x1 array): the position/orientation of the trunk and angular position of actuators

        """

        # Check if the robot is in front of the first sphere to trigger it
        if self.flag_sphere1 and (qmes12[1, 0] >= 0.9):
            pyb.resetBaseVelocity(self.sphereId1, linearVelocity=[3.0, 0.0, 2.0])
            self.flag_sphere1 = False

        # Check if the robot is in front of the second sphere to trigger it
        if self.flag_sphere2 and (qmes12[1, 0] >= 1.1):
            pyb.resetBaseVelocity(self.sphereId2, linearVelocity=[-3.0, 0.0, 2.0])
            self.flag_sphere2 = False

        # Apply perturbation
        # pyb.applyExternalForce(pyb_sim.robotId, -1, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], pyb.LINK_FRAME)"""

        # Update the PyBullet camera on the robot position to do as if it was attached to the robot
        """pyb.resetDebugVisualizerCamera(cameraDistance=0.75, cameraYaw=+50, cameraPitch=-35,
                                       cameraTargetPosition=[qmes12[0, 0], qmes12[1, 0] + 0.0, 0.0])"""

        oMb_tmp = pin.SE3(pin.Quaternion(qmes12[3:7]), np.array([0.0, 0.0, 0.0]))
        RPY = pin.rpy.matrixToRpy(oMb_tmp.rotation)

        # Update the PyBullet camera on the robot position to do as if it was attached to the robot
        pyb.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=(RPY[2, 0]*(180/3.1415)+45), cameraPitch=-39.9,
                                       cameraTargetPosition=[qmes12[0, 0], qmes12[1, 0] + 0.0, 0.0])

        return 0

    def retrieve_pyb_data(self):
        """Retrieve the position and orientation of the base in world frame as well as its linear and angular velocities
        """

        # Retrieve data from the simulation
        self.jointStates = pyb.getJointStates(self.robotId, self.revoluteJointIndices)  # State of all joints
        self.baseState = pyb.getBasePositionAndOrientation(self.robotId)  # Position and orientation of the trunk
        self.baseVel = pyb.getBaseVelocity(self.robotId)  # Velocity of the trunk

        # Joints configuration and velocity vector for free-flyer + 12 actuators
        self.qmes12 = np.vstack((np.array([self.baseState[0]]).T, np.array([self.baseState[1]]).T,
                                 np.array([[self.jointStates[i_joint][0] for i_joint in range(len(self.jointStates))]]).T))
        self.vmes12 = np.vstack((np.array([self.baseVel[0]]).T, np.array([self.baseVel[1]]).T,
                                 np.array([[self.jointStates[i_joint][1] for i_joint in range(len(self.jointStates))]]).T))

        """robotVirtualOrientation = pyb.getQuaternionFromEuler([0, 0, np.pi / 4])
        self.qmes12[3:7, 0] = robotVirtualOrientation"""

        return 0
