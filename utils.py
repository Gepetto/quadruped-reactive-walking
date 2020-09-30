import math
import numpy as np

import robots_loader  # Gepetto viewer

import Joystick
import FootstepPlanner
import Logger
import Interface
import Estimator

import pybullet as pyb  # Pybullet server
import pybullet_data
import pinocchio as pin

import time as time

##########################
# ROTATION MATRIX TO RPY #
##########################

# Taken from https://www.learnopencv.com/rotation-matrix-to-euler-angles/

# Checks if a matrix is a valid rotation matrix.


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    Id = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(Id - shouldBeIdentity)
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
    """Load the solo model and initialize the Gepetto viewer if it is enabled

    Args:
        enable_viewer (bool): if the Gepetto viewer is enabled or not
    """

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


def init_objects(dt_tsid, dt_mpc, k_max_loop, k_mpc, n_periods, T_gait, type_MPC, on_solo8,
                 predefined):
    """Create several objects that are used in the control loop

    Args:
        dt_tsid (float): time step of TSID
        dt_mpc (float): time step of the MPC
        k_max_loop (int): maximum number of iterations of the simulation
        k_mpc (int): number of tsid iterations for one iteration of the mpc
        n_periods (int): number of gait periods in the prediction horizon
        T_gait (float): duration of one gait period
        type_MPC (bool): which MPC you want to use (PA's or Thomas')
        on_solo8 (bool): whether we are working on solo8 or not
        predefined (bool): if we are using a predefined reference velocity (True) or a joystick (False)
    """

    # Create Joystick object
    joystick = Joystick.Joystick(predefined)

    # Create footstep planner object
    fstep_planner = FootstepPlanner.FootstepPlanner(dt_mpc, n_periods, T_gait, on_solo8)

    # Create logger object
    logger = Logger.Logger(k_max_loop, dt_tsid, dt_mpc, k_mpc, n_periods, T_gait, type_MPC)

    # Create Interface object
    interface = Interface.Interface()

    # Create Estimator object
    estimator = Estimator.Estimator(dt_tsid, k_max_loop)

    return joystick, fstep_planner, logger, interface, estimator


def display_all(solo, k, sequencer, fstep_planner, ftraj_gen, mpc):
    """Update various objects in the Gepetto viewer: the Solo robot as well as debug spheres

    Args:
        k (int): current iteration of the simulation
        sequencer (object): ContactSequencer object
        fstep_planner (object): FootstepPlanner object
        ftraj_gen (object): FootTrajectoryGenerator object
        mpc (object): MpcSolver object
    """

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
    qu_pinocchio[3:7] = getQuaternion(np.array([mpc.q_w[3:6, 0]])).flatten()
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
    """Wrapper for the PyBullet simulator to initialize the simulation, interact with it
    and use various utility functions

    Args:
        envID (int): identifier of the current environment to be able to handle different scenarios
        use_flat_plane (bool): to use either a flat ground or a rough ground
        enable_pyb_GUI (bool): to display PyBullet GUI or not
        dt (float): time step of the inverse dynamics
    """

    def __init__(self, envID, use_flat_plane, enable_pyb_GUI, dt=0.001):

        # Start the client for PyBullet
        if enable_pyb_GUI:
            pyb.connect(pyb.GUI)
        else:
            pyb.connect(pyb.DIRECT)
        # p.GUI for graphical version
        # p.DIRECT for non-graphical version

        # Load horizontal plane
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Either use a flat ground or a rough terrain
        if use_flat_plane:
            self.planeId = pyb.loadURDF("plane.urdf")  # Flat plane
            self.planeIdbis = pyb.loadURDF("plane.urdf")  # Flat plane
            pyb.resetBasePositionAndOrientation(self.planeIdbis, [20.0, 0, 0], [0, 0, 0, 1])
        else:
            import random
            random.seed(41)
            # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
            heightPerturbationRange = 0.05

            numHeightfieldRows = 256*2
            numHeightfieldColumns = 256*2
            heightfieldData = [0]*numHeightfieldRows*numHeightfieldColumns
            height_prev = 0.0
            for j in range(int(numHeightfieldColumns/2)):
                for i in range(int(numHeightfieldRows/2)):
                    height = random.uniform(0, heightPerturbationRange)  # uniform distribution
                    # height = 0.25*np.sin(2*np.pi*(i-128)/46)  # sinus pattern
                    heightfieldData[2*i+2*j*numHeightfieldRows] = (height + height_prev) * 0.5
                    heightfieldData[2*i+1+2*j*numHeightfieldRows] = height
                    heightfieldData[2*i+(2*j+1)*numHeightfieldRows] = (height + height_prev) * 0.5
                    heightfieldData[2*i+1+(2*j+1)*numHeightfieldRows] = height
                    height_prev = height

            # Create the collision shape based on the height field
            terrainShape = pyb.createCollisionShape(shapeType=pyb.GEOM_HEIGHTFIELD, meshScale=[.05, .05, 1],
                                                    heightfieldTextureScaling=(numHeightfieldRows-1)/2,
                                                    heightfieldData=heightfieldData,
                                                    numHeightfieldRows=numHeightfieldRows,
                                                    numHeightfieldColumns=numHeightfieldColumns)
            self.planeId = pyb.createMultiBody(0, terrainShape)
            pyb.resetBasePositionAndOrientation(self.planeId, [0, 0, 0], [0, 0, 0, 1])
            pyb.changeVisualShape(self.planeId, -1, rgbaColor=[1, 1, 1, 1])

        if envID == 1:

            # Add stairs with platform and bridge
            self.stairsId = pyb.loadURDF("../../../../../Documents/Git-Repositories/mpc-tsid/bauzil_stairs.urdf")  # ,
            """basePosition=[-1.25, 3.5, -0.1],
                                 baseOrientation=pyb.getQuaternionFromEuler([0.0, 0.0, 3.1415]))"""
            pyb.changeDynamics(self.stairsId, -1, lateralFriction=1.0)

            # Create the red steps to act as small perturbations
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

            # Create the green steps to act as bigger perturbations
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

            # Create sphere obstacles that will be thrown toward the quadruped
            mesh_scale = [0.1, 0.1, 0.1]
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

            self.sphereId1 = pyb.createMultiBody(baseMass=0.4,
                                                 baseInertialFramePosition=[0, 0, 0],
                                                 baseCollisionShapeIndex=collisionShapeId,
                                                 baseVisualShapeIndex=visualShapeId,
                                                 basePosition=[-0.6, 0.9, 0.1],
                                                 useMaximalCoordinates=True)

            self.sphereId2 = pyb.createMultiBody(baseMass=0.4,
                                                 baseInertialFramePosition=[0, 0, 0],
                                                 baseCollisionShapeIndex=collisionShapeId,
                                                 baseVisualShapeIndex=visualShapeId,
                                                 basePosition=[0.6, 1.1, 0.1],
                                                 useMaximalCoordinates=True)

            # Flag to launch the two spheres in the environment toward the robot
            self.flag_sphere1 = True
            self.flag_sphere2 = True

        # Create blue spheres without collision box for debug purpose
        mesh_scale = [0.015, 0.015, 0.015]
        visualShapeId = pyb.createVisualShape(shapeType=pyb.GEOM_MESH,
                                              fileName="sphere_smooth.obj",
                                              halfExtents=[0.5, 0.5, 0.1],
                                              rgbaColor=[0.0, 0.0, 1.0, 1.0],
                                              specularColor=[0.4, .4, 0],
                                              visualFramePosition=[0.0, 0.0, 0.0],
                                              meshScale=mesh_scale)

        self.ftps_Ids = np.zeros((4, 5), dtype=np.int)
        for i in range(4):
            for j in range(5):
                self.ftps_Ids[i, j] = pyb.createMultiBody(baseMass=0.0,
                                                          baseInertialFramePosition=[0, 0, 0],
                                                          baseVisualShapeIndex=visualShapeId,
                                                          basePosition=[0.0, 0.0, -0.1],
                                                          useMaximalCoordinates=True)

        # Create green spheres without collision box for debug purpose
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
        robotStartOrientation = pyb.getQuaternionFromEuler([0.0, 0.0, 0.0])  # -np.pi/2
        pyb.setAdditionalSearchPath("/opt/openrobots/share/example-robot-data/robots/solo_description/robots")
        self.robotId = pyb.loadURDF("solo12.urdf", robotStartPos, robotStartOrientation)

        # Disable default motor control for revolute joints
        self.revoluteJointIndices = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
        pyb.setJointMotorControlArray(self.robotId, jointIndices=self.revoluteJointIndices,
                                      controlMode=pyb.VELOCITY_CONTROL,
                                      targetVelocities=[0.0 for m in self.revoluteJointIndices],
                                      forces=[0.0 for m in self.revoluteJointIndices])

        # Initialize the robot in a specific configuration
        self.straight_standing = np.array([[0, 0.8, -1.6, 0, 0.8, -1.6, 0, -0.8, 1.6, 0, -0.8, 1.6]]).transpose()
        pyb.resetJointStatesMultiDof(self.robotId, self.revoluteJointIndices, self.straight_standing)  # q0[7:])

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

    def check_pyb_env(self, k, envID, velID, qmes12):
        """Check the state of the robot to trigger events and update camera

        Args:
            k (int): Number of inv dynamics iterations since the start of the simulation
            envID (int): Identifier of the current environment to be able to handle different scenarios
            velID (int): Identifier of the current velocity profile to be able to handle different scenarios
            qmes12 (19x1 array): the position/orientation of the trunk and angular position of actuators

        """

        # If spheres are loaded
        if envID == 1:
            # Check if the robot is in front of the first sphere to trigger it
            if self.flag_sphere1 and (qmes12[1, 0] >= 0.9):
                pyb.resetBaseVelocity(self.sphereId1, linearVelocity=[2.5, 0.0, 2.0])
                self.flag_sphere1 = False

            # Check if the robot is in front of the second sphere to trigger it
            if self.flag_sphere2 and (qmes12[1, 0] >= 1.1):
                pyb.resetBaseVelocity(self.sphereId2, linearVelocity=[-2.5, 0.0, 2.0])
                self.flag_sphere2 = False

            # Create the red steps to act as small perturbations
            """if k == 10:
                pyb.setAdditionalSearchPath(pybullet_data.getDataPath())

                mesh_scale = [2.0, 2.0, 0.3]
                visualShapeId = pyb.createVisualShape(shapeType=pyb.GEOM_MESH,
                                                      fileName="cube.obj",
                                                      halfExtents=[0.5, 0.5, 0.1],
                                                      rgbaColor=[0.0, 0.0, 1.0, 1.0],
                                                      specularColor=[0.4, .4, 0],
                                                      visualFramePosition=[0.0, 0.0, 0.0],
                                                      meshScale=mesh_scale)

                collisionShapeId = pyb.createCollisionShape(shapeType=pyb.GEOM_MESH,
                                                            fileName="cube.obj",
                                                            collisionFramePosition=[0.0, 0.0, 0.0],
                                                            meshScale=mesh_scale)

                tmpId = pyb.createMultiBody(baseMass=0.0,
                                            baseInertialFramePosition=[0, 0, 0],
                                            baseCollisionShapeIndex=collisionShapeId,
                                            baseVisualShapeIndex=visualShapeId,
                                            basePosition=[0.0, 0.0, 0.15],
                                            useMaximalCoordinates=True)
                pyb.changeDynamics(tmpId, -1, lateralFriction=1.0)

                pyb.resetBasePositionAndOrientation(self.robotId, [0, 0, 0.5], [0, 0, 0, 1])"""
            if k == 10:
                pyb.setAdditionalSearchPath(pybullet_data.getDataPath())

                mesh_scale = [0.1, 0.1, 0.04]
                visualShapeId = pyb.createVisualShape(shapeType=pyb.GEOM_MESH,
                                                      fileName="cube.obj",
                                                      halfExtents=[0.5, 0.5, 0.1],
                                                      rgbaColor=[0.0, 0.0, 1.0, 1.0],
                                                      specularColor=[0.4, .4, 0],
                                                      visualFramePosition=[0.0, 0.0, 0.0],
                                                      meshScale=mesh_scale)

                collisionShapeId = pyb.createCollisionShape(shapeType=pyb.GEOM_MESH,
                                                            fileName="cube.obj",
                                                            collisionFramePosition=[0.0, 0.0, 0.0],
                                                            meshScale=mesh_scale)

                tmpId = pyb.createMultiBody(baseMass=0.0,
                                            baseInertialFramePosition=[0, 0, 0],
                                            baseCollisionShapeIndex=collisionShapeId,
                                            baseVisualShapeIndex=visualShapeId,
                                            basePosition=[0.19, 0.15005, 0.02],
                                            useMaximalCoordinates=True)
                pyb.changeDynamics(tmpId, -1, lateralFriction=1.0)
                pyb.resetBasePositionAndOrientation(self.robotId, [0, 0, 0.25], [0, 0, 0, 1])

        # Apply perturbations by directly applying external forces on the robot trunk
        if velID == 4:
            self.apply_external_force(k, 4250, 500, np.array([0.0, 0.0, -3.0]), np.zeros((3,)))
            self.apply_external_force(k, 5250, 500, np.array([0.0, +3.0, 0.0]), np.zeros((3,)))

        # Update the PyBullet camera on the robot position to do as if it was attached to the robot
        """pyb.resetDebugVisualizerCamera(cameraDistance=0.75, cameraYaw=+50, cameraPitch=-35,
                                       cameraTargetPosition=[qmes12[0, 0], qmes12[1, 0] + 0.0, 0.0])"""

        # Get the orientation of the robot to change the orientation of the camera with the rotation of the robot
        oMb_tmp = pin.SE3(pin.Quaternion(qmes12[3:7]), np.array([0.0, 0.0, 0.0]))
        RPY = pin.rpy.matrixToRpy(oMb_tmp.rotation)

        # Update the PyBullet camera on the robot position to do as if it was attached to the robot
        pyb.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=(0.0*RPY[2]*(180/3.1415)+45), cameraPitch=-39.9,
                                       cameraTargetPosition=[qmes12[0, 0], qmes12[1, 0] + 0.0, 0.0])  # qmes12[2, 0]-0.15])

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
                                 np.array([[state[0] for state in self.jointStates]]).T))
        self.vmes12 = np.vstack((np.array([self.baseVel[0]]).T, np.array([self.baseVel[1]]).T,
                                 np.array([[state[1] for state in self.jointStates]]).T))

        """robotVirtualOrientation = pyb.getQuaternionFromEuler([0, 0, np.pi / 4])
        self.qmes12[3:7, 0] = robotVirtualOrientation"""

        # Add uncertainty to feedback from PyBullet to mimic imperfect measurements
        """tmp = np.array([pyb.getQuaternionFromEuler(pin.rpy.matrixToRpy(
            pin.Quaternion(self.qmes12[3:7, 0:1]).toRotationMatrix())
            + np.random.normal(0, 0.03, (3,)))])
        self.qmes12[3:7, 0] = tmp[0, :]
        self.vmes12[0:6, 0] += np.random.normal(0, 0.01, (6,))"""

        return 0

    def apply_external_force(self, k, start, duration, F, M):
        """Apply an external force/momentum to the robot
        4-th order polynomial: zero force and force velocity at start and end
        (bell-like force trajectory)

        Args:
            k (int): numero of the current iteration of the simulation
            start (int): numero of the iteration for which the force should start to be applied
            duration (int): number of iterations the force should last
            F (3x array): components of the force in PyBullet world frame
            M (3x array): components of the force momentum in PyBullet world frame
        """

        if ((k < start) or (k > (start+duration))):
            return 0.0
        """if k == start:
            print("Applying [", F[0], ", ", F[1], ", ", F[2], "]")"""

        ev = k - start
        t1 = duration
        A4 = 16 / (t1**4)
        A3 = - 2 * t1 * A4
        A2 = t1**2 * A4
        alpha = A2*ev**2 + A3*ev**3 + A4*ev**4
        pyb.applyExternalForce(self.robotId, -1, alpha*F, alpha*M, pyb.LINK_FRAME)

        return 0.0

    def get_to_default_position(self, qtarget):
        """Controler that tries to get the robot back to a default angular positions
        of its 12 actuators using polynomials to generate trajectories and a PD controller
        to make the actuators follow them

        Args:
            qtarget (12x1 array): the target position for the 12 actuators
        """

        qmes = np.zeros((12, 1))
        vmes = np.zeros((12, 1))

        # Retrieve angular position and velocity of actuators
        jointStates = pyb.getJointStates(self.robotId, self.revoluteJointIndices)
        qmes[:, 0] = [state[0] for state in jointStates]
        vmes[:, 0] = [state[1] for state in jointStates]

        # Create trajectory
        dt_traj = 0.002
        t1 = 4.0  # seconds
        cpt = 0

        # PD settings
        P = 1.0 * 3.0
        D = 0.05 * np.array([[1.0, 0.3, 0.3, 1.0, 0.3, 0.3, 1.0, 0.3, 0.3, 1.0, 0.3, 0.3]]).transpose()

        while True or np.max(np.abs(qtarget - qmes)) > 0.1:

            time_loop = time.time()

            # Retrieve angular position and velocity of actuators
            jointStates = pyb.getJointStates(self.robotId, self.revoluteJointIndices)
            qmes[:, 0] = [state[0] for state in jointStates]
            vmes[:, 0] = [state[1] for state in jointStates]

            # Torque PD controller
            if (cpt * dt_traj < t1):
                ev = dt_traj * cpt
                A3 = 2 * (qmes - qtarget) / t1**3
                A2 = (-3/2) * t1 * A3
                qdes = qmes + A2*(ev**2) + A3*(ev**3)
                vdes = 2*A2*ev + 3*A3*(ev**2)
            jointTorques = P * (qdes - qmes) + D * (vdes - vmes)

            # Saturation to limit the maximal torque
            t_max = 2.5
            jointTorques[jointTorques > t_max] = t_max
            jointTorques[jointTorques < -t_max] = -t_max

            # Set control torque for all joints
            pyb.setJointMotorControlArray(self.robotId, self.revoluteJointIndices,
                                          controlMode=pyb.TORQUE_CONTROL, forces=jointTorques)

            # Compute one step of simulation
            pyb.stepSimulation()

            # Increment loop counter
            cpt += 1

            while (time.time() - time_loop) < dt_traj:
                pass
