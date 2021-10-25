import numpy as np
import pybullet as pyb
import pybullet_data

class PybEnvironment3D():
    ''' Class to vizualise the 3D environment and foot trajectory and in PyBullet simulation.
    '''

    def __init__(self, params, gait, statePlanner, footStepPlanner, footTrajectoryGenerator):
        """
        Store the solo3D class, used for trajectory vizualisation.
        Args:
        - params: parameters of the simulation.
        - gait: Gait class.
        - statePlanner: State planner class.
        - footStepPlannerQP: Footstep planner class (QP version).
        - footTrajectoryGenerator: Foot trajectory class (Bezier version).
        """
        self.enable_pyb_GUI = params.enable_pyb_GUI
        self.URDF = params.environment_URDF
        self.params = params

        # Solo3D python class
        self.footStepPlanner = footStepPlanner
        self.statePlanner = statePlanner
        self.gait = gait
        self.footTrajectoryGenerator = footTrajectoryGenerator

        self.n_points = 15
        self.trajectory_Ids = np.zeros((3, 4, self.n_points))
        self.sl1m_Ids_target = np.zeros((7, 4))

        # Int to determine when refresh object position (k % refresh == 0)
        self.refresh = 1

    def update(self, k):
        ''' Update position of the objects in pybullet environment.
        Args :
        - k (int) : step of the simulation.
        '''
        # On iteration 0, PyBullet env has not been started
        if k == 1:
            self.initializeEnv()

        if self.enable_pyb_GUI and k > 1 and not self.params.enable_multiprocessing_mip:

            if k % self.refresh == 0:

                # Update target trajectory, current and next phase
                self.updateTargetTrajectory()

        return 0

    def updateCamera(self, k, device):
        # Update position of PyBullet camera on the robot position to do as if it was attached to the robot
        if k > 10 and self.enable_pyb_GUI:
            pyb.resetDebugVisualizerCamera(cameraDistance=0.95,
                                           cameraYaw=357,
                                           cameraPitch=-29,
                                           cameraTargetPosition=[0.6, 0.14, -0.22])
            # pyb.resetDebugVisualizerCamera(cameraDistance=1., cameraYaw=357, cameraPitch=-28,
            #                                cameraTargetPosition=[device.dummyHeight[0], device.dummyHeight[1], 0.0])

        return 0

    def update_target_SL1M(self, all_feet_pos):
        ''' Update position of the SL1M target
        Args :
        -  all_feet_pos : list of optimized position such as : [[Foot 1 next_pos, None , Foot1 next_pos] , [Foot 2 next_pos, None , Foot2 next_pos] ]
        '''

        for i in range(len(all_feet_pos[0])):
            for j in range(len(all_feet_pos)):
                if all_feet_pos[j][i] is None:
                    pyb.resetBasePositionAndOrientation(int(self.sl1m_Ids_target[i, j]),
                                                        posObj=np.array([0., 0., -0.5]),
                                                        ornObj=np.array([0.0, 0.0, 0.0, 1.0]))

                else:
                    pyb.resetBasePositionAndOrientation(int(self.sl1m_Ids_target[i, j]),
                                                        posObj=all_feet_pos[j][i],
                                                        ornObj=np.array([0.0, 0.0, 0.0, 1.0]))

        return 0

    def updateTargetTrajectory(self):
        ''' Update the target trajectory for current and next phases. Hide the unnecessary spheres.
        '''

        gait = self.gait.getCurrentGait()
        fsteps = self.footStepPlanner.getFootsteps()

        for j in range(4):
            # Count the position of the plotted trajectory in the temporal horizon
            # c = 0 --> Current trajectory/foot pos
            # c = 1 --> Next trajectory/foot pos
            c = 0
            i = 0

            for i in range(gait.shape[0]):
                # footsteps = fsteps[i].reshape((3, 4), order="F")
                if i > 0:
                    if (1 - gait[i - 1, j]) * gait[i, j] > 0:  # from flying phase to stance
                        if c == 0:
                            # Current flying phase, using coeff store in Bezier curve class
                            t0 = self.footTrajectoryGenerator.t0s[j]
                            t1 = self.footTrajectoryGenerator.t_swing[j]
                            t_vector = np.linspace(t0, t1, self.n_points)

                            for id_t, t in enumerate(t_vector):
                                # Bezier trajectory
                                pos = self.footTrajectoryGenerator.evaluateBezier(j, 0, t)
                                # Polynomial Curve 5th order
                                # pos = self.footTrajectoryGenerator.evaluatePoly(j, 0, t)
                                pyb.resetBasePositionAndOrientation(int(self.trajectory_Ids[c, j, id_t]),
                                                                    posObj=pos,
                                                                    ornObj=np.array([0.0, 0.0, 0.0, 1.0]))
                            c += 1

                else:
                    if gait[i, j] == 1:
                        # not hidden in the floor, traj
                        if not pyb.getBasePositionAndOrientation(int(self.trajectory_Ids[0, j, 0]))[0][2] == -0.1:
                            for t in range(self.n_points):
                                pyb.resetBasePositionAndOrientation(int(self.trajectory_Ids[0, j, t]),
                                                                    posObj=np.array([0., 0., -0.1]),
                                                                    ornObj=np.array([0.0, 0.0, 0.0, 1.0]))

                        c += 1

                i += 1

            # Hide the sphere objects not used
            while c < self.trajectory_Ids.shape[0]:

                # not hidden in the floor, traj
                if not pyb.getBasePositionAndOrientation(int(self.trajectory_Ids[c, j, 0]))[0][2] == -0.1:
                    for t in range(self.n_points):
                        pyb.resetBasePositionAndOrientation(int(self.trajectory_Ids[c, j, t]),
                                                            posObj=np.array([0., 0., -0.1]),
                                                            ornObj=np.array([0.0, 0.0, 0.0, 1.0]))

                c += 1

        return 0

    def initializeEnv(self):
        '''
        Load objects in pybullet simulation.
        '''
        print("Loading PyBullet object ...")
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load 3D environment
        tmpId = pyb.loadURDF(self.URDF)
        pyb.changeDynamics(tmpId, -1, lateralFriction=1.0)

        # Sphere Object for trajcetories :
        for i in range(self.trajectory_Ids.shape[0]):

            # rgbaColor : [R , B , G , alpha opacity]
            if i == 0:
                rgba = [0.41, 1., 0., 1.]
            else:
                rgba = [0.41, 1., 0., 0.5]

            mesh_scale = [0.0035, 0.0035, 0.0035]
            visualShapeId = pyb.createVisualShape(shapeType=pyb.GEOM_MESH,
                                                  fileName="sphere_smooth.obj",
                                                  halfExtents=[0.5, 0.5, 0.1],
                                                  rgbaColor=rgba,
                                                  specularColor=[0.4, .4, 0],
                                                  visualFramePosition=[0.0, 0.0, 0.0],
                                                  meshScale=mesh_scale)
            for j in range(4):
                for id_t in range(self.n_points):
                    self.trajectory_Ids[i, j, id_t] = pyb.createMultiBody(baseMass=0.0,
                                                                          baseInertialFramePosition=[0, 0, 0],
                                                                          baseVisualShapeIndex=visualShapeId,
                                                                          basePosition=[0.0, 0.0, -0.1],
                                                                          useMaximalCoordinates=True)
        # Sphere Object for SLM
        # 5 phases + init pos
        for i in range(6):

            rgba_list = [[1., 0., 0., 1.], [1., 0., 1., 1.], [1., 1., 0., 1.], [0., 0., 1., 1.]]
            mesh_scale = [0.01, 0.01, 0.01]

            for j in range(4):
                rgba = rgba_list[j]
                rgba[-1] = 1 - (1 / 9) * i
                visualShapeId = pyb.createVisualShape(shapeType=pyb.GEOM_MESH,
                                                      fileName="sphere_smooth.obj",
                                                      halfExtents=[0.5, 0.5, 0.1],
                                                      rgbaColor=rgba,
                                                      specularColor=[0.4, .4, 0],
                                                      visualFramePosition=[0.0, 0.0, 0.0],
                                                      meshScale=mesh_scale)

                self.sl1m_Ids_target[i, j] = pyb.createMultiBody(baseMass=0.0,
                                                                 baseInertialFramePosition=[0, 0, 0],
                                                                 baseVisualShapeIndex=visualShapeId,
                                                                 basePosition=[0.0, 0.0, -0.1],
                                                                 useMaximalCoordinates=True)

        print("PyBullet object loaded")

        return 0
