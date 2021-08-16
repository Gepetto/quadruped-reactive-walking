import numpy as np
import pybullet as pyb
import pybullet_data
from time import perf_counter as clock
import pinocchio as pin
# from solo3D.tools.geometry import EulerToQuaternion
import os 

class PybVisualizationTraj():
    ''' Class used to vizualise the feet trajectory on the pybullet simulation 
    '''

    def __init__(self, footTrajectoryGenerator, enable_pyb_GUI):

        # Pybullet enabled
        self.enable_pyb_GUI = enable_pyb_GUI

        self.footTrajectoryGenerator = footTrajectoryGenerator


        # self.ftps_Ids_target = [0] * 4
        # self.n_points = 15

        # n_points for the traj , 4 feet , 3 target max in futur
        # self.trajectory_Ids = np.zeros((3, 4, self.n_points))
        self.ftps_Ids_target = np.zeros((3, 4))



    def update(self, k , device , o_targetFootstep_mpc, o_targetFootstep_planner) :
        ''' Update position of the objects in pybullet environment.
        Args :
        - k (int) : step of the simulation
        - device (object) : Class used to set up the pybullet env
    '''

        if k == 1:  # k ==0, device is still a dummy object, pyb env did not started
            self.initializeEnv()


        if self.enable_pyb_GUI and k > 100:

            self.updateTarget(o_targetFootstep_mpc, o_targetFootstep_planner)

        #     # Update position of PyBullet camera
        #     # self.updateCamera(k , device)

        #     if k % self.refresh == 0:

        #         t_init = clock()

        #         # # Update target trajectory, current and next phase
        #         self.updateTargetTrajectory()

        #         # # Update refrence trajectory
        #         # self.updateRefTrajectory()

        #         # update constraints
        #         # self.updateConstraints()

        #         #update Sl1M :
        #         # self.updateSl1M_target(all_feet_pos)

        #         t_end = clock() - t_init
                # print("Time for update pybullet = ",  t_end)

        return 0

    def updateTarget(self, o_targetFootstep_mpc, o_targetFootstep_planner ):
        
        for foot in range(4) :
            pyb.resetBasePositionAndOrientation(int(self.ftps_Ids_target[0, foot]),
                                                            posObj=o_targetFootstep_mpc[:, foot],
                                                            ornObj=np.array([0.0, 0.0, 0.0, 1.0]))

        target_footstep_curve = self.footTrajectoryGenerator.getFootPosition()
        for foot in range(4) :
            pyb.resetBasePositionAndOrientation(int(self.ftps_Ids_target[1, foot]),
                                                            posObj=target_footstep_curve[:, foot],
                                                            ornObj=np.array([0.0, 0.0, 0.0, 1.0]))

        # for foot in range(4) :
        #     pyb.resetBasePositionAndOrientation(int(self.ftps_Ids_target[2, foot]),
        #                                                         posObj=o_targetFootstep_planner[:, foot],
        #                                                         ornObj=np.array([0.0, 0.0, 0.0, 1.0]))


        return 0
    
    def initializeEnv(self):
        ''' Load object in pybullet environment.
        '''

        print("Loading pybullet object ...")

        pyb.setAdditionalSearchPath(pybullet_data.getDataPath())   
        for i in range(self.ftps_Ids_target.shape[0]):  # nb of feet target in futur

            # rgbaColor : [R , B , G , alpha opacity]
            if i == 0:
                rgba = [0., 1., 0., 1.]

            elif i == 1 :
                rgba = [1., 0., 0., 1.]
            else :
                rgba = [0., 0., 1., 1.]


            mesh_scale = [0.008, 0.008, 0.008]
            visualShapeId = pyb.createVisualShape(shapeType=pyb.GEOM_MESH,
                                                  fileName="sphere_smooth.obj",
                                                  halfExtents=[0.5, 0.5, 0.1],
                                                  rgbaColor=rgba,
                                                  specularColor=[0.4, .4, 0],
                                                  visualFramePosition=[0.0, 0.0, 0.0],
                                                  meshScale=mesh_scale)
            for j in range(4):
                self.ftps_Ids_target[i, j] = pyb.createMultiBody(baseMass=0.0,
                                                                 baseInertialFramePosition=[0, 0, 0],
                                                                 baseVisualShapeIndex=visualShapeId,
                                                                 basePosition=[0.0, 0.0, -0.1],
                                                                 useMaximalCoordinates=True)
        return 0 



    # def updateConstraints(self):

    #     gait = self.gait.getCurrentGait() 
    #     fsteps =  self.footStepPlannerQP.getFootsteps()
    #     footsteps = fsteps[0].reshape((3,4) , order ="F")
        
    #     feet = np.where(gait[0,:] == 1)[0]
        
    #     for i in range(4):
    #         if i in feet :
    #             pyb.resetBasePositionAndOrientation(int( self.constraints_objects[i]),
    #                                             posObj=footsteps[:,i],
    #                                             ornObj=np.array([0.0, 0.0, 0.0, 1.0]))           

    #         else :
    #             pyb.resetBasePositionAndOrientation(int( self.constraints_objects[i]),
    #                                             posObj=np.array([0.,0.,-1]),
    #                                             ornObj=np.array([0.0, 0.0, 0.0, 1.0]))     

    #     return 0

    # def updateCamera(self, k, device):
    #     # Update position of PyBullet camera on the robot position to do as if it was attached to the robot
    #     if k > 10 and self.enable_pyb_GUI:
    #         pyb.resetDebugVisualizerCamera(cameraDistance=0.95, cameraYaw=357, cameraPitch=-29,
    #                                        cameraTargetPosition=[0.6, 0.14, -0.22])
    #         # pyb.resetDebugVisualizerCamera(cameraDistance=1., cameraYaw=357, cameraPitch=-28,
    #         #                                cameraTargetPosition=[device.dummyHeight[0], device.dummyHeight[1], 0.0])

    #     return 0

    # def updateSl1M_target(self,all_feet_pos) :
    #     ''' Update position of the SL1M target
    #     Args : 
    #         -  all_feet_pos : list of optimized position such as : [[Foot 1 next_pos, None , Foot1 next_pos] , [Foot 2 next_pos, None , Foot2 next_pos] ]
    #     '''

    #     for i in range(len(all_feet_pos[0])) :
    #         for j in range(len(all_feet_pos)) :
    #             if all_feet_pos[j][i] is None : 
    #                 pyb.resetBasePositionAndOrientation(int(self.sl1m_Ids_target[i, j]),
    #                                                                 posObj=np.array([0., 0., -0.5]),
    #                                                                 ornObj=np.array([0.0, 0.0, 0.0, 1.0]))                   
                
    #             else :
    #                 pyb.resetBasePositionAndOrientation(int(self.sl1m_Ids_target[i, j]),
    #                                                                 posObj=all_feet_pos[j][i],
    #                                                                 ornObj=np.array([0.0, 0.0, 0.0, 1.0]))     
                    
                


    #     return 0 

    # def updateRefTrajectory(self):
    #     ''' Update the reference state trajectory given by StatePlanner Class.
    #     '''

    #     referenceState = self.statePlanner.getReferenceStates()

    #     for k in range(self.state_Ids.shape[0]):

    #         quat = EulerToQuaternion(referenceState[3:6, k])
    #         pyb.resetBasePositionAndOrientation(int(self.state_Ids[k]),
    #                                             posObj=referenceState[:3, k] + np.array([0., 0., 0.04]),
    #                                             ornObj=quat)

    #     # Update surface :

    #     a, b, c = self.statePlanner.surface_equation
    #     pos = self.statePlanner.surface_point

    #     RPY = np.array([-np.arctan2(b, 1.), -np.arctan2(a, 1.), 0.])
    #     quat = EulerToQuaternion(RPY)
    #     pyb.resetBasePositionAndOrientation(int(self.surface_Id),
    #                                         posObj=pos,
    #                                         ornObj=quat)

    #     return 0

    # def updateTargetTrajectory(self):
    #     ''' Update the target trajectory for current and next phases. Hide the unnecessary spheres.
    #     '''

    #     gait = self.gait.getCurrentGait()
    #     fsteps =  self.footStepPlannerQP.getFootsteps()

    #     for j in range(4):

    #         # Count the position of the plotted trajectory in the temporal horizon
    #         # c = 0 --> Current trajectory/foot pos
    #         # c = 1 --> Next trajectory/foot pos
    #         c = 0
    #         i = 0

    #         init_pos = np.zeros(3)

    #         while gait[i, :].any():
    #             footsteps = fsteps[i].reshape((3,4) , order ="F")
    #             if i > 0:
    #                 if (1 - gait[i-1, j]) * gait[i, j] > 0:  # from flying phase to stance


                        

    #                     # In any case plot target
    #                     pyb.resetBasePositionAndOrientation(int(self.ftps_Ids_target[c, j]),
    #                                                         posObj=footsteps[:, j],
    #                                                         ornObj=np.array([0.0, 0.0, 0.0, 1.0]))
    #                     if c == 0:
    #                         # Current flying phase, using coeff store in Bezier curve class
    #                         t0 = self.footTrajectoryGenerator.t0s[j]
    #                         t1 = self.footTrajectoryGenerator.t_swing[j]
    #                         t_vector = np.linspace(t0, t1, self.n_points)

    #                         for id_t, t in enumerate(t_vector):
    #                             # Bezier trajectory
    #                             pos = self.footTrajectoryGenerator.evaluateBezier(j, 0, t)
    #                             # Polynomial Curve 5th order
    #                             # pos = self.footTrajectoryGenerator.evaluatePoly(j, 0, t)
    #                             pyb.resetBasePositionAndOrientation(int(self.trajectory_Ids[c, j, id_t]),
    #                                                                 posObj=pos,
    #                                                                 ornObj=np.array([0.0, 0.0, 0.0, 1.0]))

    #                     else:
    #                         # Next phase, using a simple polynomial curve to approximate the next trajectory
    #                         t0 = 0.
    #                         t1 = self.gait.getPhaseDuration(i, j, 1.0)
    #                         t_vector = np.linspace(t0, t1, self.n_points)

    #                         self.footTrajectoryGenerator.updatePolyCoeff_simple(j, init_pos, footsteps[:, j], t1)
    #                         for id_t, t in enumerate(t_vector):
    #                             pos = self.footTrajectoryGenerator.evaluatePoly_simple(j, 0, t)
    #                             pyb.resetBasePositionAndOrientation(int(self.trajectory_Ids[c, j, id_t]),
    #                                                                 posObj=pos,
    #                                                                 ornObj=np.array([0.0, 0.0, 0.0, 1.0]))

    #                     c += 1

    #                 if gait[i-1, j] * (1 - gait[i, j]) > 0:
    #                     # Starting a flying phase
    #                     footsteps_previous = fsteps[i-1].reshape((3,4) , order ="F")
    #                     init_pos[:] = footsteps_previous[:, j]

    #             else:
    #                 if gait[i, j] == 1:
    #                     # foot already on the ground
    #                     pyb.resetBasePositionAndOrientation(int(self.ftps_Ids_target[c, j]),
    #                                                         posObj=footsteps[:, j],
    #                                                         ornObj=np.array([0.0, 0.0, 0.0, 1.0]))

    #                     # not hidden in the floor, traj
    #                     if not pyb.getBasePositionAndOrientation(int(self.trajectory_Ids[0, j, 0]))[0][2] == -0.1:
    #                         for t in range(self.n_points):
    #                             pyb.resetBasePositionAndOrientation(int(self.trajectory_Ids[0, j, t]),
    #                                                                 posObj=np.array([0., 0., -0.1]),
    #                                                                 ornObj=np.array([0.0, 0.0, 0.0, 1.0]))

    #                     c += 1

    #             i += 1

    #         # Hide the sphere objects not used
    #         while c < self.ftps_Ids_target.shape[0]:

    #             # not hidden in the floor, target
    #             if not pyb.getBasePositionAndOrientation(int(self.ftps_Ids_target[c, j]))[0][2] == -0.1:
    #                 pyb.resetBasePositionAndOrientation(int(self.ftps_Ids_target[c, j]),
    #                                                     posObj=np.array([0., 0., -0.1]),
    #                                                     ornObj=np.array([0.0, 0.0, 0.0, 1.0]))

    #             # not hidden in the floor, traj
    #             if not pyb.getBasePositionAndOrientation(int(self.trajectory_Ids[c, j, 0]))[0][2] == -0.1:
    #                 for t in range(self.n_points):
    #                     pyb.resetBasePositionAndOrientation(int(self.trajectory_Ids[c, j, t]),
    #                                                         posObj=np.array([0., 0., -0.1]),
    #                                                         ornObj=np.array([0.0, 0.0, 0.0, 1.0]))

    #             c += 1

    #     return 0

    # def initializeEnv(self):
    #     ''' Load object in pybullet environment.
    #     '''

    #     print("Loading pybullet object ...")

    #     pyb.setAdditionalSearchPath(pybullet_data.getDataPath())    

        # Stairs object  
        # Add stairs with platform and bridge
        # self.stairsId = pyb.loadURDF(self.ENV_URDF)  
        # pyb.changeDynamics(self.stairsId, -1, lateralFriction=1.0)
        #pyb.setAdditionalSearchPath("/home/corberes/Bureau/edin/my_quad_reactive/quadruped-reactive-walking/scripts/solo3D/objects/")
        # path_mesh = "solo3D/objects/object_" + str(self.object_stair) + "/meshes/"
        # for elt in os.listdir(path_mesh) :
        #     name_object = path_mesh + elt
        
        #     mesh_scale = [1.0, 1., 1.]
        #     # Add stairs with platform and bridge
        #     visualShapeId = pyb.createVisualShape(shapeType=pyb.GEOM_MESH,
        #                                         fileName=name_object,
        #                                         rgbaColor=[.3, 0.3, 0.3, 1.0],
        #                                         specularColor=[0.4, .4, 0],
        #                                         visualFramePosition=[0.0, 0.0, 0.0],
        #                                         meshScale=mesh_scale)

        #     collisionShapeId = pyb.createCollisionShape(shapeType=pyb.GEOM_MESH,
        #                                                             fileName=name_object,
        #                                                             collisionFramePosition=[0.0, 0.0, 0.0],
        #                                                             meshScale=mesh_scale)

        #     tmpId = pyb.createMultiBody(baseMass=0.0,
        #                                                 baseInertialFramePosition=[0, 0, 0],
        #                                                 baseCollisionShapeIndex=collisionShapeId,
        #                                                 baseVisualShapeIndex=visualShapeId,
        #                                                 basePosition=[0.0, 0., 0.0],
        #                                                 useMaximalCoordinates=True)
        #     pyb.changeDynamics(tmpId, -1, lateralFriction=1.0)   
        
        # Sphere Object for target footsteps :
        # for i in range(self.ftps_Ids_target.shape[0]):  # nb of feet target in futur

        #     # rgbaColor : [R , B , G , alpha opacity]
        #     if i == 0:
        #         rgba = [0.41, 1., 0., 1.]

        #     else:
        #         rgba = [0.41, 1., 0., 0.5]

        #     mesh_scale = [0.008, 0.008, 0.008]
        #     visualShapeId = pyb.createVisualShape(shapeType=pyb.GEOM_MESH,
        #                                           fileName="sphere_smooth.obj",
        #                                           halfExtents=[0.5, 0.5, 0.1],
        #                                           rgbaColor=rgba,
        #                                           specularColor=[0.4, .4, 0],
        #                                           visualFramePosition=[0.0, 0.0, 0.0],
        #                                           meshScale=mesh_scale)
        #     for j in range(4):
        #         self.ftps_Ids_target[i, j] = pyb.createMultiBody(baseMass=0.0,
        #                                                          baseInertialFramePosition=[0, 0, 0],
        #                                                          baseVisualShapeIndex=visualShapeId,
        #                                                          basePosition=[0.0, 0.0, -0.1],
        #                                                          useMaximalCoordinates=True)

        # Sphere Object for trajcetories :
        # for i in range(self.trajectory_Ids.shape[0]):

        #     # rgbaColor : [R , B , G , alpha opacity]
        #     if i == 0:
        #         rgba = [0.41, 1., 0., 1.]
        #     else:
        #         rgba = [0.41, 1., 0., 0.5]

        #     mesh_scale = [0.0035, 0.0035, 0.0035]
        #     visualShapeId = pyb.createVisualShape(shapeType=pyb.GEOM_MESH,
        #                                           fileName="sphere_smooth.obj",
        #                                           halfExtents=[0.5, 0.5, 0.1],
        #                                           rgbaColor=rgba,
        #                                           specularColor=[0.4, .4, 0],
        #                                           visualFramePosition=[0.0, 0.0, 0.0],
        #                                           meshScale=mesh_scale)
        #     for j in range(4):
        #         for id_t in range(self.n_points):
        #             self.trajectory_Ids[i, j, id_t] = pyb.createMultiBody(baseMass=0.0,
        #                                                                   baseInertialFramePosition=[0, 0, 0],
        #                                                                   baseVisualShapeIndex=visualShapeId,
        #                                                                   basePosition=[0.0, 0.0, -0.1],
        #                                                                   useMaximalCoordinates=True)

        # # Cube for planner trajectory
        # for i in range(self.state_Ids.shape[0]):
        #     # rgbaColor : [R , B , G , alpha opacity]
        #     if i == 0:
        #         rgba = [0., 0., 1., 1.]
        #     else:
        #         rgba = [0., 0., 1., 1.]

        #     mesh_scale = [0.005, 0.005, 0.005]
        #     visualShapeId = pyb.createVisualShape(shapeType=pyb.GEOM_MESH,
        #                                           fileName="sphere_smooth.obj",
        #                                           halfExtents=[0.5, 0.5, 0.1],
        #                                           rgbaColor=rgba,
        #                                           specularColor=[0.4, .4, 0],
        #                                           visualFramePosition=[0.0, 0.0, 0.0],
        #                                           meshScale=mesh_scale)

        #     self.state_Ids[i] = pyb.createMultiBody(baseMass=0.0,
        #                                             baseInertialFramePosition=[0, 0, 0],
        #                                             baseVisualShapeIndex=visualShapeId,
        #                                             basePosition=[0.0, 0.0, -0.1],
        #                                             useMaximalCoordinates=True)

        # # Cube for surface Planner

        # rgba = [0., 0., 1., 0.2]

        # mesh_scale = [2*self.statePlanner.FIT_SIZE_X, 2*self.statePlanner.FIT_SIZE_Y, 0.001]
        # visualShapeId = pyb.createVisualShape(shapeType=pyb.GEOM_MESH,
        #                                       fileName="cube.obj",
        #                                       halfExtents=[0.5, 0.5, 0.1],
        #                                       rgbaColor=rgba,
        #                                       specularColor=[0.4, .4, 0],
        #                                       visualFramePosition=[0.0, 0.0, 0.0],
        #                                       meshScale=mesh_scale)

        # self.surface_Id = pyb.createMultiBody(baseMass=0.0,
        #                                                 baseInertialFramePosition=[0, 0, 0],
        #                                                 baseVisualShapeIndex=visualShapeId,
        #                                                 basePosition=[0.0, 0.0, -0.1],
        #                                                 useMaximalCoordinates=True)

        # Object for constraints : 
        # n_effectors = 4 
        # # kinematic_constraints_path = "/home/thomas_cbrs/Library/solo-rbprm/data/com_inequalities/feet_quasi_flat/"
        # # limbs_names = ["FLleg" , "FRleg" , "HLleg" , "HRleg"]
        # kinematic_constraints_path = os.getcwd() + "/solo3D/objects/constraints_sphere/"
        # limbs_names = ["FL" , "FR" , "HL" , "HR"]
        # for foot in range(n_effectors):
        #     foot_name = limbs_names[foot]
        #     # filekin = kinematic_constraints_path + "COM_constraints_in_" + \
        #     #     foot_name + "_effector_frame_quasi_static_reduced.obj"
        #     filekin = kinematic_constraints_path + "COM_constraints_" + \
        #         foot_name + "3.obj"

        #     mesh_scale = [1.,1.,1.] 
        #     rgba = [0.,0.,1.,0.2]
        #     visualShapeId = pyb.createVisualShape(shapeType=pyb.GEOM_MESH,
        #                                             fileName=filekin,
        #                                             rgbaColor=rgba,   
        #                                             specularColor=[0.4, .4, 0],
        #                                             visualFramePosition=[0.0, 0.0, 0.0],
        #                                             meshScale=mesh_scale)

        #     self.constraints_objects[foot]  = pyb.createMultiBody(baseMass=0.0,
        #                                                 baseInertialFramePosition=[0, 0, 0],
        #                                                 baseVisualShapeIndex=visualShapeId,
        #                                                 basePosition=[0.0, 0.0, -0.1],
        #                                                 useMaximalCoordinates=True)

        # Sphere Object for trajcetories :
        # 5 phases + init pos
        # for i in range(6):

        #     # rgbaColor : [R , B , G , alpha opacity]
        #     # if i == 0 :
        #     #     rgba = [1., 0., 1., 1.]
        #     # else :
        #     #     rgba = [1., (1/7)*i, 0., 1.]

        #     rgba_list = [[1.,0.,0.,1.],
        #             [1.,0.,1.,1.],
        #             [1.,1.,0.,1.],
        #             [0.,0.,1.,1.]]
            

        #     mesh_scale = [0.01, 0.01, 0.01]
            
        #     for j in range(4):
        #         rgba = rgba_list[j]
        #         rgba[-1] = 1 - (1/9)*i
        #         visualShapeId = pyb.createVisualShape(shapeType=pyb.GEOM_MESH,
        #                                           fileName="sphere_smooth.obj",
        #                                           halfExtents=[0.5, 0.5, 0.1],
        #                                           rgbaColor=rgba,
        #                                           specularColor=[0.4, .4, 0],
        #                                           visualFramePosition=[0.0, 0.0, 0.0],
        #                                           meshScale=mesh_scale)
                
                
        #         self.sl1m_Ids_target[i, j] = pyb.createMultiBody(baseMass=0.0,
        #                                                                 baseInertialFramePosition=[0, 0, 0],
        #                                                                 baseVisualShapeIndex=visualShapeId,
        #                                                                 basePosition=[0.0, 0.0, -0.1],
        #                                                                 useMaximalCoordinates=True)

        

        print("pybullet object loaded")

        return 0