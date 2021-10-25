import numpy as np
import pinocchio as pin
from sl1m.solver import solve_least_square
import pickle

# from solo3D.tools.ProfileWrapper import ProfileWrapper

# Store the results from cprofile
# profileWrap = ProfileWrapper()


class StatePlanner():

    def __init__(self, params):
        self.n_surface_configs = 3
        self.h_ref = params.h_ref
        self.dt_mpc = params.dt_mpc
        self.T_step = params.T_gait / 2

        self.n_steps = int(params.gait.shape[0])
        self.referenceStates = np.zeros((12, 1 + self.n_steps))

        filehandler = open(params.environment_heightmap, 'rb')
        self.map = pickle.load(filehandler)
        self.FIT_SIZE_X = 0.4
        self.FIT_SIZE_Y = 0.4
        self.surface_point = np.zeros(3)

        self.configs = [np.zeros(7) for _ in range(self.n_surface_configs)]

        self.result = [0., 0., 0.]

    def computeReferenceStates(self, q, v, v_ref, new_step=False):
        '''
        - q (7x1) : [px , py , pz , x , y , z , w]  --> here x,y,z,w quaternion
        - v (6x1) : current v linear in world frame
        - v_ref (6x1) : vref in world frame
        '''
        rpy = q[3:6]
        if new_step:
            # id_x, id_y = self.rpy_map.map_index(q[0], q[1])
            # self.result = self.rpy_map.result[id_x, id_y]
            self.result = self.compute_mean_surface(q[:3])
            # from IPython import embed
            # embed()

        # Update the current state
        # self.referenceStates[:3, 0] = q[:3] + pin.rpy.rpyToMatrix(rpy).dot(np.array([-0.04, 0., 0.]))
        self.referenceStates[:2, 0] = np.zeros(2)
        self.referenceStates[2, 0] = q[2]
        self.referenceStates[3:6, 0] = rpy
        self.referenceStates[5, 0] = 0.
        self.referenceStates[6:9, 0] = v[:3]
        self.referenceStates[9:12, 0] = v[3:6]

        for i in range(1, self.n_steps + 1):
            dt = i * self.dt_mpc

            if v_ref[5] < 10e-3:
                self.referenceStates[0, i] = v_ref[0] * dt
                self.referenceStates[1, i] = v_ref[1] * dt
            else:
                self.referenceStates[0, i] = (v_ref[0] * np.sin(v_ref[5] * dt) + v_ref[1] *
                                              (np.cos(v_ref[5] * dt) - 1.)) / v_ref[5]
                self.referenceStates[1, i] = (v_ref[1] * np.sin(v_ref[5] * dt) - v_ref[0] *
                                              (np.cos(v_ref[5] * dt) - 1.)) / v_ref[5]

            self.referenceStates[:2, i] += self.referenceStates[:2, 0]

            # self.referenceStates[5, i] = rpy[2] + v_ref[5] * dt
            self.referenceStates[5, i] = v_ref[5] * dt

            self.referenceStates[6, i] = v_ref[0] * np.cos(v_ref[5] * dt) - v_ref[1] * np.sin(v_ref[5] * dt)
            self.referenceStates[7, i] = v_ref[0] * np.sin(v_ref[5] * dt) + v_ref[1] * np.cos(v_ref[5] * dt)

            self.referenceStates[11, i] = v_ref[5]

        # Update according to heightmap
        # result = self.compute_mean_surface(q[:3])

        rpy_map = np.zeros(3)
        rpy_map[0] = -np.arctan2(self.result[1], 1.)
        rpy_map[1] = -np.arctan2(self.result[0], 1.)

        self.referenceStates[3, 1:] = rpy_map[0] * np.cos(rpy[2]) - rpy_map[1] * np.sin(rpy[2])
        self.referenceStates[4, 1:] = rpy_map[0] * np.sin(rpy[2]) + rpy_map[1] * np.cos(rpy[2])

        v_max = 0.3  # rad.s-1
        self.referenceStates[9, 1] = max(min((self.referenceStates[3, 1] - rpy[0]) / self.dt_mpc, v_max), -v_max)
        self.referenceStates[9, 2:] = 0.
        self.referenceStates[10, 1] = max(min((self.referenceStates[4, 1] - rpy[1]) / self.dt_mpc, v_max), -v_max)
        self.referenceStates[10, 2:] = 0.

        for k in range(1, self.n_steps + 1):
            i, j = self.map.map_index(self.referenceStates[0, k], self.referenceStates[1, k])
            z = self.result[0] * self.map.x[i] + self.result[1] * self.map.y[j] + self.result[2]
            self.referenceStates[2, k] = z + self.h_ref
            if k == 1:
                self.surface_point = z

        v_max = 0.1  # m.s-1
        self.referenceStates[8, 1] = max(min((self.referenceStates[2, 1] - q[2]) / self.dt_mpc, v_max), -v_max)
        self.referenceStates[8, 2:] = (self.referenceStates[2, 2] - self.referenceStates[2, 1]) / self.dt_mpc

        if new_step:
            self.compute_configurations(q, v_ref, self.result)

    def compute_mean_surface(self, q):
        '''  Compute the surface equation to fit the heightmap, [a,b,c] such as ax + by -z +c = 0
        Args :
            - q (array 3x) : current [x,y,z] position in world frame 
        '''
        # Fit the map
        i_min, j_min = self.map.map_index(q[0] - self.FIT_SIZE_X, q[1] - self.FIT_SIZE_Y)
        i_max, j_max = self.map.map_index(q[0] + self.FIT_SIZE_X, q[1] + self.FIT_SIZE_Y)

        n_points = (i_max - i_min) * (j_max - j_min)
        A = np.zeros((n_points, 3))
        b = np.zeros(n_points)
        i_pb = 0
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                A[i_pb, :] = [self.map.x[i], self.map.y[j], 1.]
                b[i_pb] = self.map.zv[i, j]
                i_pb += 1

        return solve_least_square(np.array(A), np.array(b)).x

    def compute_configurations(self, q, v_ref, result):
        """
        Compute the surface equation to fit the heightmap, [a,b,c] such as ax + by -z +c = 0
        Args :
            - q (array 6x) : current [x,y,z, r, p, y] position in world frame
            - v_ref (array 6x) : cdesired velocity in world frame
        """
        for k, config in enumerate(self.configs):
            dt = self.T_step * k
            config[:2] = q[:2]
            if v_ref[5] < 10e-3:
                config[0] += v_ref[0] * dt
                config[1] += v_ref[1] * dt
            else:
                config[0] += (v_ref[0] * np.sin(v_ref[5] * dt) + v_ref[1] * (np.cos(v_ref[5] * dt) - 1.)) / v_ref[5]
                config[1] += (v_ref[1] * np.sin(v_ref[5] * dt) - v_ref[0] * (np.cos(v_ref[5] * dt) - 1.)) / v_ref[5]

            rpy_config = np.zeros(3)
            rpy_config[2] = q[5] + v_ref[5] * dt

            # Update according to heightmap
            i, j = self.map.map_index(config[0], config[1])
            config[2] = result[0] * self.map.x[i] + result[1] * self.map.y[j] + result[2] + self.h_ref

            rpy_map = np.zeros(3)
            rpy_map[0] = -np.arctan2(result[1], 1.)
            rpy_map[1] = -np.arctan2(result[0], 1.)

            rpy_config[0] = rpy_map[0] * np.cos(rpy_config[2]) - rpy_map[1] * np.sin(rpy_config[2])
            rpy_config[1] = rpy_map[0] * np.sin(rpy_config[2]) + rpy_map[1] * np.cos(rpy_config[2])
            quat = pin.Quaternion(pin.rpy.rpyToMatrix(rpy_config))
            config[3:7] = [quat.x, quat.y, quat.z, quat.w]

    def getReferenceStates(self):
        return self.referenceStates

    # def print_profile(self, output_file):
    #     ''' Print the profile computed with cProfile
    #     Args :
    #     - output_file (str) :  file name
    #     '''
    #     profileWrap.print_stats(output_file)

    #     return 0
