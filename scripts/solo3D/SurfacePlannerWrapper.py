from solo3D.SurfacePlanner import SurfacePlanner

from multiprocessing import Process, Lock
from multiprocessing.sharedctypes import Value, Array
from ctypes import Structure, c_double
import ctypes

import libquadruped_reactive_walking as lqrw

from time import perf_counter as clock
import numpy as np

# TODO : Modify this, should not be defined here
N_VERTICES_MAX = 4
N_SURFACE_MAX = 10
N_SURFACE_CONFIG = 3
N_gait = 100
N_POTENTIAL_SURFACE = 5
N_FEET = 4
N_PHASE = 3


class SurfaceDataCtype(Structure):
    ''' ctype data structure for the shared memory between processes, for surfaces
    Ax <= b
    If normal as inequalities : A = [A , n , -n] , b = [b , d + eps, -d-eps]
    Normal as equality : A = [A , n] , b = [b , d]
    A =  inequality matrix, last line equality if normal as equality
    b =  inequality vector,
    vertices =  vertices of the surface  [[x1,y1,z1],
                                                      [x2,y2,z2], ...]

    on = Bool if the surface is used or not
    TODO : if more than 4 vertices, add a variable for the number of vertice to reshape the appropriate buffer
    '''
    nvertices = 4
    nrow = nvertices + 2
    _fields_ = [('A', ctypes.c_double * 3 * nrow), ('b', ctypes.c_double * nrow),
                ('vertices', ctypes.c_double * 3 * nvertices), ('on', ctypes.c_bool)]


class DataOutCtype(Structure):
    '''  ctype data structure for the shared memory between processes
    Data Out, list of potential and the selected surfaces given by the MIP
    Potential surfaces are used if the MIP has not converged
    '''
    _fields_ = [('potentialSurfaces', SurfaceDataCtype * N_POTENTIAL_SURFACE * N_FEET),
                ('selectedSurfaces', SurfaceDataCtype * N_FEET), ('all_feet', ctypes.c_double * 12 * N_PHASE),
                ('success', ctypes.c_bool)]


class DataInCtype(Structure):
    ''' ctype data structure for the shared memory between processes
    TODO : if more than 4 vertices, add a variable for the number of vertice to reshape the appropriate buffer
    '''
    _fields_ = [('gait', ctypes.c_int64 * 4 * N_gait), ('configs', ctypes.c_double * 7 * N_SURFACE_CONFIG),
                ('o_vref', ctypes.c_double * 6), ('contacts', ctypes.c_double * 12), ('iteration', ctypes.c_int64)]


class SurfacePlanner_Wrapper():
    ''' Wrapper for the class SurfacePlanner for the paralellisation
    '''

    def __init__(self, params):
        self.urdf = params.environment_URDF
        self.T_gait = params.T_gait
        self.shoulders = np.reshape(params.shoulders.tolist(), (3,4), order = "F")

        # TODO : Modify this
        # Usefull for 1st iteration of QP
        A = [[-1.0000000, 0.0000000, 0.0000000],
            [0.0000000, -1.0000000, 0.0000000],
            [0.0000000, 1.0000000, 0.0000000],
            [1.0000000, 0.0000000, 0.0000000],
            [0.0000000, 0.0000000, 1.0000000],
            [-0.0000000, -0.0000000, -1.0000000]]

        b = [1.3946447, 0.9646447, 0.9646447, 0.5346446, 0.0000, 0.0000]

        vertices = [[-1.3946447276978748, 0.9646446609406726, 0.0], [-1.3946447276978748, -0.9646446609406726, 0.0],
                    [0.5346445941834704, -0.9646446609406726, 0.0], [0.5346445941834704, 0.9646446609406726, 0.0]]

        self.floor_surface = lqrw.Surface(np.array(A), np.array(b), np.array(vertices))

        # Results from MIP
        # Potential and selected surfaces for QP planner
        self.potential_surfaces = lqrw.SurfaceVectorVector()
        self.selected_surfaces = lqrw.SurfaceVector()

        self.all_feet_pos = []

        # MIP status
        self.mip_iteration = 0
        self.mip_success = False
        self.first_iteration = True

        # MIP status synchronous, this is updated just after the run of SL1M,
        # but used in the controller only at the next flying phase
        self.mip_iteration_syn = 0
        self.mip_success_syn = False

        self.multiprocessing = params.enable_multiprocessing_mip
        if self.multiprocessing:  # Setup variables in the shared memory
            self.newData = Value('b', False)
            self.newResult = Value('b', True)
            self.running = Value('b', True)

            # Data Out :
            self.dataOut = Value(DataOutCtype)
            # Data IN :
            self.dataIn = Value(DataInCtype)

        else:
            self.surfacePlanner = SurfacePlanner(self.urdf, self.T_gait, self.shoulders, N_PHASE)
        
        # Store results to mimic multiprocessing behaviour with synchronous loop
        self.selected_surfaces_syn = lqrw.SurfaceVector()
        self.all_feet_pos_syn = []

    def run(self, configs, gait_in, current_contacts, o_v_ref):
        if self.multiprocessing:
            self.run_asynchronous(configs, gait_in, current_contacts, o_v_ref)
        else:
            self.run_synchronous(configs, gait_in, current_contacts, o_v_ref)

    def run_synchronous(self, configs, gait_in, current_contacts, o_v_ref):
        surfaces, surface_inequalities, surfaces_indices, all_feet_pos, success = self.surfacePlanner.run(
            configs, gait_in, current_contacts, o_v_ref)
        self.mip_iteration_syn += 1
        self.mip_success_syn = success

        # Retrieve potential data, usefull if solver not converged
        self.potential_surfaces = lqrw.SurfaceVectorVector()
        for foot, foot_surfaces in enumerate(surface_inequalities):
            list_surfaces = lqrw.SurfaceVector()
            for i, (S, s) in enumerate(foot_surfaces):
                list_surfaces.append(lqrw.Surface(S, s, surfaces[foot][i].T))
            self.potential_surfaces.append(list_surfaces)

        # Use directly the MIP surfaces computed
        # self.selected_surfaces = lqrw.SurfaceVector()
        # if success:
        #     for foot, foot_surfaces in enumerate(surface_inequalities):
        #         i = surfaces_indices[foot]
        #         S, s = foot_surfaces[i]
        #         self.selected_surfaces.append(lqrw.Surface(S, s, surfaces[foot][i].T))

        #     self.all_feet_pos = all_feet_pos.copy()

        # Mimic the multiprocessing behaviour, store the resuts and get them with update function
        self.selected_surfaces_syn = lqrw.SurfaceVector()
        if success:
            for foot, foot_surfaces in enumerate(surface_inequalities):
                i = surfaces_indices[foot]
                S, s = foot_surfaces[i]
                self.selected_surfaces_syn.append(lqrw.Surface(S, s, surfaces[foot][i].T))

            self.all_feet_pos_syn = all_feet_pos.copy()

    def run_asynchronous(self, configs, gait_in, current_contacts, o_v_ref):

        # If this is the first iteration, creation of the parallel process
        with self.dataIn.get_lock():
            if self.dataIn.iteration == 0:
                p = Process(target=self.create_MIP_asynchronous,
                            args=(self.newData, self.newResult, self.running, self.dataIn, self.dataOut))
                p.start()
        # Stacking data to send them to the parallel process
        self.compress_dataIn(configs, gait_in, current_contacts, o_v_ref)
        self.newData.value = True

    def create_MIP_asynchronous(self, newData, newResult, running, dataIn, dataOut):
        while running.value:
            # Checking if new data is available to trigger the asynchronous MPC
            if newData.value:
                # Set the shared variable to false to avoid re-trigering the asynchronous MPC
                newData.value = False

                configs, gait_in, o_v_ref, current_contacts = self.decompress_dataIn(dataIn)

                with self.dataIn.get_lock():
                    if self.dataIn.iteration == 0:
                        loop_planner = SurfacePlanner(self.urdf, self.T_gait, self.shoulders, N_PHASE)

                surfaces, surface_inequalities, surfaces_indices, all_feet_pos, success = loop_planner.run(
                    configs, gait_in, current_contacts, o_v_ref)

                with self.dataIn.get_lock():
                    self.dataIn.iteration += 1

                t1 = clock()
                self.compress_dataOut(surfaces, surface_inequalities, surfaces_indices, all_feet_pos, success)
                t2 = clock()
                # print("TIME COMPRESS DATA [ms] :  ", 1000 * (t2 - t1))

                # Set shared variable to true to signal that a new result is available
                newResult.value = True

    def compress_dataIn(self, configs, gait_in, current_contacts, o_v_ref):

        with self.dataIn.get_lock():

            for i, config in enumerate(configs):
                dataConfig = np.frombuffer(self.dataIn.configs[i])
                dataConfig[:] = config[:]

            gait = np.frombuffer(self.dataIn.gait).reshape((N_gait, 4))
            gait[:, :] = gait_in

            o_vref = np.frombuffer(self.dataIn.o_vref)
            o_vref[:] = o_v_ref[:, 0]

            contact = np.frombuffer(self.dataIn.contacts).reshape((3, 4))
            contact[:, :] = current_contacts[:, :]

    def decompress_dataIn(self, dataIn):

        with dataIn.get_lock():

            configs = [np.frombuffer(config) for config in dataIn.configs]

            gait = np.frombuffer(self.dataIn.gait).reshape((N_gait, 4))

            o_v_ref = np.frombuffer(self.dataIn.o_vref).reshape((6, 1))

            contacts = np.frombuffer(self.dataIn.contacts).reshape((3, 4))

        return configs, gait, o_v_ref, contacts

    def compress_dataOut(self, surfaces, surface_inequalities, surfaces_indices, all_feet_pos, success):
        # Modify this
        nvertices = 4
        nrow = nvertices + 2

        with self.dataOut.get_lock():
            # Compress potential surfaces :
            for foot, foot_surfaces in enumerate(surface_inequalities):
                i = 0
                for i, (S, s) in enumerate(foot_surfaces):
                    A = np.frombuffer(self.dataOut.potentialSurfaces[foot][i].A).reshape((nrow, 3))
                    b = np.frombuffer(self.dataOut.potentialSurfaces[foot][i].b)
                    vertices = np.frombuffer(self.dataOut.potentialSurfaces[foot][i].vertices).reshape((nvertices, 3))
                    A[:, :] = S[:, :]
                    b[:] = s[:]
                    vertices[:, :] = surfaces[foot][i].T[:, :]
                    self.dataOut.potentialSurfaces[foot][i].on = True

                for j in range(i + 1, N_POTENTIAL_SURFACE):
                    self.dataOut.potentialSurfaces[foot][j].on = False

            if success:
                self.dataOut.success = True
                # self.dataOut.all_feet_pos = all_feet_pos

                # Compress selected surfaces
                for foot, index in enumerate(surfaces_indices):
                    A = np.frombuffer(self.dataOut.selectedSurfaces[foot].A).reshape((nrow, 3))
                    b = np.frombuffer(self.dataOut.selectedSurfaces[foot].b)
                    vertices = np.frombuffer(self.dataOut.selectedSurfaces[foot].vertices).reshape((nvertices, 3))
                    A[:, :] = surface_inequalities[foot][index][0][:, :]
                    b[:] = surface_inequalities[foot][index][1][:]
                    vertices = surfaces[foot][index].T[:, :]
                    self.dataOut.selectedSurfaces[foot].on = True

            else:
                self.dataOut.success = False

    def update_latest_results(self):
        ''' Update latest results : 2 list 
        - potential surfaces : used if MIP has not converged
        - selected_surfaces : surfaces selected for the next phases
        '''
        if self.multiprocessing:
            with self.dataOut.get_lock():
                if self.newResult.value:
                    self.newResult.value = False

                    self.potential_surfaces = lqrw.SurfaceVectorVector()
                    for foot_surfaces in self.dataOut.potentialSurfaces:
                        list_surfaces = lqrw.SurfaceVector()
                        for s in foot_surfaces:
                            if s.on:
                                list_surfaces.append(lqrw.Surface(np.array(s.A), np.array(s.b), np.array(s.vertices)))
                        self.potential_surfaces.append(list_surfaces)

                    self.selected_surfaces = lqrw.SurfaceVector()

                    if self.dataOut.success:
                        self.mip_success = True
                        self.mip_iteration += 1

                        for s in self.dataOut.selectedSurfaces:
                            self.selected_surfaces.append(
                                lqrw.Surface(np.array(s.A), np.array(s.b), np.array(s.vertices)))

                        # self.all_feet_pos = self.dataOut.all_feet_pos.copy()
                        # for foot in self.all_feet_pos:
                        #     foot.pop(0)

                    else:
                        self.mip_success = False
                        self.mip_iteration += 1

                else:
                    # TODO : So far, only the convergence or not of the solver has been taken into account,
                    # What if the solver take too long ?
                    pass
        else:
            # Results have been already updated
            self.mip_success = self.mip_success_syn
            self.mip_iteration = self.mip_iteration_syn

            if self.mip_success:
                self.selected_surfaces = self.selected_surfaces_syn
                self.all_feet_pos = self.all_feet_pos_syn.copy()

    def stop_parallel_loop(self):
        """Stop the infinite loop in the parallel process to properly close the simulation
        """

        self.running.value = False

    # def print_profile(self, output_file):
    #     ''' Print the profile computed with cProfile
    #     Args :
    #     - output_file (str) :  file name
    #     '''
    #     profileWrap.print_stats(output_file)
