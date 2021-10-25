import pinocchio as pin
import numpy as np
import os
from time import perf_counter as clock

from sl1m.problem_definition import Problem
from sl1m.generic_solver import solve_MIP
# import matplotlib.pyplot as plt
# import sl1m.tools.plot_tools as plot

from solo_rbprm.solo_abstract import Robot as SoloAbstract

from hpp.corbaserver.affordance.affordance import AffordanceTool
from hpp.corbaserver.rbprm.tools.surfaces_from_path import getAllSurfacesDict
from hpp.corbaserver.problem_solver import ProblemSolver
from hpp.gepetto import ViewerFactory

# from solo3D.tools.ProfileWrapper import ProfileWrapper

# # Store the results from cprofile
# profileWrap = ProfileWrapper()

# --------------------------------- PROBLEM DEFINITION ---------------------------------------------------------------

paths = [
    os.environ["INSTALL_HPP_DIR"] + "/solo-rbprm/com_inequalities/feet_quasi_flat/",
    os.environ["INSTALL_HPP_DIR"] + "/solo-rbprm/relative_effector_positions/"
]
limbs = ['FLleg', 'FRleg', 'HLleg', 'HRleg']
others = ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT']
rom_names = ['solo_LFleg_rom', 'solo_RFleg_rom', 'solo_LHleg_rom', 'solo_RHleg_rom']
N_PHASE = 3

class SurfacePlanner:
    """
    Choose the next surface to use by solving a MIP problem
    """

    def __init__(self, environment_URDF, T_gait, shoulders, n_phase):
        """
        Initialize the affordance tool and save the solo abstract rbprm builder, and surface dictionary
        """
        self.T_gait = T_gait
        self.shoulders = shoulders
        self.n_phase = n_phase

        self.solo_abstract = SoloAbstract()
        self.solo_abstract.setJointBounds("root_joint", [-5., 5., -5., 5., 0.241, 1.5])
        self.solo_abstract.boundSO3([-3.14, 3.14, -0.01, 0.01, -0.01, 0.01])
        self.solo_abstract.setFilter(rom_names)
        for limb in rom_names:
            self.solo_abstract.setAffordanceFilter(limb, ['Support'])
        self.ps = ProblemSolver(self.solo_abstract)
        self.vf = ViewerFactory(self.ps)
        self.afftool = AffordanceTool()
        self.afftool.setAffordanceConfig('Support', [0.5, 0.03, 0.00005])

        self.afftool.loadObstacleModel(environment_URDF, "environment", self.vf, reduceSizes=[0.05, 0.0, 0.])
        self.ps.selectPathValidation("RbprmPathValidation", 0.05)

        self.all_surfaces = getAllSurfacesDict(self.afftool)

        self.potential_surfaces = []

        self.pb = Problem(limb_names=limbs, other_names=others, constraint_paths=paths)

    def compute_gait(self, gait_in):
        """
        Get a gait matrix with only one line per phase
        :param gait_in: gait matrix with several line per phase
        :return: gait matrix
        """
        gait = [gait_in[0, :]]
        # for i in range(1, gait_in.shape[0] - 1):
        #     if (gait_in[i, :] != gait[-1]).any():
        #         gait.append(gait_in[i, :])

        # TODO only works if we the last phase is not a flying phase
        # gait.pop(-1)
        # gait = np.roll(gait, -2, axis=0)

        for i in range(1, gait_in.shape[0] - 1):
            new_phase = True
            for row in gait:
                if (gait_in[i, :] == row).any():
                    new_phase = False

            if new_phase:
                gait.append(gait_in[i, :])

        gait = np.roll(gait, -2, axis=0)


        return gait

    def compute_step_length(self, o_v_ref):
        """
        Compute the step_length used for the cost
        :param o_v_ref: desired velocity
        :return: desired step_length
        """
        # TODO: Divide by number of phases in gait
        # step_length = o_v_ref * self.T_gait/4
        step_length = o_v_ref * self.T_gait / 2

        return np.array([step_length[i][0] for i in range(2)])

    def compute_effector_positions(self, configs):
        """
        Compute the step_length used for the cost
        :param o_v_ref: desired velocity
        :return: desired step_length
        """
        effector_positions = np.zeros((4, self.pb.n_phases, 2))
        for phase in self.pb.phaseData:
            for foot in phase.moving:
                rpy = pin.rpy.matrixToRpy(pin.Quaternion(configs[phase.id][3:7]).toRotationMatrix())
                shoulders = np.zeros(2)
                shoulders[0] = self.shoulders[0, foot] * np.cos(rpy[2]) - self.shoulders[1, foot] * np.sin(rpy[2])
                shoulders[1] = self.shoulders[0, foot] * np.sin(rpy[2]) + self.shoulders[1, foot] * np.cos(rpy[2])
                effector_positions[foot][phase.id] = np.array(configs[phase.id][:2] + shoulders)

        return effector_positions

    def get_potential_surfaces(self, configs, gait):
        """
        Get the rotation matrix and surface condidates for each configuration in configs
        :param configs: a list of successive configurations of the robot
        :param gait: a gait matrix
        :return: a list of surface candidates
        """
        surfaces_list = []
        empty_list = False
        for id, config in enumerate(configs):
            stance_feet = np.nonzero(gait[id % len(gait)] == 1)[0]
            previous_swing_feet = np.nonzero(gait[(id - 1) % len(gait)] == 0)[0]
            moving_feet = stance_feet[np.in1d(stance_feet, previous_swing_feet, assume_unique=True)]
            roms = np.array(rom_names)[moving_feet]

            foot_surfaces = []
            for rom in roms:
                surfaces = []
                surfaces_names = self.solo_abstract.clientRbprm.rbprm.getCollidingObstacleAtConfig(
                    config.tolist(), rom)
                for name in surfaces_names:
                    surfaces.append(self.all_surfaces[name][0])

                if not len(surfaces_names):
                    empty_list = True

                # Sort and then convert to array
                surfaces = sorted(surfaces)
                surfaces_array = []
                for surface in surfaces:
                    surfaces_array.append(np.array(surface).T)

                # Add to surfaces list
                foot_surfaces.append(surfaces_array)
            surfaces_list.append(foot_surfaces)

        return surfaces_list, empty_list

    def retrieve_surfaces(self, surfaces, indices=None):
        """
        Get the surface vertices,  inequalities and selected surface indices if need be
        """
        vertices = []
        surfaces_inequalities = []
        if indices is not None:
            surface_indices = []
        else:
            surface_indices = None
        first_phase_i = 0
        second_phase_i = 0
        for foot in range(4):
            if foot in self.pb.phaseData[0].moving:
                vertices.append(surfaces[0][first_phase_i])
                surfaces_inequalities.append(self.pb.phaseData[0].S[first_phase_i])
                if indices is not None:
                    surface_indices.append(indices[0][first_phase_i])
                first_phase_i += 1
            elif foot in self.pb.phaseData[1].moving:
                vertices.append(surfaces[1][second_phase_i])
                surfaces_inequalities.append(self.pb.phaseData[1].S[second_phase_i])
                if indices is not None:
                    surface_indices.append(indices[1][second_phase_i])
                second_phase_i += 1
            else:
                print("Error : the foot is not moving in any of the first two phases")

        return vertices, surfaces_inequalities, surface_indices

    def run(self, configs, gait_in, current_contacts, o_v_ref):
        """
        Select the nex surfaces to use
        :param xref: successive states
        :param gait: a gait matrix
        :param current_contacts: the initial_contacts to use in the computation
        :param o_v_ref: the desired velocity for the cost
        :return: the selected surfaces for the first phase
        """
        t0 = clock()

        R = [pin.XYZQUATToSE3(np.array(config)).rotation for config in configs]

        gait = self.compute_gait(gait_in)

        step_length = self.compute_step_length(o_v_ref)

        surfaces, empty_list = self.get_potential_surfaces(configs, gait)

        initial_contacts = [current_contacts[:, i].tolist() for i in range(4)]

        self.pb.generate_problem(R, surfaces, gait, initial_contacts, c0=None, com=False)

        # from IPython import embed
        # embed()

        if empty_list:
            print("Surface planner: one step has no potential surface to use.")
            vertices, inequalities, indices = self.retrieve_surfaces(surfaces)
            return vertices, inequalities, indices, None, False

        # effector_positions = self.compute_effector_positions(configs)
        # costs = {"step_size": [1.0, step_length], "effector_positions": [10.0, effector_positions]}
        costs = {"step_size": [1.0, step_length]}
        pb_data = solve_MIP(self.pb, costs=costs, com=False)

        if pb_data.success:
            surface_indices = pb_data.surface_indices

            selected_surfaces = []
            for foot, index in enumerate(surface_indices[0]):
                selected_surfaces.append(surfaces[0][foot][index])

            t1 = clock()
            if 1000. * (t1 - t0) > 150.:
                print("Run took ", 1000. * (t1 - t0))

            # import matplotlib.pyplot as plt
            # import sl1m.tools.plot_tools as plot

            # ax = plot.draw_whole_scene(self.all_surfaces)
            # plot.plot_planner_result(pb_data.all_feet_pos, step_size=step_length, ax=ax, show=True)

            vertices, inequalities, indices = self.retrieve_surfaces(surfaces, surface_indices)

            return vertices, inequalities, indices, pb_data.all_feet_pos, True

        else:
            # ax = plot.draw_whole_scene(self.all_surfaces)
            # plot.plot_initial_contacts(initial_contacts, ax=ax)
            # ax.scatter([c[0] for c in configs], [c[1] for c in configs], [c[2] for c in configs], marker='o', linewidth=5)
            # ax.plot([c[0] for c in configs], [c[1] for c in configs], [c[2] for c in configs])

            # plt.show()

            print("The MIP problem did NOT converge")
            # TODO what if the problem did not converge ???

            vertices, inequalities, indices = self.retrieve_surfaces(surfaces)

            return vertices, inequalities, indices, None, False

    # def print_profile(self, output_file):
    #     ''' Print the profile computed with cProfile
    #     Args :
    #     - output_file (str) :  file name
    #     '''
    #     profileWrap.print_stats(output_file)
