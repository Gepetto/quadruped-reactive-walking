import pinocchio as pin
import numpy as np
import os
import copy

from sl1m.problem_definition import Problem
from sl1m.generic_solver import solve_MIP

from solo_rbprm.solo_abstract import Robot as SoloAbstract

from hpp.corbaserver.affordance.affordance import AffordanceTool
from hpp.corbaserver.rbprm.tools.surfaces_from_path import getAllSurfacesDict
from solo3D.tools.utils import getAllSurfacesDict_inner
from hpp.corbaserver.problem_solver import ProblemSolver
from hpp.gepetto import ViewerFactory

# --------------------------------- PROBLEM DEFINITION ---------------------------------------------------------------

paths = [os.environ["INSTALL_HPP_DIR"] + "/solo-rbprm/com_inequalities/feet_quasi_flat/",
         os.environ["INSTALL_HPP_DIR"] + "/solo-rbprm/relative_effector_positions/"]
limbs = ['FLleg', 'FRleg', 'HLleg', 'HRleg']
others = ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT']
rom_names = ['solo_LFleg_rom', 'solo_RFleg_rom', 'solo_LHleg_rom', 'solo_RHleg_rom']


class SurfacePlanner:
    """
    Choose the next surface to use by solving a MIP problem
    """

    def __init__(self, params):
        """
        Initialize the affordance tool and save the solo abstract rbprm builder, and surface dictionary
        """
        self.plot = False

        self.use_heuristique = params.use_heuristic
        self.step_duration = params.T_gait/2
        self.shoulders = np.reshape(params.footsteps_under_shoulders, (3, 4), order="F")

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

        self.afftool.loadObstacleModel(os.environ["SOLO3D_ENV_DIR"] + params.environment_URDF, "environment", self.vf)
        self.ps.selectPathValidation("RbprmPathValidation", 0.05)

        self.all_surfaces = getAllSurfacesDict_inner(getAllSurfacesDict(self.afftool), margin=0.02)

        self.potential_surfaces = []

        self.pb = Problem(limb_names=limbs, other_names=others, constraint_paths=paths)

    def compute_gait(self, gait_in):
        """
        Get a gait matrix with only one line per phase
        :param gait_in: gait matrix with several line per phase
        :return: gait matrix
        """
        gait = [gait_in[0, :]]

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
        step_length = o_v_ref * self.step_duration
        return np.array([step_length[i] for i in range(2)])

    def compute_effector_positions(self, configs, h_v_ref):
        """
        Compute the desired effector positions in 2D
        :param configs the list of configurations
        :param h_v_ref, Array (x3) the desired velocity in horizontal frame
        :param yaw, Array (x3) yaw of the horizontal frame wrt the world frame
        """
        effector_positions = [[] for _ in range(4)]
        for phase in self.pb.phaseData:
            for foot in range(4):
                if foot in phase.moving:
                    rpy = pin.rpy.matrixToRpy(pin.Quaternion(configs[phase.id][3:7].copy()).toRotationMatrix())
                    hRb = pin.rpy.rpyToMatrix(np.array([rpy[0], rpy[1], 0.]))
                    wRh = pin.rpy.rpyToMatrix(np.array([0., 0., rpy[2]]))
                    heuristic = wRh @ (0.5 * self.step_duration * copy.deepcopy(h_v_ref)) + wRh @ hRb @ copy.deepcopy(self.shoulders[:, foot])
                    effector_positions[foot].append(np.array(configs[phase.id][:2] + heuristic[:2]))
                else:
                    effector_positions[foot].append(None)

        return effector_positions

    def compute_shoulder_positions(self, configs):
        """
        Compute the shoulder positions 
        :param configs the list of configurations
        """
        shoulder_positions = np.zeros((4, self.pb.n_phases, 3))
        for phase in self.pb.phaseData:
            for foot in phase.moving:
                R = pin.Quaternion(configs[phase.id][3:7]).toRotationMatrix()
                shoulder_positions[foot][phase.id] = R @ self.shoulders[:, foot] + configs[phase.id][:3]
        return shoulder_positions

    def get_potential_surfaces(self, configs, gait):
        """
        Get the rotation matrix and surface condidates for each configuration in configs
        :param configs: a list of successive configurations of the robot
        :param gait: a gait matrix
        :return: a list of surface candidates and a boolean set to false if one foot hase no potential surface
        """
        surfaces_list = []
        empty_list = False
        for id, config in enumerate(configs):
            stance_feet = np.nonzero(gait[id % len(gait)] == 1)[0]
            previous_swing_feet = np.nonzero(gait[(id-1) % len(gait)] == 0)[0]
            moving_feet = stance_feet[np.in1d(stance_feet, previous_swing_feet, assume_unique=True)]
            roms = np.array(rom_names)[moving_feet]

            foot_surfaces = []
            for rom in roms:
                surfaces = []
                surfaces_names = self.solo_abstract.clientRbprm.rbprm.getCollidingObstacleAtConfig(config.tolist(), rom)
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
        Get all the potential surfaces as vertices and as inequalities for the first step of each foot
        and get the selected surface indices if need be
        return vertices a list of all potential surfaces vertices for each foot's first step
        return inequalities a list of all potential surfaces inequalities for each foot's  first step
        return indices the selected surface indices for each foot's first step
        """
        vertices = []
        inequalities = []
        selected_surfaces_indices = []

        first_phase_i = 0
        second_phase_i = 0
        for foot in range(4):
            if foot in self.pb.phaseData[0].moving:
                vertices.append(surfaces[0][first_phase_i])
                inequalities.append(self.pb.phaseData[0].S[first_phase_i])
                if indices is not None:
                    selected_surfaces_indices.append(indices[0][first_phase_i])
                first_phase_i += 1
            elif foot in self.pb.phaseData[1].moving:
                vertices.append(surfaces[1][second_phase_i])
                inequalities.append(self.pb.phaseData[1].S[second_phase_i])
                if indices is not None:
                    selected_surfaces_indices.append(indices[1][second_phase_i])
                second_phase_i += 1
            else:
                print("Error : the foot is not moving in any of the first two phases")

        return vertices, inequalities, selected_surfaces_indices

    def run(self, configs, gait_in, current_contacts, h_v_ref):
        """
        Select the next surfaces to use
        :param configs: successive states
        :param gait_in: a gait matrix
        :param current_contacts: the initial_contacts to use in the computation
        :param h_v: Array (x3) the current velocity for the cost, in horizontal frame
        :param h_v_ref: Array (x3) the desired velocity for the cost, in horizontal frame
        :return: the selected surfaces for the first phase
        """
        R = [pin.XYZQUATToSE3(np.array(config)).rotation for config in configs]
        gait = self.compute_gait(gait_in)
        initial_contacts = [current_contacts[:, i].tolist() for i in range(4)]

        surfaces, empty_list = self.get_potential_surfaces(configs, gait)
        if empty_list:
            print("Surface planner: one step has no potential surface to use.")
            vertices, inequalities, _ = self.retrieve_surfaces(surfaces)
            return vertices, inequalities, None, None, False

        self.pb.generate_problem(R, surfaces, gait, initial_contacts, c0=None, com=False)

        shoulder_positions = self.compute_shoulder_positions(configs)
        if self.use_heuristique:
            effector_positions = self.compute_effector_positions(configs, h_v_ref)
            costs = {"effector_positions_xy": [1., effector_positions], "effector_positions_3D": [0.1, shoulder_positions]}
        else:
            step_length = self.compute_step_length(h_v_ref[:2])
            costs = {"step_length": [1.0, step_length], "effector_positions_3D": [0.1, shoulder_positions]}
        result = solve_MIP(self.pb, costs=costs, com=False)

        if result.success:
            if self.plot:
                import matplotlib.pyplot as plt
                import sl1m.tools.plot_tools as plot
                ax = plot.draw_whole_scene(self.all_surfaces)
                plot.plot_planner_result(result.all_feet_pos, effector_positions=effector_positions, coms=configs, ax=ax, show=True)

            vertices, inequalities, indices = self.retrieve_surfaces(surfaces, result.surface_indices)
            return vertices, inequalities, indices, result.all_feet_pos, True

        if self.plot:
            import matplotlib.pyplot as plt
            import sl1m.tools.plot_tools as plot
            ax = plot.draw_whole_scene(self.all_surfaces)
            plot.plot_initial_contacts(initial_contacts, ax=ax)
            ax.scatter([c[0] for c in configs], [c[1] for c in configs], [c[2] for c in configs], marker='o', linewidth=5)
            ax.plot([c[0] for c in configs], [c[1] for c in configs], [c[2] for c in configs])
            plt.show()

        print("The MIP problem did NOT converge")
        vertices, inequalities, _ = self.retrieve_surfaces(surfaces)
        return vertices, inequalities, None, None, False
