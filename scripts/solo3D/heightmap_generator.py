import matplotlib.pyplot as plt
import os
import numpy as np

from solo3D.heightmap_tools import Heightmap
from solo3D.utils import getAllSurfacesDict_inner

from solo_rbprm.solo_abstract import Robot

from hpp.corbaserver.affordance.affordance import AffordanceTool
from hpp.corbaserver.rbprm.tools.surfaces_from_path import getAllSurfacesDict
from hpp.corbaserver.problem_solver import ProblemSolver
from hpp.gepetto import ViewerFactory
import quadruped_reactive_walking as qrw

# --------------------------------- PROBLEM DEFINITION ---------------------------------------------------------------
params = qrw.Params()

N_X = 400
N_Y = 90
X_BOUNDS = [-0.3, 2.5]
Y_BOUNDS = [-0.8, 0.8]

rom_names = ['solo_LFleg_rom', 'solo_RFleg_rom', 'solo_LHleg_rom', 'solo_RHleg_rom']
others = ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT']
LIMBS = ['solo_RHleg_rom', 'solo_LHleg_rom', 'solo_LFleg_rom', 'solo_RFleg_rom']
paths = [os.environ["INSTALL_HPP_DIR"] + "/solo-rbprm/com_inequalities/feet_quasi_flat/",
         os.environ["INSTALL_HPP_DIR"] + "/solo-rbprm/relative_effector_positions/"]

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
# --------------------------------- METHODS ---------------------------------------------------------------


def init_afftool():
    """
    Initialize the affordance tool and return the solo abstract rbprm builder, the surface
    dictionary and all the affordance points
    """
    robot = Robot()
    robot.setJointBounds("root_joint", [-5., 5., -5., 5., 0.241, 1.5])
    robot.boundSO3([-3.14, 3.14, -0.01, 0.01, -0.01, 0.01])
    robot.setFilter(LIMBS)
    for limb in LIMBS:
        robot.setAffordanceFilter(limb, ['Support'])
    ps = ProblemSolver(robot)
    vf = ViewerFactory(ps)
    afftool = AffordanceTool()
    afftool.setAffordanceConfig('Support', [0.5, 0.03, 0.00005])
    afftool.loadObstacleModel(os.environ["SOLO3D_ENV_DIR"] + params.environment_URDF, "environment", vf)
    ps.selectPathValidation("RbprmPathValidation", 0.05)

    return afftool


def plot_surface(points, ax, color_id=0, alpha=1.):
    """
    Plot a surface
    """
    xs = np.append(points[0, :], points[0, 0]).tolist()
    ys = np.append(points[1, :], points[1, 0]).tolist()
    zs = np.append(points[2, :], points[2, 0]).tolist()
    if color_id == -1:
        ax.plot(xs, ys, zs)
    else:
        ax.plot(xs, ys, zs, color=COLORS[color_id % len(COLORS)], alpha=alpha)


def draw_whole_scene(surface_dict, ax=None, title=None, color_id=5):
    """
    Plot all the potential surfaces
    """
    if ax is None:
        fig = plt.figure()
        if title is not None:
            fig.suptitle(title, fontsize=16)
        ax = fig.add_subplot(111, projection="3d")
    for key in surface_dict.keys():
        plot_surface(np.array(surface_dict[key][0]).T, ax, color_id)
    return ax

# --------------------------------- MAIN ---------------------------------------------------------------
if __name__ == "__main__":
    afftool = init_afftool()
    affordances = afftool.getAffordancePoints('Support')
    all_surfaces = getAllSurfacesDict(afftool)
    new_surfaces = getAllSurfacesDict_inner(all_surfaces, 0.03)

    heightmap = Heightmap(N_X, N_Y, X_BOUNDS, Y_BOUNDS)
    heightmap.build(affordances)
    heightmap.save_binary(os.environ["SOLO3D_ENV_DIR"] + params.environment_heightmap)
    # heightmap.save_pickle(os.environ["SOLO3D_ENV_DIR"] + params.environment_heightmap + ".pickle")

    heightmap.plot()
    ax = draw_whole_scene(all_surfaces)
    draw_whole_scene(new_surfaces, ax, color_id=0)
    plt.show(block=True)