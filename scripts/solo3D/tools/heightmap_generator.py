import matplotlib.pyplot as plt
import os

import sl1m.tools.plot_tools as plot
from sl1m.tools.plot_tools import draw_whole_scene
from solo3D.tools.heightmap_tools import Heightmap

from solo_rbprm.solo_abstract import Robot

from hpp.corbaserver.affordance.affordance import AffordanceTool
from hpp.corbaserver.rbprm.tools.surfaces_from_path import getAllSurfacesDict
from hpp.corbaserver.problem_solver import ProblemSolver
from hpp.gepetto import ViewerFactory
import libquadruped_reactive_walking as lqrw

# --------------------------------- PROBLEM DEFINITION ---------------------------------------------------------------
params = lqrw.Params()

N_X = 100
N_Y = 100
X_BOUNDS = [-0.5, 1.5]
Y_BOUNDS = [-1.5, 1.5]

rom_names = ['solo_LFleg_rom', 'solo_RFleg_rom', 'solo_LHleg_rom', 'solo_RHleg_rom']
others = ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT']
LIMBS = ['solo_RHleg_rom', 'solo_LHleg_rom', 'solo_LFleg_rom', 'solo_RFleg_rom']
paths = [
    os.environ["INSTALL_HPP_DIR"] + "/solo-rbprm/com_inequalities/feet_quasi_flat/",
    os.environ["INSTALL_HPP_DIR"] + "/solo-rbprm/relative_effector_positions/"
]
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
    afftool.loadObstacleModel(params.environment_URDF, "environment", vf)
    ps.selectPathValidation("RbprmPathValidation", 0.05)

    return afftool


# --------------------------------- MAIN ---------------------------------------------------------------
if __name__ == "__main__":
    afftool = init_afftool()
    affordances = afftool.getAffordancePoints('Support')
    all_surfaces = getAllSurfacesDict(afftool)

    heightmap = Heightmap(N_X, N_Y, X_BOUNDS, Y_BOUNDS)
    heightmap.build(affordances)
    # heightmap.save_pickle(ENV_HEIGHTMAP)
    heightmap.save_binary(params.environment_heightmap)

    ax_heightmap = plot.plot_heightmap(heightmap)
    draw_whole_scene(all_surfaces)
    plt.show(block = True)
