# coding: utf8

import time
import numpy as np
import matplotlib.pylab as plt
import Joystick

#########################################
#  Definition of parameters of the MPC  #
#########################################

# Position of the center of mass at the beginning
pos_CoM = np.array([[0.0], [0.0], [0.0]])

# Initial position of contacts (under shoulders)
settings.p_contacts = settings.shoulders.copy()

# Initial (x, y) positions of footholds
feet_target = np.array([[0.19, 0.19, -0.19, -0.19], [0.15005, -0.15005, 0.15005, -0.15005], [0.0, 0.0, 0.0, 0.0]])
p = np.array([[0.19, 0.19, -0.19, -0.19], [0.15005, -0.15005, 0.15005, -0.15005]])
goal_on_ground = np.array([[0.19, 0.19, -0.19, -0.19], [0.15005, -0.15005, 0.15005, -0.15005]])

# Initial (x, y) positions of footholds in the trunk frame
footholds_local = np.array([[0.19, 0.19, -0.19, -0.19], [0.15005, -0.15005, 0.15005, -0.15005]])

# Initial (x, y) target positions of footholds in the trunk frame
footholds_local_target = np.array([[0.19, 0.19, -0.19, -0.19], [0.15005, -0.15005, 0.15005, -0.15005]])

# Maximum height at which the robot should lift its feet during swing phase
# max_height_feet = 0.1

# Lock target positions of footholds before touchdown
t_lock_before_touchdown = 0.001

# Foot trajectory generator objects (one for each foot)
# ftgs = [Foot_trajectory_generator(max_height_feet, t_lock_before_touchdown) for i in range(4)]

# Number of loops
k_max_loop = 800

settings.h_ref = settings.qu_m[2, 0]

# Position of the trunk in the world frame
settings.q_w = (settings.qu_m).copy()

# Enable display with gepetto-gui
enable_gepetto_viewer = True

#
# MAIN LOOP
#

for k in range(100):

    #####################
    #   MPC FUNCTIONS   #
    #####################

    # Run MPC once every 20 iterations of TSID

    if k == 0:
        settings.qu_m = np.array([[0.0, 0.0, 0.235 - 0.01205385, 0.0, 0.0, 0.0]]).transpose()
        settings.vu_m = np.zeros((6, 1))

    ########################
    #  REFERENCE VELOCITY  #
    ########################

    # Create the joystick object
    if k == 0:
        joystick = Joystick.Joystick()

    # Update the reference velocity coming from the joystick
    joystick.update_v_ref(k)

    # Saving into settings
    settings.v_ref = joystick.v_ref

    # Get the reference velocity in global frame
    c, s = np.cos(settings.qu_m[5, 0]), np.sin(settings.qu_m[5, 0])
    R = np.array([[c, -s, 0., 0., 0., 0.], [s, c, 0., 0., 0., 0], [0., 0., 1.0, 0., 0., 0.],
                  [0., 0., 0., c, -s, 0.], [0., 0., 0., s, c, 0.], [0., 0., 0., 0., 0., 1.0]])
    settings.v_ref_world = np.dot(R, settings.v_ref)

    ######################
    #  CONTACT SEQUENCE  #
    ######################

    # Get contact sequence
    settings.t = settings.dt * k
    if k == 0:
        settings.S = footSequence(settings.t, settings.dt, settings.T_gait, settings.phases)
    else:
        settings.S = np.vstack((settings.S[1:, :], settings.S[0:1, :]))

    ########################
    #  FOOTHOLDS LOCATION  #
    ########################

    # Create the objects during the first iteration then updating in the following iterations
    if k == 0:
        fstep_planner = FootstepPlanner.FootstepPlanner(0.03, settings.shoulders, settings.dt)
        ftraj_gen = FootTrajectoryGenerator.FootTrajectoryGenerator(settings.shoulders, settings.dt)
    else:
        ftraj_gen.update_frame(settings.vu_m)

    # Update desired location of footsteps using the footsteps planner
    fstep_planner.update_footsteps_mpc(settings.v_ref, settings.vu_m, settings.t_stance,
                                       settings.S, settings.T_gait, settings.qu_m[2, 0])

    # Updating quantities expressed in world frame
    fstep_planner.update_world_frame(settings.q_w)

    # Update 3D desired feet pos using the trajectory generator
    ftraj_gen.update_desired_feet_pos(fstep_planner.footsteps, settings.S,
                                      settings.dt, settings.T_gait - settings.t_stance, settings.q_w)

    # Get number of feet in contact with the ground for each step of the gait sequence
    settings.n_contacts = np.sum(settings.S, axis=1).astype(int)

    #########
    #  MPC  #
    #########

    # Create the MPC solver object
    if (k == 0):
        mpc = MpcSolver.MpcSolver(settings.dt, settings.S, k_max_loop)

    ##########################
    #  REFERENCE TRAJECTORY  #
    ##########################

    # Get the reference trajectory over the prediction horizon
    mpc.getRefStatesDuringTrajectory(settings)

    #####################
    #  SOLVER MATRICES  #
    #####################

    # Retrieve data from FootstepPlanner and FootTrajectoryGenerator
    mpc.retrieve_data(fstep_planner, ftraj_gen)

    # Create the constraints matrices used by the QP solver
    # Minimize x^T.P.x + x^T.q with constraints A.X == b and G.X <= h
    mpc.create_constraints_matrices(settings, solo, k)

    # Create the weights matrices used by the QP solver
    # P and q in the cost x^T.P.x + x^T.q
    if k == 0:  # Weight matrices are always the same
        mpc.create_weight_matrices(settings)

    #################
    #  CALL SOLVER  #
    #################

    # Create an initial guess and call the solver to solve the QP problem
    mpc.call_solver(settings)

    #####################
    #  RETRIEVE RESULT  #
    #####################

    # Extract relevant information from the output of the QP solver
    mpc.retrieve_result(settings)

    if k == 240:
        debug = 1

    #########################
    # UPDATE WORLD POSITION #
    #########################

    # Variation of position in world frame using the linear speed in local frame
    c_yaw, s_yaw = np.cos(settings.q_w[5, 0]), np.sin(settings.q_w[5, 0])
    R = np.array([[c_yaw, -s_yaw, 0], [s_yaw, c_yaw, 0], [0, 0, 1]])
    settings.q_w[0:3, 0:1] += np.dot(R, mpc.vu[0:3, 0:1] * settings.dt)

    # Variation of orientation in world frame using the angular speed in local frame
    settings.q_w[3:6, 0] += mpc.vu[3:6, 0] * settings.dt

    #####################
    #  GEPETTO VIEWER   #
    #####################

    if enable_gepetto_viewer:

        # Display non-locked target footholds with green spheres (gepetto gui)
        fstep_planner.update_viewer(solo.viewer, (k == 0))

        # Display locked target footholds with red spheres (gepetto gui)
        # Display desired 3D position of feet with magenta spheres (gepetto gui)
        ftraj_gen.update_viewer(solo.viewer, (k == 0))

        # Display reference trajectory, predicted trajectory, desired contact forces, current velocity
        mpc.update_viewer(solo.viewer, (k == 0), settings)

        qu_pinocchio = solo.q0
        qu_pinocchio[0:3, 0:1] = settings.q_w[0:3, 0:1]
        # TODO: Check why orientation of q_w and qu are different
        #qu_pinocchio[3:7, 0:1] = getQuaternion(settings.q_w[3:6, 0:1])
        qu_pinocchio[3:7, 0:1] = utils.getQuaternion(mpc.qu[3:6, 0:1])

        # Refresh the gepetto viewer display
        solo.display(qu_pinocchio)
        # solo.viewer.gui.refresh()

    # Get measured position and velocity after one time step
    # settings.qu_m, settings.vu_m = low_pass_robot(qu, vu)
    settings.qu_m[[2, 3, 4]] = mpc.qu[[2, 3, 4]]  # coordinate in x, y, yaw is always 0 in local frame
    settings.vu_m = mpc.vu

    print(mpc.f_applied)
