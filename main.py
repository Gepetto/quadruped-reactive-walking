# coding: utf8

import time
import numpy as np
import matplotlib.pylab as plt

import Joystick
import ContactSequencer
import FootstepPlanner
import FootTrajectoryGenerator
import MpcSolver

####################
# CREATING OBJECTS #
####################

# Time step
dt = 0.02

# Maximum number of loops
k_max_loop = 800

# Create Joystick object
joystick = Joystick.Joystick()

# Create contact sequencer object
sequencer = ContactSequencer.ContactSequencer(dt)

# Create footstep planner object
fstep_planner = FootstepPlanner.FootstepPlanner(dt)

# Create trajectory generator object
ftraj_gen = FootTrajectoryGenerator.FootTrajectoryGenerator(dt)

# Create MPC solver object
mpc = MpcSolver.MpcSolver(dt, sequencer, k_max_loop)

#############
# MAIN LOOP #
#############

for k in range(k_max_loop):

    joystick.update_v_ref(k)  # Update the reference velocity coming from the joystick
    mpc.update_v_ref(joystick)  # Retrieve reference velocity

    sequencer.updateSequence()  # Update contact sequence

    if k > 0:  # In local frame, contacts moves in the opposite direction of the base
        ftraj_gen.update_frame(mpc.v)  # Update contacts depending on the velocity of the base

    # Update desired location of footsteps using the footsteps planner
    fstep_planner.update_footsteps_mpc(sequencer, mpc)

    fstep_planner.update_world_frame(mpc.q_w)  # Updating quantities expressed in world frame

    # Update 3D desired feet pos using the trajectory generator
    ftraj_gen.update_desired_feet_pos(sequencer, fstep_planner, mpc)

    # Get the reference trajectory over the prediction horizon
    mpc.getRefStatesDuringTrajectory(sequencer)

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
