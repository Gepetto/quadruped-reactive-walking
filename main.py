# coding: utf8

import numpy as np
import matplotlib.pylab as plt
import utils

##################
# INITIALISATION #
##################

# Time step
dt = 0.02

# Maximum number of loops
k_max_loop = 800

# Enable/Disable Gepetto viewer
enable_gepetto_viewer = False
if enable_gepetto_viewer:
    solo = utils.init_viewer()

# Create Joystick, ContactSequencer, FootstepPlanner, FootTrajectoryGenerator
# and MpcSolver objects
joystick, sequencer, fstep_planner, ftraj_gen, mpc = utils.init_objects(dt, k_max_loop)

#############
# MAIN LOOP #
#############

for k in range(k_max_loop):

    joystick.update_v_ref(k)  # Update the reference velocity coming from the joystick
    mpc.update_v_ref(joystick)  # Retrieve reference velocity

    if k > 0:
        sequencer.updateSequence()  # Update contact sequence

    ###########################################
    # FOOTSTEP PLANNER & TRAJECTORY GENERATOR #
    ###########################################

    if k > 0:  # In local frame, contacts moves in the opposite direction of the base
        ftraj_gen.update_frame(mpc.v)  # Update contacts depending on the velocity of the base

    # Update desired location of footsteps using the footsteps planner
    fstep_planner.update_footsteps_mpc(sequencer, mpc)

    # Update 3D desired feet pos using the trajectory generator
    ftraj_gen.update_desired_feet_pos(sequencer, fstep_planner, mpc)

    ###################
    #  MPC FUNCTIONS  #
    ###################

    # Run the MPC to get the reference forces and the next predicted state
    # Result is stored in mpc.f_applied, mpc.q_next, mpc.v_next
    mpc.run(k, sequencer, fstep_planner, ftraj_gen)

    # Visualisation with gepetto viewer
    if enable_gepetto_viewer:
        utils.display_all(solo, k, sequencer, fstep_planner, ftraj_gen, mpc)

    # Get measured position and velocity after one time step (here perfect simulation)
    mpc.q[[2, 3, 4]] = mpc.q_next[[2, 3, 4]]  # coordinates in x, y, yaw are always 0 in local frame
    mpc.v = mpc.v_next
