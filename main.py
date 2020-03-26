# coding: utf8

import numpy as np
import matplotlib.pylab as plt
import utils
import time

import pybullet as pyb
import pybullet_data
from TSID_Debug_controller_four_legs_fb_vel import controller, dt, q0, omega
import Safety_controller
import EmergencyStop_controller
import ForceMonitor
from IPython import embed
import os

########################################################################
#                        Parameters definition                         #
########################################################################

# Time step
dt_mpc = 0.02
t = 0.0  # Time

# Simulation parameters
N_SIMULATION = 10000  # number of time steps simulated

# Initialize the error for the simulation time
time_error = False

# Lists to log the duration of 1 iteration of the MPC/TSID
t_list_mpc = [0] * int(N_SIMULATION)
t_list_tsid = [0] * int(N_SIMULATION)
t_list_state = [0] * int(N_SIMULATION)
t_list_ft = [0] * int(N_SIMULATION)

# Enable/Disable Gepetto viewer
enable_gepetto_viewer = False

# Create Joystick, ContactSequencer, FootstepPlanner, FootTrajectoryGenerator
# and MpcSolver objects
joystick, sequencer, fstep_planner, ftraj_gen, mpc, logger, mpc_interface = utils.init_objects(dt_mpc, N_SIMULATION)

########################################################################
#                            Gepetto viewer                            #
########################################################################

# Initialisation of the Gepetto viewer
solo = utils.init_viewer()

########################################################################
#                              PyBullet                                #
########################################################################

# Initialisation of the PyBullet simulator
pyb_sim = utils.pybullet_simulator(dt=0.001)

# Force monitor to display contact forces in PyBullet with red lines
myForceMonitor = ForceMonitor.ForceMonitor(pyb_sim.robotId, pyb_sim.planeId)

########################################################################
#                             Simulator                                #
########################################################################

# Define the default controller as well as emergency and safety controller
myController = controller(q0, omega, t, int(N_SIMULATION))
mySafetyController = Safety_controller.controller_12dof()
myEmergencyStop = EmergencyStop_controller.controller_12dof()

for k in range(int(N_SIMULATION)):

    # Starting time of the whole iteration
    time_start_all = time.time()

    if (k % 1000) == 0:
        print("Iteration: ", k)

    ###################################
    #  Data collection from PyBullet  #
    ###################################

    # Retrieve data from the simulation
    pyb_sim.retrieve_pyb_data()

    ##########################
    #  PyBullet environment  #
    ##########################

    # Check the state of the robot to trigger events and update the camera
    pyb_sim.check_pyb_env(pyb_sim.qmes12)

    ########################
    # Update MPC interface #
    ########################

    # Call the mpc_interface that makes the interface between the simulation and the MPC/TSID
    mpc_interface.update(solo, pyb_sim.qmes12, pyb_sim.vmes12)

    ###############################
    #  Update reference velocity  #
    ###############################

    # Update the reference velocity coming from the joystick
    joystick.update_v_ref(k)

    ######################
    #  Update footsteps  #
    ######################

    # Update footsteps desired location once every 20 iterations of TSID
    if (k % 20) == 0:
        fstep_planner.update_fsteps(k, mpc_interface.l_feet, pyb_sim.vmes12[0:6, 0:1], joystick.v_ref,
                                    mpc_interface.lC[2, 0], mpc_interface.oMl, pyb_sim.ftps_Ids)

    #############
    #  Run MPC  #
    #############

    # Run MPC once every 20 iterations of TSID
    if (k % 20) == 0:

        ####################
        # Footstep planner #
        ####################

        # Get the reference trajectory over the prediction horizon
        fstep_planner.getRefStates((k/20), sequencer.T_gait, mpc_interface.lC, mpc_interface.abg,
                                   mpc_interface.lV, mpc_interface.lW, joystick.v_ref, h_ref=0.2027682)

        #########
        #  MPC  #
        #########

        time_mpc = time.time()

        # Run the MPC to get the reference forces and the next predicted state
        # Result is stored in mpc.f_applied, mpc.q_next, mpc.v_next
        mpc.run((k/20), sequencer.T_gait, sequencer.t_stance,
                mpc_interface.lC, mpc_interface.abg, mpc_interface.lV, mpc_interface.lW,
                mpc_interface.l_feet, fstep_planner.xref, fstep_planner.x0, joystick.v_ref,
                fstep_planner.fsteps)

        # Logging the time spent to run this iteration of the MPC
        t_list_mpc[k] = time.time() - time_mpc

        # Visualisation with gepetto viewer
        # if enable_gepetto_viewer:
        #     utils.display_all(solo, k, sequencer, fstep_planner, ftraj_gen, mpc)

        # Logging various stuff
        # logger.call_log_functions(sequencer, fstep_planner, ftraj_gen, mpc, k)

        # Output of the MPC
        f_applied = mpc.f_applied

    ####################
    # Inverse Dynamics #
    ####################

    time_tsid = time.time()

    #############################
    # Check if an error occured #
    #############################

    # If the limit bounds are reached, controller is switched to a pure derivative controller
    """if(myController.error):
        print("Safety bounds reached. Switch to a safety controller")
        myController = mySafetyController"""

    # If the simulation time is too long, controller is switched to a zero torques controller
    """time_error = time_error or (time.time()-time_start > 0.01)
    if (time_error):
        print("Computation time lasted to long. Switch to a zero torque control")
        myController = myEmergencyStop"""

    #####################################
    # Get torques with inverse dynamics #
    #####################################

    # Retrieve the joint torques from the current active controller
    jointTorques = myController.control(pyb_sim.qmes12, pyb_sim.vmes12, t, k, solo,
                                        sequencer, mpc_interface, joystick.v_ref, f_applied,
                                        fstep_planner.fsteps).reshape((12, 1))

    # Time incrementation
    t += dt

    # Logging the time spent to run this iteration of inverse dynamics
    t_list_tsid[k] = time.time() - time_tsid

    ######################
    # PyBullet iteration #
    ######################

    # Set control torque for all joints
    pyb.setJointMotorControlArray(pyb_sim.robotId, pyb_sim.revoluteJointIndices,
                                  controlMode=pyb.TORQUE_CONTROL, forces=jointTorques)

    # Apply perturbation
    """if k >= 50 and k < 100:
        pyb.applyExternalForce(pyb_sim.robotId, -1, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], pyb.LINK_FRAME)"""

    # Compute one step of simulation
    pyb.stepSimulation()

    # Refresh force monitoring for PyBullet
    # myForceMonitor.display_contact_forces()

    # Save PyBullet camera frame
    """if (k % 40) == 0:
        print(k)
        img = pyb.getCameraImage(width=1920, height=1080, renderer=pyb.ER_BULLET_HARDWARE_OPENGL)
        if k == 0:
            newpath = r'/tmp/recording'
            if not os.path.exists(newpath):
                os.makedirs(newpath)
        plt.imsave('/tmp/recording/frame_'+str(int(k/20))+'.png', img[2])"""

####################
# END OF MAIN LOOP #
####################

print("END")

# Display duration of MPC block and Inverse Dynamics block
plt.figure()
plt.plot(t_list_ft, 'k+')
plt.title("Time ft")

plt.figure()
plt.plot(t_list_mpc, 'k+')
plt.title("Time MPC")

plt.figure()
plt.plot(t_list_tsid, 'k+')
plt.title("Time TSID")
plt.show(block=True)

quit()


# Display graphs of the logger
logger.plot_graphs(dt_mpc, N_SIMULATION, myController)

# Plot TSID related graphs
plt.figure()
for i in range(4):
    plt.subplot(2, 2, 1*i+1)
    plt.title("Trajectory of the front right foot over time")
    l_str = ["X", "Y", "Z"]
    plt.plot(myController.f_pos_ref[i, :, 2], linewidth=2, marker='x')
    plt.plot(myController.f_pos[i, :, 2], linewidth=2, marker='x')
    plt.plot(myController.h_ref_feet, linewidth=2, marker='x')
    plt.legend(["FR Foot ref pos along " + l_str[2], "FR foot pos along " + l_str[2], "Zero altitude"])
    plt.xlim((0, 70))

plt.figure()
plt.title("Trajectory of the front right foot over time")
l_str = ["X", "Y", "Z"]
for i in range(3):
    plt.subplot(3, 1, 1*i+1)
    plt.plot(myController.f_pos_ref[1, :, i])
    plt.plot(myController.f_pos[1, :, i])
    if i == 2:
        plt.plot(myController.h_ref_feet)
        plt.legend(["FR Foot ref pos along " + l_str[i], "FR foot pos along " + l_str[i], "Zero altitude"])
    else:
        plt.legend(["FR Foot ref pos along " + l_str[i], "FR foot pos along " + l_str[i]])
    plt.xlim((20, 40))
plt.figure()
plt.title("Velocity of the front right foot over time")
l_str = ["X", "Y", "Z"]
for i in range(3):
    plt.subplot(3, 1, 1*i+1)
    plt.plot(myController.f_vel_ref[1, :, i])
    plt.plot(myController.f_vel[1, :, i])
    plt.legend(["FR Foot ref vel along " + l_str[i], "FR foot vel along " + l_str[i]])
    plt.xlim((20, 40))
    """plt.subplot(3, 3, 3*i+3)
    plt.plot(myController.f_acc_ref[1, :, i])
    plt.plot(myController.f_acc[1, :, i])
    plt.legend(["Ref acc along " + l_str[i], "Acc along " + l_str[i]])"""

plt.figure()
l_str = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(myController.b_pos[:, i])
    if i < 2:
        plt.plot(np.zeros((N_SIMULATION,)))
    else:
        plt.plot((0.2027682) * np.ones((N_SIMULATION,)))
    plt.legend([l_str[i] + "of base", l_str[i] + "reference of base"])

plt.figure()
l_str = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(myController.b_vel[:, i])
    plt.plot(np.zeros((N_SIMULATION,)))
    plt.legend(["Velocity of base along" + l_str[i], "Reference velocity of base along" + l_str[i]])

if hasattr(myController, 'com_pos_ref'):
    plt.figure()
    plt.title("Trajectory of the CoM over time")
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(myController.com_pos_ref[:, i], "b", linewidth=3)
        plt.plot(myController.com_pos[:, i], "r", linewidth=2)
        plt.legend(["CoM ref pos along " + l_str[0], "CoM pos along " + l_str[0]])


plt.show()

plt.figure(9)
plt.plot(t_list_tsid, 'k+')
plt.show()

quit()
