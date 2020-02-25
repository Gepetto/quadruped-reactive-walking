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

########################################################################
#                        Parameters definition                         #
########################################################################

# Time step
dt_mpc = 0.005
t = 0.0  # Time

# Simulation parameters
N_SIMULATION = 600  # number of time steps simulated

# Initialize the error for the simulation time
time_error = False

t_list = []

# Enable/Disable Gepetto viewer
enable_gepetto_viewer = False

# Create Joystick, ContactSequencer, FootstepPlanner, FootTrajectoryGenerator
# and MpcSolver objects
joystick, sequencer, fstep_planner, ftraj_gen, mpc, logger = utils.init_objects(dt_mpc, N_SIMULATION)

########################################################################
#                            Gepetto viewer                            #
########################################################################

solo = utils.init_viewer()

########################################################################
#                              PyBullet                                #
########################################################################

pyb_sim = utils.pybullet_simulator()

########################################################################
#                             Simulator                                #
########################################################################

myController = controller(q0, omega, t, int(N_SIMULATION))
mySafetyController = Safety_controller.controller_12dof()
myEmergencyStop = EmergencyStop_controller.controller_12dof()
myForceMonitor = ForceMonitor.ForceMonitor(mpc.footholds, pyb_sim.robotId, pyb_sim.planeId)

for k in range(int(N_SIMULATION)):

    if (k % 100) == 0:
        print("Iteration: ", k)

    joystick.update_v_ref(k)  # Update the reference velocity coming from the joystick
    mpc.update_v_ref(joystick)  # Retrieve reference velocity

    if k > 0:
        sequencer.updateSequence()  # Update contact sequence

    if k > 0:
        RPY = utils.rotationMatrixToEulerAngles(myController.robot.framePosition(
            myController.invdyn.data(), myController.model.getFrameId("base_link")).rotation)
        """settings.qu_m[2] = myController.robot.framePosition(
                myController.invdyn.data(), myController.model.getFrameId("base_link")).translation[2, 0]"""

        # RPY[1] *= -1  # Pitch is inversed

        mpc.q[0:2, 0] = np.array([0.0, 0.0])
        mpc.q[2] = myController.robot.com(myController.invdyn.data())[2]
        mpc.q[3:5, 0] = RPY[0:2]
        mpc.q[5, 0] = 0.0
        mpc.v = myController.vtsid[:6, 0:1]
        if k == 200:
            mpc.v[0, 0] += 0.1
        # settings.vu_m[4] *= -1  # Pitch is inversed

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
    # mpc.q[[2, 3, 4]] = mpc.q_next[[2, 3, 4]]  # coordinates in x, y, yaw are always 0 in local frame
    # mpc.v = mpc.v_next

    # Logging various stuff
    logger.call_log_functions(sequencer, fstep_planner, ftraj_gen, mpc, k)

    for i in range(1):

        time_start = time.time()

        ####################################################################
        #                 Data collection from PyBullet                    #
        ####################################################################

        jointStates = pyb.getJointStates(pyb_sim.robotId, pyb_sim.revoluteJointIndices)  # State of all joints
        baseState = pyb.getBasePositionAndOrientation(pyb_sim.robotId)  # Position and orientation of the trunk
        baseVel = pyb.getBaseVelocity(pyb_sim.robotId)  # Velocity of the trunk

        # Joints configuration and velocity vector for free-flyer + 12 actuators
        qmes12 = np.vstack((np.array([baseState[0]]).T, np.array([baseState[1]]).T,
                            np.array([[jointStates[i_joint][0] for i_joint in range(len(jointStates))]]).T))
        vmes12 = np.vstack((np.array([baseVel[0]]).T, np.array([baseVel[1]]).T,
                            np.array([[jointStates[i_joint][1] for i_joint in range(len(jointStates))]]).T))

        ####################################################################
        #                Select the appropriate controller 				   #
        #                               &								   #
        #               Load the joint torques into the robot			   #
        ####################################################################

        # If the limit bounds are reached, controller is switched to a pure derivative controller
        """if(myController.error):
            print("Safety bounds reached. Switch to a safety controller")
            myController = mySafetyController"""

        # If the simulation time is too long, controller is switched to a zero torques controller
        """time_error = time_error or (time.time()-time_start > 0.01)
        if (time_error):
            print("Computation time lasted to long. Switch to a zero torque control")
            myController = myEmergencyStop"""

        # Retrieve the joint torques from the appropriate controller
        jointTorques = myController.control(qmes12, vmes12, t, i+k, solo, mpc).reshape((12, 1))

        # Set control torque for all joints
        pyb.setJointMotorControlArray(pyb_sim.robotId, pyb_sim.revoluteJointIndices,
                                      controlMode=pyb.TORQUE_CONTROL, forces=jointTorques)

        # Compute one step of simulation
        # pyb.stepSimulation()

        # Time incrementation
        t += dt

        # Time spent to run this iteration of the loop
        time_spent = time.time() - time_start

        # Logging the time spent
        t_list.append(time_spent)

        # Refresh force monitoring for PyBullet
        # myForceMonitor.display_contact_forces()
        # time.sleep(0.001)


# Display graphs of the logger
logger.plot_graphs(dt_mpc, N_SIMULATION, myController)

# Plot TSID related graphs
plt.figure()
plt.title("Trajectory of the front right foot over time")
l_str = ["X", "Y", "Z"]
for i in range(3):
    plt.subplot(3, 1, 1*i+1)
    plt.plot(myController.f_pos_ref[1, :, i])
    plt.plot(myController.f_pos[1, :, i])
    plt.legend(["FR Foot ref pos along " + l_str[i], "FR foot pos along " + l_str[i]])

plt.figure()
plt.title("Velocity of the front right foot over time")
l_str = ["X", "Y", "Z"]
for i in range(3):
    plt.subplot(3, 1, 1*i+1)
    plt.plot(myController.f_vel_ref[1, :, i])
    plt.plot(myController.f_vel[1, :, i])
    plt.legend(["FR Foot ref vel along " + l_str[i], "FR foot vel along " + l_str[i]])
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
        plt.plot((0.2027) * np.ones((N_SIMULATION,)))
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
        plt.legend(["COo ref pos along " + l_str[0], "CoM pos along " + l_str[i-1]])


plt.show()

plt.figure(9)
plt.plot(t_list, 'k+')
plt.show()

quit()
