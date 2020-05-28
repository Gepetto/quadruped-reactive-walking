# coding: utf8

import numpy as np
import matplotlib.pylab as plt
import utils
import time

import pybullet as pyb
import pybullet_data
from TSID_Debug_controller_four_legs_fb_vel import controller, dt
import Safety_controller
import EmergencyStop_controller
import ForceMonitor
from IPython import embed
import os
import MPC_Wrapper
import pinocchio as pin
########################################################################
#                        Parameters definition                         #
########################################################################

# Time step
dt_mpc = 0.02
k_mpc = int(dt_mpc / dt)  # dt is dt_tsid, defined in the TSID controller script
t = 0.0  # Time
n_periods = 1  # Number of periods in the prediction horizon

# Simulation parameters
N_SIMULATION = 5000  # number of time steps simulated

# Initialize the error for the simulation time
time_error = False

# Lists to log the duration of 1 iteration of the MPC/TSID
t_list_tsid = [0] * int(N_SIMULATION)

# List to store the IDs of debug lines
ID_deb_lines = []

# Enable/Disable Gepetto viewer
enable_gepetto_viewer = False

# Create Joystick, ContactSequencer, FootstepPlanner, FootTrajectoryGenerator
# and MpcSolver objects
joystick, sequencer, fstep_planner, ftraj_gen, mpc, logger, mpc_interface = utils.init_objects(dt, dt_mpc, N_SIMULATION, k_mpc, n_periods)

# Enable/Disable multiprocessing (MPC running in a parallel process)
enable_multiprocessing = False
mpc_wrapper = MPC_Wrapper.MPC_Wrapper(dt_mpc, sequencer.S.shape[0], k_mpc, multiprocessing=enable_multiprocessing)

# Enable/Disable hybrid control
enable_hybrid_control = True

########################################################################
#                            Gepetto viewer                            #
########################################################################

# Initialisation of the Gepetto viewer
solo = utils.init_viewer(True)

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
myController = controller(int(N_SIMULATION), k_mpc, n_periods)
mySafetyController = Safety_controller.controller_12dof()
myEmergencyStop = EmergencyStop_controller.controller_12dof()

for k in range(int(N_SIMULATION)):

    if (k % 10000) == 0:
        print("Iteration: ", k)

    # Retrieve data from the simulation (position/orientation/velocity of the robot)
    # pyb_sim.retrieve_pyb_data()

    # Algorithm needs the velocity of the robot in world frame
    if k == 0:
        pyb_sim.retrieve_pyb_data()
        pyb_sim.qmes12[2, 0] = 0.2027682
    elif (k % k_mpc) == 0:
        # Using TSID future state as the robot state
        """pyb_sim.qmes12 = myController.qtsid.copy()
        pyb_sim.vmes12[0:3, 0:1] = mpc_interface.oMb.rotation @ myController.vtsid[0:3, 0:1]
        pyb_sim.vmes12[3:6, 0:1] = mpc_interface.oMb.rotation @ myController.vtsid[3:6, 0:1]
        pyb_sim.vmes12[7:, 0:1] = myController.vtsid[7:, 0:1].copy()"""

        # Using MPC future state as the robot state
        lMn = pin.SE3(pin.Quaternion(np.array([pyb.getQuaternionFromEuler(mpc_wrapper.mpc.q_next[3:6, 0])]).transpose()),
                      mpc_wrapper.mpc.q_next[0:3, 0] - mpc_wrapper.mpc.x0[0:3, 0])
        tmp = mpc_interface.oMl.rotation @ lMn.translation
        pyb_sim.qmes12[0:3, 0:1] = mpc_interface.oMl * lMn.translation # tmp[0:3, 0:1]
        #pyb_sim.qmes12[2, 0] = tmp[2, 0]
        pyb_sim.qmes12[3:7, 0:1] = utils.getQuaternion(np.array([utils.rotationMatrixToEulerAngles(mpc_interface.oMl.rotation @ lMn.rotation)]).transpose())
        pyb_sim.vmes12[0:3, 0] = mpc_interface.oMl.rotation @ mpc_wrapper.mpc.v_next[0:3, 0]
        pyb_sim.vmes12[3:6, 0] = mpc_interface.oMl.rotation @ mpc_wrapper.mpc.v_next[3:6, 0]

        if False: #k > 0:
            for i_foot in range(4):
                if fstep_planner.gait[0, i_foot+1] == 1:
                    footsteps_ideal[:, i_foot:(i_foot+1)] = mpc_interface.oMl.inverse() * ((mpc_interface.oMl * footsteps_ideal[:, i_foot:(i_foot+1)]) - tmp)

        if pyb_sim.vmes12[0, 0] > 0:
            deb = 1

    # Check the state of the robot to trigger events and update the simulator camera
    # pyb_sim.check_pyb_env(pyb_sim.qmes12)

    # Update the mpc_interface that makes the interface between the simulation and the MPC/TSID
    #pyb_sim.vmes12[0, 0] = 0.1
    if (k % k_mpc) == 0:
        if k == 0:
            mpc_interface.update(solo, pyb_sim.qmes12, pyb_sim.vmes12)

        if k > 0:
            # Get position, linear velocity and angular velocity in local frame
            mpc_interface.lC[:, 0] = np.array([[0.0, 0.0, mpc_wrapper.mpc.q_next[2, 0]]]).transpose()
            tmp_RPY = pin.rpy.matrixToRpy(lMn.rotation)
            tmp_lMn = pin.SE3(pin.utils.rotate('z', tmp_RPY[2, 0]),np.array([0.0, 0.0, 0.0]))
            mpc_interface.abg[0:2] = tmp_RPY[0:2]
            mpc_interface.abg[2] = 0.0
            mpc_interface.lV[:, 0:1] = (tmp_lMn.rotation.transpose() @ mpc_wrapper.mpc.v_next[0:3, 0]).transpose()
            mpc_interface.lW[:, 0:1] = (tmp_lMn.rotation.transpose() @ mpc_wrapper.mpc.v_next[3:6, 0]).transpose()

    if (k % k_mpc) == 0:
        if k == 0:
            footsteps_ideal = mpc_interface.l_feet.copy()
            footsteps_ideal[2, :] = 0.0
        elif False:
            for i_foot in range(4):
                if fstep_planner.gait[0, i_foot+1] == 0:
                    # footsteps_ideal[:, i_foot] = np.array(mpc_interface.oMl.inverse() * myController.footsteps[:, i_foot])      
                    """index = next((idx for idx, val in np.ndenumerate(fsteps_invdyn[:, 3*i_foot+1]) if ((not (val==0)) and (not np.isnan(val)))), [-1])[0]
                    pos_tmp = np.array([fsteps_invdyn[index, (1+i_foot*3):(4+i_foot*3)]])
                    footsteps_ideal[:, i_foot] = pos_tmp.copy()"""
                    footsteps_ideal[:, i_foot] = (fstep_planner.fsteps[1, (1+3*i_foot):(1+3*i_foot+3)]).copy()
    
            mpc_interface.l_feet = footsteps_ideal.copy()

    if (k % k_mpc) == 0:
        if k > 0:
            for i_foot in range(4):
                tmp_tr = mpc_wrapper.mpc.q_next[0:3, 0] - mpc_wrapper.mpc.x0[0:3, 0]
                tmp_tr[2] = 0.0
                tmp_lMn = pin.SE3(pin.utils.rotate('z', tmp_RPY[2, 0]), tmp_tr)
                tmp = tmp_lMn.inverse() * footsteps_ideal[:, i_foot]
                tmp[2, 0] = 0.0
                footsteps_ideal[:, i_foot:(i_foot+1)] = tmp.copy()
        footsteps_ideal[2, :] = 0.0
        mpc_interface.l_feet = footsteps_ideal.copy()

    # Update the reference velocity coming from the gamepad once every k_mpc iterations of TSID
    if (k % k_mpc) == 0:
        joystick.update_v_ref_predefined(k)

    if (k == 0):
        fstep_planner.update_fsteps(k, footsteps_ideal.copy(), np.vstack((mpc_interface.lV, mpc_interface.lW)), joystick.v_ref,
                                    mpc_interface.lC[2, 0], mpc_interface.oMl, pyb_sim.ftps_Ids, False)

    """elif (k > 0) and (k % 320 == 20):
        if joystick.gp.R1Button.value:
            gait_ID += 1
            if gait_ID == 1:
                fstep_planner.create_bounding()
            elif gait_ID == 2:
                fstep_planner.create_side_walking()
            else:
                gait_ID == 0
                fstep_planner.create_walking_trot()
            fstep_planner.update_fsteps(k, mpc_interface.l_feet, np.vstack((mpc_interface.lV, mpc_interface.lW)), joystick.v_ref,
                                    mpc_interface.lC[2, 0], mpc_interface.oMl, pyb_sim.ftps_Ids, False)
    """

    # Update footsteps desired location once every k_mpc iterations of TSID
    """for i_foot in range(4):
        if fstep_planner.fsteps
    mpc_interface.l_feet"""

    if (k % k_mpc) == 0:
        fsteps_invdyn = fstep_planner.fsteps.copy()
        gait_invdyn = fstep_planner.gait.copy()
        fstep_planner.update_fsteps(k+1, mpc_interface.l_feet, np.vstack((mpc_interface.lV, mpc_interface.lW)), joystick.v_ref,
                                    mpc_interface.lC[2, 0], mpc_interface.oMl, pyb_sim.ftps_Ids, joystick.reduced)

    if (k % k_mpc) == 0:
        if k > 0:
            for i_foot in range(4):
                if fstep_planner.gait[0, i_foot+1] == 0:
                    # footsteps_ideal[:, i_foot] = np.array(mpc_interface.oMl.inverse() * myController.footsteps[:, i_foot])      
                    """index = next((idx for idx, val in np.ndenumerate(fsteps_invdyn[:, 3*i_foot+1]) if ((not (val==0)) and (not np.isnan(val)))), [-1])[0]
                    pos_tmp = np.array([fsteps_invdyn[index, (1+i_foot*3):(4+i_foot*3)]])
                    footsteps_ideal[:, i_foot] = pos_tmp.copy()"""
                    footsteps_ideal[:, i_foot] = (fstep_planner.fsteps[1, (1+3*i_foot):(1+3*i_foot+3)]).copy()
            mpc_interface.l_feet = footsteps_ideal.copy()

    if joystick.v_ref[0, 0] > 0.01:
        deb = 1

    """if (k % k_mpc) == 0:
        mpc_interface.l_feet = footsteps_ideal.copy()
        ftp_flattened = footsteps_ideal.ravel(order='F')
        i_foot = 0
        while (fstep_planner.fsteps[i_foot, 0] != 0):
            fstep_planner.fsteps[i_foot, np.logical_not(np.isnan(fstep_planner.fsteps[i_foot, :]))] = np.hstack((np.array(fstep_planner.fsteps[i_foot, 0]),ftp_flattened[np.logical_not(np.isnan(fstep_planner.fsteps[i_foot, 1:]))]))
            i_foot += 1"""
    


    #######
    # MPC #
    #######

    # Run MPC once every k_mpc iterations of TSID
    if (k % k_mpc) == 0:

        # Debug lines
        if len(ID_deb_lines) == 0:
            for i_line in range(4):
                start = mpc_interface.oMl * np.array([[mpc_interface.l_shoulders[0, i_line], mpc_interface.l_shoulders[1, i_line], 0.01]]).transpose()
                end = mpc_interface.oMl * np.array([[mpc_interface.l_shoulders[0, i_line] + 0.4, mpc_interface.l_shoulders[1, i_line], 0.01]]).transpose()
                lineID = pyb.addUserDebugLine(np.array(start).ravel().tolist(), np.array(end).ravel().tolist(), lineColorRGB=[1.0, 0.0, 0.0], lineWidth=8)
                ID_deb_lines.append(lineID)
        else:
            for i_line in range(4):
                start = mpc_interface.oMl * np.array([[mpc_interface.l_shoulders[0, i_line], mpc_interface.l_shoulders[1, i_line], 0.01]]).transpose()
                end = mpc_interface.oMl * np.array([[mpc_interface.l_shoulders[0, i_line] + 0.4, mpc_interface.l_shoulders[1, i_line], 0.01]]).transpose()
                lineID = pyb.addUserDebugLine(np.array(start).ravel().tolist(), np.array(end).ravel().tolist(), lineColorRGB=[1.0, 0.0, 0.0], lineWidth=8,
                                              replaceItemUniqueId=ID_deb_lines[i_line])

        # Get the reference trajectory over the prediction horizon
        fstep_planner.getRefStates((k/k_mpc), sequencer.T_gait, mpc_interface.lC, mpc_interface.abg,
                                   mpc_interface.lV, mpc_interface.lW, joystick.v_ref, h_ref=0.2027682)


        if k > 0:
            if np.abs(mpc_wrapper.mpc.x_robot[7, 0] - mpc_interface.lV[1, 0]) > 0.00001:
                debug = 1


        # Output of the MPC
        f_applied = mpc_wrapper.get_latest_result(k)

        """if k > 0:
            print(mpc_wrapper.mpc.x_robot[0:6, 0] - fstep_planner.x0[0:6].ravel())
            print(mpc_wrapper.mpc.x_robot[6:12, 0] - fstep_planner.x0[6:12].ravel())
            print("###")"""
        # Run the MPC to get the reference forces and the next predicted state
        # Result is stored in mpc.f_applied, mpc.q_next, mpc.v_next
        mpc_wrapper.run_MPC(dt_mpc, sequencer.S.shape[0], k, sequencer.T_gait,
                            sequencer.t_stance, joystick, fstep_planner, mpc_interface)

    ####################
    # Inverse Dynamics #
    ####################

    time_tsid = time.time()

    # Check if an error occured
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

    myController.f_applied = f_applied
    # TSID needs the velocity of the robot in base frame
    """pyb_sim.vmes12[0:3, 0:1] = mpc_interface.oMb.rotation.transpose() @ pyb_sim.vmes12[0:3, 0:1]
    pyb_sim.vmes12[3:6, 0:1] = mpc_interface.oMb.rotation.transpose() @ pyb_sim.vmes12[3:6, 0:1]

    # Initial conditions
    if k == 0:
        myController.qtsid = pyb_sim.qmes12.copy()
        myController.vtsid = pyb_sim.vmes12.copy()

    # Retrieve the joint torques from the current active controller
    if enable_hybrid_control:
        jointTorques = myController.control(myController.qtsid, myController.vtsid, t, k, solo,
                                            sequencer, mpc_interface, joystick.v_ref, f_applied,
                                            fsteps_invdyn, gait_invdyn, pyb_sim.ftps_Ids_deb,
                                            enable_hybrid_control, pyb_sim.qmes12, pyb_sim.vmes12,
                                            mpc_wrapper.mpc.q_next, mpc_wrapper.mpc.v_next).reshape((12, 1))
    else:
        jointTorques = myController.control(pyb_sim.qmes12, pyb_sim.vmes12, t, k, solo,
                                            sequencer, mpc_interface, joystick.v_ref, f_applied,
                                            fsteps_invdyn, gait_invdyn, pyb_sim.ftps_Ids_deb).reshape((12, 1))"""
    # Time incrementation
    t += dt

    # Logging the time spent to run this iteration of inverse dynamics
    t_list_tsid[k] = time.time() - time_tsid

    ######################
    # PyBullet iteration #
    ######################

    # Set control torque for all joints
    # pyb.setJointMotorControlArray(pyb_sim.robotId, pyb_sim.revoluteJointIndices,
    #                              controlMode=pyb.TORQUE_CONTROL, forces=jointTorques)

    # Compute one step of simulation
    # pyb.stepSimulation()

    # Call logger object to log various parameters
    logger.call_log_functions(k, sequencer, joystick, fstep_planner, mpc_interface, mpc_wrapper, myController,
                              enable_multiprocessing, pyb_sim.robotId, pyb_sim.planeId, solo)

    # Refresh force monitoring for PyBullet
    # myForceMonitor.display_contact_forces()

    # Save PyBullet camera frame
    """step = 10
    if (k % step) == 0:
        if (k % 1000):
            print(k)
        img = pyb.getCameraImage(width=1920, height=1080, renderer=pyb.ER_BULLET_HARDWARE_OPENGL)
        if k == 0:
            newpath = r'/tmp/recording'
            if not os.path.exists(newpath):
                os.makedirs(newpath)
        if (int(k/step) < 10):
            plt.imsave('/tmp/recording/frame_000'+str(int(k/step))+'.png', img[2])
        elif int(k/step) < 100:
            plt.imsave('/tmp/recording/frame_00'+str(int(k/step))+'.png', img[2])
        elif int(k/step) < 1000:
            plt.imsave('/tmp/recording/frame_0'+str(int(k/step))+'.png', img[2])
        else:
            plt.imsave('/tmp/recording/frame_'+str(int(k/step))+'.png', img[2])"""

####################
# END OF MAIN LOOP #
####################

print("END")

logger.plot_graphs(enable_multiprocessing)

quit()

# Display duration of MPC block and Inverse Dynamics block
plt.figure()
plt.plot(t_list_tsid, 'k+')
plt.title("Time TSID")
plt.show(block=False)

"""plt.figure()
index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
index = [1, 3, 5, 2, 4, 6]
for i in range(6):
    plt.subplot(3, 2, index[i])
    plt.plot(log_feet[i%3, np.int(i/3), :], linewidth=2, marker='x')
    plt.plot(log_target[i%3, np.int(i/3), :], linewidth=2, marker='x')
    plt.legend(["Position", "Goal"])

plt.figure()
index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
index = [1, 3, 5, 2, 4, 6]
for i in range(6):
    plt.subplot(3, 2, index[i])
    plt.plot(log_vfeet[i%3, np.int(i/3), :], linewidth=2, marker='x')
    plt.plot(log_vtarget[i%3, np.int(i/3), :], linewidth=2, marker='x')
    plt.legend(["Velocity", "Goal"])

plt.figure()
index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
index = [1, 3, 5, 2, 4, 6]
for i in range(6):
    plt.subplot(3, 2, index[i])
    plt.plot(log_afeet[i%3, np.int(i/3), :], linewidth=2, marker='x')
    plt.plot(log_atarget[i%3, np.int(i/3), :], linewidth=2, marker='x')
    plt.legend(["Acc", "Goal"])
plt.show(block=True)"""
quit()

##########
# GRAPHS #
##########

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
