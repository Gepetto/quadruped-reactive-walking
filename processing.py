# coding: utf8

import numpy as np
import pybullet as pyb


def process_states(solo, k, k_mpc, pyb_sim, interface, joystick, tsid_controller):
    """Update states by retrieving information from the simulation and the gamepad

    Args:
        solo (object): Pinocchio wrapper for the quadruped
        k (int): Number of inv dynamics iterations since the start of the simulation
        k_mpc (int): Number of inv dynamics iterations for one iteration of the MPC
        pyb_sim (object): PyBullet simulation
        interface (object): Interface object of the control loop
        joystick (object): Interface with the gamepad
        tsid_controller (object): Inverse dynamics controller
    """

    # Algorithm needs the velocity of the robot in world frame
    if k == 0:
        # Retrieve data from the simulation (position/orientation/velocity of the robot)
        pyb_sim.retrieve_pyb_data()
        pyb_sim.qmes12[2, 0] = 0.2027682
    elif (k % k_mpc) == 0:
        # Using TSID future state as the robot state
        pyb_sim.qmes12 = tsid_controller.qtsid.copy()
        pyb_sim.vmes12[0:3, 0:1] = interface.oMb.rotation @ tsid_controller.vtsid[0:3, 0:1]
        pyb_sim.vmes12[3:6, 0:1] = interface.oMb.rotation @ tsid_controller.vtsid[3:6, 0:1]
        pyb_sim.vmes12[7:, 0:1] = tsid_controller.vtsid[7:, 0:1].copy()

        # Using MPC future state as the robot state
        """lMn = pin.SE3(pin.Quaternion(np.array([pyb.getQuaternionFromEuler(mpc_wrapper.mpc.q_next[3:6, 0])]).transpose()),
                      mpc_wrapper.mpc.q_next[0:3, 0] - mpc_wrapper.mpc.x0[0:3, 0])
        tmp = interface.oMl.rotation @ lMn.translation
        pyb_sim.qmes12[0:3, 0:1] = interface.oMl * lMn.translation # tmp[0:3, 0:1]
        #pyb_sim.qmes12[2, 0] = tmp[2, 0]
        pyb_sim.qmes12[3:7, 0:1] = utils.getQuaternion(np.array([utils.rotationMatrixToEulerAngles(interface.oMl.rotation @ lMn.rotation)]).transpose())
        pyb_sim.vmes12[0:3, 0] = interface.oMl.rotation @ mpc_wrapper.mpc.v_next[0:3, 0]
        pyb_sim.vmes12[3:6, 0] = interface.oMl.rotation @ mpc_wrapper.mpc.v_next[3:6, 0]

        if False: #k > 0:
            for i_foot in range(4):
                if fstep_planner.gait[0, i_foot+1] == 1:
                    footsteps_ideal[:, i_foot:(i_foot+1)] = interface.oMl.inverse() * ((interface.oMl * footsteps_ideal[:, i_foot:(i_foot+1)]) - tmp)

        if pyb_sim.vmes12[0, 0] > 0:
            deb = 1"""

    # Check the state of the robot to trigger events and update the simulator camera
    # pyb_sim.check_pyb_env(pyb_sim.qmes12)

    # Update the interface that makes the interface between the simulation and the MPC/TSID
    interface.update(solo, pyb_sim.qmes12, pyb_sim.vmes12)

    # Update the reference velocity coming from the gamepad once every k_mpc iterations of TSID
    if (k % k_mpc) == 0:
        joystick.update_v_ref(k, predefined=True)

    return 0


def process_footsteps_planner(k, k_mpc, pyb_sim, interface, joystick, fstep_planner):
    """Update desired location of footsteps depending on the current state of the robot
    and the reference velocity

    Args:
        k (int): Number of inv dynamics iterations since the start of the simulation
        k_mpc (int): Number of inv dynamics iterations for one iteration of the MPC
        pyb_sim (object): PyBullet simulation
        interface (object): Interface object of the control loop
        joystick (object): Interface with the gamepad
        fstep_planner (object): Footsteps planner object
    """

    # Initialization of the desired location of footsteps (need to run update_fsteps once)
    if (k == 0):
        fstep_planner.update_fsteps(k, interface.l_feet, np.vstack((interface.lV, interface.lW)), joystick.v_ref,
                                    interface.lC[2, 0], interface.oMl, pyb_sim.ftps_Ids, False)

    # Update footsteps desired location once every k_mpc iterations of TSID
    if (k % k_mpc) == 0:

        # fstep_planner.fsteps_invdyn = fstep_planner.fsteps.copy()
        # fstep_planner.gait_invdyn = fstep_planner.gait.copy()

        fstep_planner.update_fsteps(k+1, interface.l_feet, np.vstack((interface.lV, interface.lW)), joystick.v_ref,
                                    interface.lC[2, 0], interface.oMl, pyb_sim.ftps_Ids, joystick.reduced)

        fstep_planner.fsteps_invdyn = fstep_planner.fsteps.copy()
        fstep_planner.gait_invdyn = fstep_planner.gait.copy()

    return 0


def process_mpc(k, k_mpc, interface, joystick, fstep_planner, mpc_wrapper, dt_mpc, sequencer, ID_deb_lines):
    """Update and run the model predictive control to get the reference contact forces that should be
    applied by feet in stance phase

    Args:
        k (int): Number of inv dynamics iterations since the start of the simulation
        k_mpc (int): Number of inv dynamics iterations for one iteration of the MPC
        interface (object): Interface object of the control loop
        joystick (object): Interface with the gamepad
        fstep_planner (object): Footsteps planner object
        mpc_wrapper (object): Wrapper that acts as a black box for the MPC
        dt_mpc (float): time step of the MPC
        sequencer (object): sequencer object that contains information about the contact sequence
        ID_deb_lines (list): IDs of lines in PyBullet for debug purpose
    """

    # Debug lines
    if len(ID_deb_lines) == 0:
        for i_line in range(4):
            start = interface.oMl * np.array([[interface.l_shoulders[0, i_line], interface.l_shoulders[1, i_line], 0.01]]).transpose()
            end = interface.oMl * np.array([[interface.l_shoulders[0, i_line] + 0.4, interface.l_shoulders[1, i_line], 0.01]]).transpose()
            lineID = pyb.addUserDebugLine(np.array(start).ravel().tolist(), np.array(end).ravel().tolist(), lineColorRGB=[1.0, 0.0, 0.0], lineWidth=8)
            ID_deb_lines.append(lineID)
    else:
        for i_line in range(4):
            start = interface.oMl * np.array([[interface.l_shoulders[0, i_line], interface.l_shoulders[1, i_line], 0.01]]).transpose()
            end = interface.oMl * np.array([[interface.l_shoulders[0, i_line] + 0.4, interface.l_shoulders[1, i_line], 0.01]]).transpose()
            lineID = pyb.addUserDebugLine(np.array(start).ravel().tolist(), np.array(end).ravel().tolist(), lineColorRGB=[1.0, 0.0, 0.0], lineWidth=8,
                                            replaceItemUniqueId=ID_deb_lines[i_line])

    # Get the reference trajectory over the prediction horizon
    fstep_planner.getRefStates((k/k_mpc), sequencer.T_gait, interface.lC, interface.abg,
                               interface.lV, interface.lW, joystick.v_ref, h_ref=0.2027682)

    if k > 0:
        if np.abs(mpc_wrapper.mpc.x_robot[7, 0] - interface.lV[1, 0]) > 0.00001:
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
                        sequencer.t_stance, joystick, fstep_planner, interface)
    f_applied = mpc_wrapper.mpc.f_applied

    return f_applied


def process_invdyn(solo, k, f_applied, pyb_sim, interface, joystick, fstep_planner, mpc_wrapper, myController,
                   sequencer, enable_hybrid_control):
    """Update and run the whole body inverse dynamics using information coming from the MPC and the footstep planner

    Args:
        solo (object): Pinocchio wrapper for the quadruped
        k (int): Number of inv dynamics iterations since the start of the simulation
        f_applied (12x1 array): Reference contact forces for all feet (0s for feet in swing phase)
        pyb_sim (object): PyBullet simulation
        interface (object): Interface object of the control loop
        joystick (object): Interface with the gamepad
        fstep_planner (object): Footsteps planner object
        mpc_wrapper (object): Wrapper that acts as a black box for the MPC
        myController (object): Inverse Dynamics controller
        sequencer (object): sequencer object that contains information about the contact sequence
        enable_hybrid_control (bool): whether hybrid control is enabled or not
    """

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

    # TSID needs the velocity of the robot in base frame
    """pyb_sim.vmes12[0:3, 0:1] = interface.oMb.rotation.transpose() @ pyb_sim.vmes12[0:3, 0:1]
    pyb_sim.vmes12[3:6, 0:1] = interface.oMb.rotation.transpose() @ pyb_sim.vmes12[3:6, 0:1]"""
    pyb_sim.qmes12 = myController.qtsid.copy()
    pyb_sim.vmes12 = myController.vtsid.copy()

    # Initial conditions
    if k == 0:
        myController.qtsid = pyb_sim.qmes12.copy()
        myController.vtsid = pyb_sim.vmes12.copy()

    # Retrieve the joint torques from the current active controller
    if enable_hybrid_control:
        jointTorques = myController.control(myController.qtsid, myController.vtsid, k, solo,
                                            sequencer, interface, joystick.v_ref, f_applied,
                                            fstep_planner.fsteps_invdyn, fstep_planner.gait_invdyn, pyb_sim.ftps_Ids_deb,
                                            enable_hybrid_control, pyb_sim.qmes12, pyb_sim.vmes12,
                                            mpc_wrapper.mpc.q_next, mpc_wrapper.mpc.v_next).reshape((12, 1))
    else:
        jointTorques = myController.control(pyb_sim.qmes12, pyb_sim.vmes12, k, solo,
                                            sequencer, interface, joystick.v_ref, f_applied,
                                            fstep_planner.fsteps_invdyn, fstep_planner.gait_invdyn, pyb_sim.ftps_Ids_deb).reshape((12, 1))

    return jointTorques


def process_pybullet(pyb_sim, jointTorques):
    """Update the torques applied by the actuators of the quadruped and run one step of simulation

    Args:
        pyb_sim (object): PyBullet simulation
        jointTorques (12x1 array): Reference torques for the actuators
    """

    # Set control torque for all joints
    pyb.setJointMotorControlArray(pyb_sim.robotId, pyb_sim.revoluteJointIndices,
                                  controlMode=pyb.TORQUE_CONTROL, forces=jointTorques)

    # Compute one step of simulation
    pyb.stepSimulation()

    # Refresh force monitoring for PyBullet
    # myForceMonitor.display_contact_forces()

    # Save PyBullet camera frame
    # You have to process them with something like FFMPEG to create a video
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

    return 0
