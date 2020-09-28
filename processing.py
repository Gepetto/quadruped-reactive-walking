# coding: utf8

import numpy as np
import pybullet as pyb
import pinocchio as pin


def process_states(solo, k, k_mpc, velID, pyb_sim, interface, joystick, tsid_controller, estimator, pyb_feedback):
    """Update states by retrieving information from the simulation and the gamepad

    Args:
        solo (object): Pinocchio wrapper for the quadruped
        k (int): Number of inv dynamics iterations since the start of the simulation
        k_mpc (int): Number of inv dynamics iterations for one iteration of the MPC
        velID (int): Identifier of the current velocity profile to be able to handle different scenarios
        pyb_sim (object): PyBullet simulation
        interface (object): Interface object of the control loop
        joystick (object): Interface with the gamepad
        tsid_controller (object): Inverse dynamics controller
        estimator (object): Estimator object which estimates the state of the robot
        pyb_feedback (bool): Whether PyBullet feedback is enabled or not
    """

    if k != 0:
        # Retrieve data from the simulation (position/orientation/velocity of the robot)
        # Stored in pyb_sim.qmes12 and pyb_sim.vmes12 (quantities in PyBullet world frame)
        # pyb_sim.retrieve_pyb_data()
        estimator.log_v_truth[:, estimator.k_log] = estimator.b_baseVel.ravel()
        pyb_sim.qmes12[0:3, 0] = estimator.cheat_lin_pos
        pyb_sim.qmes12[3:7, 0] = estimator.quat_oMb
        pyb_sim.vmes12[0:3, 0] = (estimator.oMb.rotation @ np.array([estimator.filt_lin_vel]).transpose()).ravel()
        pyb_sim.vmes12[3:6, 0] = (estimator.oMb.rotation @ np.array([estimator.filt_ang_vel]).transpose()).ravel()

        # Retrieve state desired by TSID (position/orientation/velocity of the robot)
        tsid_controller.qtsid[:, 0] = tsid_controller.qdes.copy()  # in TSID world frame
        tsid_controller.vtsid[:, 0:1] = tsid_controller.vdes.copy()  # in robot base frame

        # Take into account vertical drift in TSID world
        # tsid_controller.qtsid[2, 0] -= interface.offset_z

        # If PyBullet feedback is enabled, we want to mix PyBullet data into TSID desired state
        if pyb_feedback:
            # Orientation is roll/pitch of PyBullet and Yaw of TSID
            interface.RPY_pyb = pin.rpy.matrixToRpy(pin.Quaternion(pyb_sim.qmes12[3:7]).toRotationMatrix())
            interface.RPY_tsid = pin.rpy.matrixToRpy(pin.Quaternion(tsid_controller.qtsid[3:7]).toRotationMatrix())
            tsid_controller.qtsid[3:7, 0:1] = np.array([pyb.getQuaternionFromEuler(np.array([interface.RPY_pyb[0],
                                                                                             interface.RPY_pyb[1],
                                                                                             interface.RPY_tsid[2]]))]
                                                       ).transpose()

        # Transform from TSID world frame to robot base frame (just rotation part)
        interface.oMb_tsid = pin.SE3(pin.Quaternion(tsid_controller.qtsid[3:7, 0:1]), np.array([0.0, 0.0, 0.0]))

        # If PyBullet feedback is enabled, we want to mix PyBullet data into TSID desired state
        if pyb_feedback:

            # Linear/angular velocity taken from PyBullet (in PyBullet world frame)
            tsid_controller.vtsid[0:6, 0:1] = pyb_sim.vmes12[0:6, 0:1].copy()

            # Transform from PyBullet world frame to robot base frame (just rotation part)
            interface.oMb_pyb = pin.SE3(pin.Quaternion(pyb_sim.qmes12[3:7, 0:1]), np.array([0.0, 0.0, 0.0]))

            # Get linear and angular velocities from PyBullet world frame to robot base frame
            tsid_controller.vtsid[0:3, 0:1] = interface.oMb_pyb.rotation.transpose() @ tsid_controller.vtsid[0:3, 0:1]
            tsid_controller.vtsid[3:6, 0:1] = interface.oMb_pyb.rotation.transpose() @ tsid_controller.vtsid[3:6, 0:1]

    # Algorithm needs the velocity of the robot in world frame
    if k == 0:
        # Retrieve data from the simulation (position/orientation/velocity of the robot)
        pyb_sim.retrieve_pyb_data()
        pyb_sim.qmes12[2, 0] = 0.2027682

        # Update the interface that makes the interface between the simulation and the MPC/TSID
        interface.update(solo, pyb_sim.qmes12, pyb_sim.vmes12)

    else:
        # To update the interface we need the position/velocity in TSID world frame
        # qtsid is already in TSID world frame
        # vtsid is in robot base frame, need to get it into TSID world frame
        pyb_sim.qtsid_w = tsid_controller.qtsid.copy()
        pyb_sim.vtsid_w = tsid_controller.vtsid.copy()
        pyb_sim.vtsid_w[0:3, 0:1] = interface.oMb_tsid.rotation @ pyb_sim.vtsid_w[0:3, 0:1]
        pyb_sim.vtsid_w[3:6, 0:1] = interface.oMb_tsid.rotation @ pyb_sim.vtsid_w[3:6, 0:1]

        # Update the interface that makes the interface between the simulation and the MPC/TSID
        interface.update(solo, pyb_sim.qtsid_w, pyb_sim.vtsid_w)

    # Update the reference velocity coming from the gamepad once every k_mpc iterations of TSID
    if (k % k_mpc) == 0:
        joystick.update_v_ref(k, velID)

        # Legs have a limited length so the reference velocity has to be limited
        # v_max = (4 / tsid_controller.T_gait) * 0.155
        # math.sqrt(0.3**2 - pyb_sim.qmes12[2, 0]**2)  # (length leg - 2 cm)** 2 - h_base ** 2
        # (joystick.v_ref[0:2])[joystick.v_ref[0:2] > v_max] = v_max
        # (joystick.v_ref[0:2])[joystick.v_ref[0:2] < -v_max] = -v_max

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
        fstep_planner.update_fsteps(k, k_mpc, interface.l_feet, np.vstack((interface.lV, interface.lW)),
                                    joystick.v_ref, interface.lC[2], interface.oMl, pyb_sim.ftps_Ids, False)

    # Update position of debug spheres
    """import pybullet as pyb
    for j in range(4):
        pos_tmp = np.array([interface.o_feet[:, j]]).transpose()
        pyb.resetBasePositionAndOrientation(pyb_sim.ftps_Ids[j, 0],
                                            posObj=pos_tmp,
                                            ornObj=np.array([0.0, 0.0, 0.0, 1.0]))"""

    # Update footsteps desired location once every k_mpc iterations of TSID
    if True:  # (k % k_mpc) == 0:

        # fstep_planner.fsteps_invdyn = fstep_planner.fsteps.copy()
        # fstep_planner.gait_invdyn = fstep_planner.gait.copy()

        if (k != 0):
            fstep_planner.update_fsteps(k+1, k_mpc, interface.l_feet, np.vstack((interface.lV, interface.lW)),
                                        joystick.v_ref, interface.lC[2, 0], interface.oMl, pyb_sim.ftps_Ids,
                                        joystick.reduced)

        fstep_planner.fsteps_invdyn = fstep_planner.fsteps.copy()
        fstep_planner.gait_invdyn = fstep_planner.gait.copy()

        #if k > 640:
            #print("###")
            #print(fstep_planner.gait_invdyn[0:5, :])
            #print(fstep_planner.fsteps_invdyn[0:5, :])

        fstep_planner.fsteps_mpc = fstep_planner.fsteps.copy()
        fstep_planner.gait_mpc = fstep_planner.gait.copy()

        """if (k % k_mpc) == 0:
            # Since the MPC will output its results one gait step in the future we give it a gait matrix that is
            # shifted one gait step in the future compared to TSID (so that the result is properly timed when we
            # retrieve it).
            fstep_planner.update_fsteps(0, k_mpc, interface.l_feet, np.vstack((interface.lV, interface.lW)),
                                        joystick.v_ref, interface.lC[2, 0], interface.oMl, pyb_sim.ftps_Ids,
                                        joystick.reduced)
            fstep_planner.fsteps_mpc = fstep_planner.fsteps.copy()
            fstep_planner.gait_mpc = fstep_planner.gait.copy()

            # Reverse the "one gait step into the future" for TSID
            fstep_planner.gait = fstep_planner.gait_invdyn.copy()"""

    return 0


def process_mpc(k, k_mpc, interface, joystick, fstep_planner, mpc_wrapper, dt_mpc, ID_deb_lines):
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
        ID_deb_lines (list): IDs of lines in PyBullet for debug purpose
    """

    # Debug lines to better visualizer the forward direction of the robot and the vertical of the shoulders
    """for i_line in range(4):
        start = interface.oMl * np.array([[interface.l_shoulders[0, i_line],
                                           interface.l_shoulders[1, i_line], 0.01]]).transpose()
        end = interface.oMl * np.array([[interface.l_shoulders[0, i_line] + 0.4,
                                         interface.l_shoulders[1, i_line], 0.01]]).transpose()
        if len(ID_deb_lines) < 4:  # Create the lines
            lineID = pyb.addUserDebugLine(np.array(start).ravel().tolist(), np.array(end).ravel().tolist(),
                                          lineColorRGB=[1.0, 0.0, 0.0], lineWidth=8)
            ID_deb_lines.append(lineID)
        else:  # Update the lines
            lineID = pyb.addUserDebugLine(np.array(start).ravel().tolist(), np.array(end).ravel().tolist(),
                                          lineColorRGB=[1.0, 0.0, 0.0], lineWidth=8,
                                          replaceItemUniqueId=ID_deb_lines[i_line])"""

    # Get the reference trajectory over the prediction horizon
    fstep_planner.getRefStates((k/k_mpc), fstep_planner.T_gait, interface.lC, interface.abg,
                               interface.lV, interface.lW, joystick.v_ref, h_ref=0.2027682,
                               predefined=joystick.predefined)

    # Run the MPC to get the reference forces and the next predicted state
    # Result is stored in mpc.f_applied, mpc.q_next, mpc.v_next
    try:
        mpc_wrapper.solve(k, fstep_planner)
    except ValueError:
        print("MPC Problem")

    return 0


def process_invdyn(solo, k, f_applied, pyb_sim, interface, fstep_planner, myController,
                   enable_hybrid_control, enable_gepetto_viewer):
    """Update and run the whole body inverse dynamics using information coming from the MPC and the footstep planner

    Args:
        solo (object): Pinocchio wrapper for the quadruped
        k (int): Number of inv dynamics iterations since the start of the simulation
        f_applied (12x1 array): Reference contact forces for all feet (0s for feet in swing phase)
        pyb_sim (object): PyBullet simulation
        interface (object): Interface object of the control loop
        joystick (object): Interface with the gamepad
        fstep_planner (object): Footsteps planner object
        myController (object): Inverse Dynamics controller
        enable_hybrid_control (bool): whether hybrid control is enabled or not
        enable_gepetto_viewer (bool): whether the gepetto viewer is enabled or not (to update it if it is)
    """

    #####################################
    # Get torques with inverse dynamics #
    #####################################

    # TSID needs the velocity of the robot in base frame
    if not enable_hybrid_control:
        pyb_sim.vmes12[0:3, 0:1] = interface.oMb.rotation.transpose() @ pyb_sim.vmes12[0:3, 0:1]
        pyb_sim.vmes12[3:6, 0:1] = interface.oMb.rotation.transpose() @ pyb_sim.vmes12[3:6, 0:1]

    """pyb_sim.qmes12 = myController.qtsid.copy()
    pyb_sim.vmes12 = myController.vtsid.copy()"""

    # Initial conditions
    if k == 0:
        myController.qtsid = pyb_sim.qmes12.copy()
        myController.vtsid = pyb_sim.vmes12.copy()

    # Retrieve the joint torques from the current active controller
    if enable_hybrid_control:
        myController.control(myController.qtsid, myController.vtsid, k, solo,
                             interface, f_applied, fstep_planner.fsteps_invdyn,
                             fstep_planner.gait_invdyn, pyb_sim.ftps_Ids_deb,
                             enable_hybrid_control, enable_gepetto_viewer,
                             pyb_sim.qmes12, pyb_sim.vmes12
                             )
    else:
        myController.control(pyb_sim.qmes12, pyb_sim.vmes12, k, solo,
                             interface, f_applied, fstep_planner.fsteps_invdyn,
                             fstep_planner.gait_invdyn, pyb_sim.ftps_Ids_deb)  # .reshape((12, 1))

    return 0


def process_pdp(pyb_sim, myController, estimator):
    """Compute the output of the PD+ controller by comparing the current angular position
    and velocities of actuators with the desired ones
    It returns the desired torques that should be sent to the drivers of the actuators

    Args:
        pyb_sim (object): Wrapper of the PyBullet simulator
        myController (object): Inverse Dynamics controller
        estimator (object): state estimator object
    """

    myController.qmes[7:, 0] = estimator.actuators_pos[:]
    myController.vmes[6:, 0] = estimator.actuators_vel[:]

    return myController.run_PDplus()


def process_pybullet(pyb_sim, k, envID, velID, jointTorques):
    """Update the torques applied by the actuators of the quadruped and run one step of simulation

    Args:
        pyb_sim(object): PyBullet simulation
        k(int): Number of inv dynamics iterations since the start of the simulation
        envID(int): Identifier of the current environment to be able to handle different scenarios
        velID(int): Identifier of the current velocity profile to be able to handle different scenarios
        jointTorques(12x1 array): Reference torques for the actuators
    """

    # Check the state of the robot to trigger events and update the simulator camera
    pyb_sim.check_pyb_env(k, envID, velID, pyb_sim.qmes12)

    # Set control torque for all joints
    pyb.setJointMotorControlArray(pyb_sim.robotId, pyb_sim.revoluteJointIndices,
                                  controlMode=pyb.TORQUE_CONTROL, forces=jointTorques)

    # Compute one step of simulation
    pyb.stepSimulation()

    # Refresh force monitoring for PyBullet
    # myForceMonitor.display_contact_forces()

    # Save PyBullet camera frame
    # You have to process them with something like FFMPEG to create a video
    """
    import os
    from matplotlib import pyplot as plt
    step = 10
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
