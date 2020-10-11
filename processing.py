# coding: utf8

import numpy as np
import pybullet as pyb
import pinocchio as pin


def process_states(solo, k, k_mpc, velID, interface, joystick, tsid_controller, estimator, pyb_feedback):
    """Update states by retrieving information from the simulation and the gamepad

    Args:
        solo (object): Pinocchio wrapper for the quadruped
        k (int): Number of inv dynamics iterations since the start of the simulation
        k_mpc (int): Number of inv dynamics iterations for one iteration of the MPC
        velID (int): Identifier of the current velocity profile to be able to handle different scenarios
        interface (object): Interface object of the control loop
        joystick (object): Interface with the gamepad
        tsid_controller (object): Inverse dynamics controller
        estimator (object): Estimator object which estimates the state of the robot
        pyb_feedback (bool): Whether PyBullet feedback is enabled or not
    """

    ###############
    # PROCESS MPC #
    ###############

    mpc_feedback_orientation = False
    mpc_feedback_velocity = False

    if k != 0:

        # Height of the robot for the MPC
        interface.qmpc[2, 0] = tsid_controller.qdes[2]

        # Positions and velocities of actuators
        interface.qmpc[7:, 0] = tsid_controller.qdes[7:]  # in TSID world frame
        interface.vmpc[6:, 0:1] = tsid_controller.vdes[6:, 0:1]  # in robot base frame

        # Orientation feedback from the real robot to the MPC
        if mpc_feedback_orientation:
            # Roll and Pitch from the estimator
            RPY_pyb = pin.rpy.matrixToRpy(pin.Quaternion(estimator.q_filt[3:7]).toRotationMatrix())
            interface.qmpc[3:7, 0] = np.array(pyb.getQuaternionFromEuler(np.array([RPY_pyb[0], RPY_pyb[1], 0.0])))
        else:
            # Roll and Pitch from TSID
            RPY_tsid = pin.rpy.matrixToRpy(pin.Quaternion((tsid_controller.qdes[3:7]).reshape(4, 1)).toRotationMatrix())
            interface.qmpc[3:7, 0] = np.array(pyb.getQuaternionFromEuler(np.array([RPY_tsid[0], RPY_tsid[1], 0.0])))

        # Velocity feedback from the real robot to the MPC
        if mpc_feedback_velocity:
            # Linear/angular velocity from the estimator
            interface.vmpc[0:6, 0:1] = estimator.v_filt[0:6, 0:1].copy()
        else:
            # Linear/angular velocity from TSID
            interface.vmpc[0:6, 0:1] = tsid_controller.vdes[0:6, 0:1].copy()

        # Update data using the state of the MPC
        interface.update_mpc(solo, interface.qmpc, interface.vmpc)

    ################
    # PROCESS TSID #
    ################

    tsid_feedback_orientation = False
    tsid_feedback_velocity = False

    if k != 0:

        # State of the robot for TSID
        interface.qtsid[:, 0] = tsid_controller.qdes.copy()  # in TSID world frame
        interface.vtsid[:, 0:1] = tsid_controller.vdes.copy()  # in robot base frame

        # Orientation feedback from the real robot to TSID
        if tsid_feedback_orientation:

            # Roll and Pitch from the estimator
            RPY_pyb = pin.rpy.matrixToRpy(pin.Quaternion(estimator.q_filt[3:7]).toRotationMatrix())
            RPY_tsid = pin.rpy.matrixToRpy(pin.Quaternion(interface.qtsid[3:7]).toRotationMatrix())
            interface.qtsid[3:7, 0] = np.array(pyb.getQuaternionFromEuler(
                np.array([RPY_pyb[0], RPY_pyb[1], RPY_tsid[2]])))

        # Velocity feedback from the real robot to the MPC
        if tsid_feedback_velocity:

            # Linear/angular velocity from the estimator
            interface.vtsid[0:6, 0:1] = estimator.v_filt[0:6, 0:1].copy()

        # Update data using the state of TSID
        interface.update_tsid(solo, interface.qtsid, interface.vtsid)

        print("###")
        print("TSID vdes: ", tsid_controller.vdes[0:6, 0:1].ravel())
        print("TSID vtsid:", interface.vtsid[0:6, 0:1].ravel())
        print("MPC vmpc:  ", interface.vmpc[0:6, 0:1].ravel())
        print("Est vfilt: ", estimator.v_filt[0:6, 0])

    else:

        # Starting values are those of the estimator
        interface.qmpc = estimator.q_filt.copy()
        interface.vmpc = estimator.v_filt.copy()
        interface.qtsid = estimator.q_filt.copy()
        interface.vtsid = estimator.v_filt.copy()

        # Update data using the default state of the estimator
        interface.update_mpc(solo, interface.qmpc, interface.vmpc)
        interface.update_tsid(solo, interface.qtsid, interface.vtsid)

    ##############################
    # PROCESS REFERENCE VELOCITY #
    ##############################


    # Update the reference velocity coming from the gamepad once every k_mpc iterations of TSID
    if (k % k_mpc) == 0:
        joystick.update_v_ref(k, velID)

        # Legs have a limited length so the reference velocity has to be limited
        # v_max = (4 / tsid_controller.T_gait) * 0.155
        # math.sqrt(0.3**2 - pyb_sim.qmes12[2, 0]**2)  # (length leg - 2 cm)** 2 - h_base ** 2
        # (joystick.v_ref[0:2])[joystick.v_ref[0:2] > v_max] = v_max
        # (joystick.v_ref[0:2])[joystick.v_ref[0:2] < -v_max] = -v_max

    return 0


def process_footsteps_planner(k, k_mpc, interface, joystick, fstep_planner):
    """Update desired location of footsteps depending on the current state of the robot
    and the reference velocity

    Args:
        k (int): Number of inv dynamics iterations since the start of the simulation
        k_mpc (int): Number of inv dynamics iterations for one iteration of the MPC
        interface (object): Interface object of the control loop
        joystick (object): Interface with the gamepad
        fstep_planner (object): Footsteps planner object
    """

    # Initialization of the desired location of footsteps (need to run update_fsteps once)
    if (k == 0):
        fstep_planner.update_fsteps(k, k_mpc, interface.l_feet, np.vstack((interface.lV, interface.lW)),
                                    joystick.v_ref, interface.lC[2], interface.oMl, False)

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
                                        joystick.v_ref, interface.lC[2, 0], interface.oMl, joystick.reduced)

        fstep_planner.fsteps_invdyn = fstep_planner.fsteps.copy()
        fstep_planner.gait_invdyn = fstep_planner.gait.copy()

        # if k > 640:
        # print("###")
        # print(fstep_planner.gait_invdyn[0:5, :])
        # print(fstep_planner.fsteps_invdyn[0:5, :])

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

    print("X0:", fstep_planner.x0.ravel())
    """print(fstep_planner.xref[:, 0:3].transpose())"""

    # if k > 2100:
    #    print(fstep_planner.xref)

    # Run the MPC to get the reference forces and the next predicted state
    # Result is stored in mpc.f_applied, mpc.q_next, mpc.v_next
    try:
        mpc_wrapper.solve(k, fstep_planner)
    except ValueError:
        print("MPC Problem")

    return 0


def process_invdyn(solo, k, f_applied, estimator, interface, fstep_planner, myController,
                   enable_hybrid_control, enable_gepetto_viewer):
    """Update and run the whole body inverse dynamics using information coming from the MPC and the footstep planner

    Args:
        solo (object): Pinocchio wrapper for the quadruped
        k (int): Number of inv dynamics iterations since the start of the simulation
        f_applied (12x1 array): Reference contact forces for all feet (0s for feet in swing phase)
        estimator (object): state estimator object
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
        print("CONTROL WITHOUT ENABLE HYBRID CONTROL NEEDS TO BE CHECKED")
        # estimator.v_filt[0:3, 0:1] = interface.oMb.rotation.transpose() @ estimator.v_filt[0:3, 0:1]
        # estimator.v_filt[3:6, 0:1] = interface.oMb.rotation.transpose() @ estimator.v_filt[3:6, 0:1]

    """pyb_sim.qmes12 = myController.qtsid.copy()
    pyb_sim.vmes12 = myController.vtsid.copy()"""

    # Initial conditions
    if k == 0:
        myController.qtsid = interface.qtsid.copy()
        myController.vtsid = interface.vtsid.copy()

    # Retrieve the joint torques from the current active controller
    if enable_hybrid_control:
        myController.control(interface.qtsid, interface.vtsid, k, solo,
                             interface, f_applied, fstep_planner.fsteps_invdyn,
                             fstep_planner.gait_invdyn,
                             enable_hybrid_control, enable_gepetto_viewer,
                             estimator.q_filt, estimator.v_filt
                             )
    else:
        myController.control(estimator.q_filt, estimator.v_filt, k, solo,
                             interface, f_applied, fstep_planner.fsteps_invdyn,
                             fstep_planner.gait_invdyn)  # .reshape((12, 1))

    return 0


def process_pdp(myController, estimator):
    """Compute the output of the PD+ controller by comparing the current angular position
    and velocities of actuators with the desired ones
    It returns the desired torques that should be sent to the drivers of the actuators

    Args:
        myController (object): Inverse Dynamics controller
        estimator (object): state estimator object
    """

    myController.qmes[7:, 0] = estimator.q_filt[7:, 0]
    myController.vmes[6:, 0] = estimator.v_filt[6:, 0]

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
