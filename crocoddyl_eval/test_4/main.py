# coding: utf8
import sys
import os 
from sys import argv
sys.path.insert(0, os.getcwd()) # adds current directory to python path

import numpy as np
import matplotlib.pylab as plt
import utils
import time

from TSID_Debug_controller_four_legs_fb_vel import controller, dt
import Safety_controller
import EmergencyStop_controller
import ForceMonitor
import processing as proc
import MPC_Wrapper
import pybullet as pyb
from crocoddyl_class.MPC_crocoddyl_planner import *

def run_scenario(envID, velID, dt_mpc, k_mpc, t, n_periods, T_gait, N_SIMULATION_, type_MPC, pyb_feedback, desired_speed):

    ########################################################################
    #                        Parameters definition                         #
    ########################################################################
    """# Time step
    dt_mpc = 0.02
    k_mpc = int(dt_mpc / dt)  # dt is dt_tsid, defined in the TSID controller script
    t = 0.0  # Time
    n_periods = 1  # Number of periods in the prediction horizon
    T_gait = 0.48  # Duration of one gait period
    # Simulation parameters
    N_SIMULATION = 1000  # number of time steps simulated
    desired_speed = np.array x6 , [Vx, Vy, Vz , Vroll, Vpitch, Vyaw]    
    """
    # The time of the simulation depends on the speeds :
    # 0.1 m.s-1 + 2s to check stability (Vx = 1m.s-1 --> 12s of simulation)
    
    N_1 = np.round(int(np.around(abs(desired_speed[0]), decimals = 1 ) * 10000 + 3000) , -3 )

    # 0.4 rad.s-1 for 1s
    N_2 = np.round(int(np.around(abs(desired_speed[5]), decimals = 1 ) * 2500 + 3000) , -3 )

    N_3 = np.round(int(np.around(abs(desired_speed[1]), decimals = 1 ) * 10000 + 3000) , -3 )


    N_SIMULATION = max(N_1 , N_2 , N_3)
   

    # Initialize the error for the simulation time
    time_error = False

    # Lists to log the duration of 1 iteration of the MPC/TSID
    t_list_tsid = [0] * int(N_SIMULATION)
    t_list_loop = [0] * int(N_SIMULATION)
    t_list_mpc = [0] * int(N_SIMULATION)

    # List to store the IDs of debug lines
    ID_deb_lines = []

    # Enable/Disable Gepetto viewer
    enable_gepetto_viewer = False

    # Which MPC solver you want to use
    # True to have PA's MPC, to False to have Thomas's MPC
    """type_MPC = True"""

    # Create Joystick, FootstepPlanner, Logger and Interface objects
    joystick, fstep_planner, logger, interface = utils.init_objects(
        dt, dt_mpc, N_SIMULATION, k_mpc, n_periods, T_gait, type_MPC)

    # Multi simulation environment
    joystick.multi_simu = True
    joystick.Vx_ref = desired_speed[0]
    joystick.Vy_ref = desired_speed[1]
    joystick.Vw_ref = desired_speed[5]

    # Wrapper that makes the link with the solver that you want to use for the MPC
    # First argument to True to have PA's MPC, to False to have Thomas's MPC
    enable_multiprocessing = False
    mpc_wrapper = MPC_Wrapper.MPC_Wrapper(type_MPC, dt_mpc, fstep_planner.n_steps,
                                          k_mpc, fstep_planner.T_gait, enable_multiprocessing)

    # MPC with augmented states
    mpc_planner = MPC_crocoddyl_planner(dt = dt_mpc , T_mpc = fstep_planner.T_gait)
                                        
    # Enable/Disable hybrid control
    enable_hybrid_control = True

    ########################################################################
    #                            Logger DDP                                #
    ########################################################################

    # pred_trajectories = np.zeros((20, int(T_gait/dt_mpc), int(N_SIMULATION/k_mpc)))
    # pred_forces = np.zeros((12, int(T_gait/dt_mpc), int(N_SIMULATION/k_mpc)))
    # fsteps = np.zeros((20,13,int(N_SIMULATION/k_mpc)))
    # gait_ = np.zeros((20,5,int(N_SIMULATION/k_mpc)))
    # l_feet_ = np.zeros((3,4,int(N_SIMULATION/k_mpc)))
    # o_feet_ = np.zeros((3,4,int(N_SIMULATION/k_mpc)))

    ########################################################################
    #                            Gepetto viewer                            #
    ########################################################################

    # Initialisation of the Gepetto viewer
    solo = utils.init_viewer(enable_gepetto_viewer)

    ########################################################################
    #                              PyBullet                                #
    ########################################################################

    # Initialisation of the PyBullet simulator
    pyb_sim = utils.pybullet_simulator(envID, dt=0.001)

    # Force monitor to display contact forces in PyBullet with red lines
    myForceMonitor = ForceMonitor.ForceMonitor(pyb_sim.robotId, pyb_sim.planeId)

    ########################################################################
    #                             Simulator                                #
    ########################################################################

    # Define the default controller as well as emergency and safety controller
    myController = controller(int(N_SIMULATION), k_mpc, n_periods, T_gait)
    mySafetyController = Safety_controller.controller_12dof()
    myEmergencyStop = EmergencyStop_controller.controller_12dof()

    tic = time.time()
    for k in range(int(N_SIMULATION)):
        time_loop = time.time()

        if (k % 1000) == 0:
            pass
            # print("Iteration: ", k)

        # Process states update and joystick
        proc.process_states(solo, k, k_mpc, velID, pyb_sim, interface, joystick, myController, pyb_feedback)

        if np.isnan(interface.lC[2]):
            print("NaN value for the position of the center of mass. Simulation likely crashed. Ending loop.")
            break

        # Process footstep planner
        proc.process_footsteps_planner(k, k_mpc, pyb_sim, interface, joystick, fstep_planner)

        # Process MPC once every k_mpc iterations of TSID
        if (k % k_mpc) == 0:
            time_mpc = time.time()
            if k == 0 :
                proc.process_mpc(k, k_mpc, interface, joystick, fstep_planner, mpc_wrapper,
                                dt_mpc, ID_deb_lines)
            
            else : 
                # if not proc.proc
                proc.process_mpc(k, k_mpc, interface, joystick, fstep_planner, mpc_wrapper,
                                dt_mpc, ID_deb_lines)
            t_list_mpc[k] = time.time() - time_mpc

            # running mpc planner in parallel (xref has been updated in process_mpc)
            # start_time = time.time()
            if type_MPC == False :
                mpc_planner.solve(k, fstep_planner.xref , interface.l_feet )
            # print("Temps d execution : %s secondes ---" % (time.time() - start_time)) 

            #############
            # ddp logger
            #############
            # pred_trajectories[:,:,int(k/k_mpc)] = mpc_planner.Xs
            # pred_forces[:,:,int(k/k_mpc)] = mpc_planner.Us
            # fsteps[:,:,int(k/k_mpc)] = mpc_planner.fsteps.copy()
            # gait_[:,:,int(k/k_mpc)] = mpc_planner.gait
            # l_feet_[:,:,int(k/k_mpc)] = interface.l_feet
            # for i in range(4):
            #     index = next((idx for idx, val in np.ndenumerate(mpc_planner.fsteps[:, 3*i+1]) if ((not (val==0)) and (not np.isnan(val)))), [-1])[0]
            #     pos_tmp = np.reshape(np.array(interface.oMl * (np.array([mpc_planner.fsteps[index, (1+i*3):(4+i*3)]]).transpose())) , (3,1) )
            #     o_feet_[:2, i , int(k/k_mpc)] = pos_tmp[0:2, 0]

        # Replace the fstep_invdyn by the ddp one
        if type_MPC == False : 
            fstep_planner.fsteps_invdyn = mpc_planner.fsteps.copy()

        if k == 0:
            if type_MPC == True : 
                f_applied = mpc_wrapper.get_latest_result()
            else : 
                f_applied = mpc_planner.get_latest_result()
        else:
            # Output of the MPC (with delay)
            if type_MPC == True : 
                f_applied = mpc_wrapper.get_latest_result()
            else : 
                f_applied = mpc_planner.get_latest_result() 
                

        # Process Inverse Dynamics
        time_tsid = time.time()
        jointTorques = proc.process_invdyn(solo, k, f_applied, pyb_sim, interface, fstep_planner,
                                           myController, enable_hybrid_control)
        t_list_tsid[k] = time.time() - time_tsid  # Logging the time spent to run this iteration of inverse dynamics

        # Process PD+ (feedforward torques and feedback torques)
        for i_step in range(1):

            # Process the PD+
            jointTorques = proc.process_pdp(pyb_sim, myController)

            if myController.error:
                # print('NaN value in feedforward torque. Ending loop.')
                break

            # Process PyBullet
            proc.process_pybullet(pyb_sim, k, envID, jointTorques)

        # Call logger object to log various parameters
        logger.call_log_functions(k, pyb_sim, joystick, fstep_planner, interface, mpc_wrapper, myController,
                                 False, pyb_sim.robotId, pyb_sim.planeId, solo)

        t_list_loop[k] = time.time() - time_loop     

        #########################
        #   Camera
        #########################

        # if (k % 20) == 0:
        #     img = pyb.getCameraImage(width=1920, height=1080, renderer=pyb.ER_BULLET_HARDWARE_OPENGL)
        #     #if k == 0:
        #     #    newpath = r'/tmp/recording'
        #     #    if not os.path.exists(newpath):
        #     #       os.makedirs(newpath)
        #     if (int(k/20) < 10):
        #         plt.imsave('tmp/recording/frame_00'+str(int(k/20))+'.png', img[2])
        #     elif int(k/20) < 100:
        #         plt.imsave('tmp/recording/frame_0'+str(int(k/20))+'.png', img[2])
        #     else:
        #         plt.imsave('tmp/recording/frame_'+str(int(k/20))+'.png', img[2])


    ####################
    # END OF MAIN LOOP #
    ####################

    # print("Computation duration: ", time.time()-tic)
    # print("Simulated duration: ", N_SIMULATION*0.001)
    # print("END")
    

    pyb.disconnect()

    finished = False 

    if k == N_SIMULATION - 1 : 
        finished = True 
    
    print("Vx = " , desired_speed[0] , " ;  Vy = " , desired_speed[1] ,  " ;  Vw = " , desired_speed[5] , "  -->  " , finished )


    return finished , desired_speed 


"""# Display what has been logged by the logger
logger.plot_graphs(enable_multiprocessing=False)

quit()

# Display duration of MPC block and Inverse Dynamics block
plt.figure()
plt.plot(t_list_tsid, 'k+')
plt.title("Time TSID")
plt.show(block=True)"""
