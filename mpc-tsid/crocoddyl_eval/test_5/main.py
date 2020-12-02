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
import Logger
from crocoddyl_class.MPC_crocoddyl import MPC_crocoddyl
from crocoddyl_class.MPC_crocoddyl_2 import MPC_crocoddyl_2

def run_scenario(envID, velID, dt_mpc, k_mpc, t, n_periods, T_gait, N_SIMULATION, type_MPC, pyb_feedback , desired_speed):

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
    N_SIMULATION = 1000  # number of time steps simulated"""

    N_1 = np.round(int(np.around(abs(desired_speed[0]), decimals = 1 ) * 10000 + 2000) , -3 )
    N_2 = np.round(int(np.around(abs(desired_speed[1]), decimals = 1 ) * 10000 + 10000) , -3 )
    N_3 = np.round(int(np.around(abs(desired_speed[5]), decimals = 1 ) * 2500 + 10000) , -3 )
    
    N_SIMULATION = max(N_1 , N_2,N_3)

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
    joystick, fstep_planner, logger_ddp, interface = utils.init_objects(
        dt, dt_mpc, N_SIMULATION, k_mpc, n_periods, T_gait, False)

    # Multi simulation environment
    joystick.multi_simu = True
    joystick.Vx_ref = desired_speed[0]
    joystick.Vy_ref = desired_speed[1]
    joystick.Vw_ref = desired_speed[5]

    # Create a new logger type for the second solver
    logger_osqp = Logger.Logger(N_SIMULATION, dt, dt_mpc, k_mpc, n_periods, T_gait, True)

    # Wrapper that makes the link with the solver that you want to use for the MPC
    # First argument to True to have PA's MPC, to False to have Thomas's MPC
    enable_multiprocessing = False


    # Initialize the two algorithms
    # Both are running, usefull to compare the solvers
    mpc_wrapper_ddp = MPC_Wrapper.MPC_Wrapper(False, dt_mpc, fstep_planner.n_steps,
                                          k_mpc, fstep_planner.T_gait, enable_multiprocessing)

                        
    mpc_wrapper_ddp_2 = MPC_crocoddyl_2( dt = dt_mpc , T_mpc = T_gait , mu = 0.9, inner = False, linearModel = False  , n_period = 1 , dt_tsid = dt)

    # Enable/Disable hybrid control
    enable_hybrid_control = True

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

    for k in range(int(N_SIMULATION)):
        time_loop = time.time()

        if (k % 1000) == 0:
            print("Iteration: ", k)

        # Process states update and joystick
        proc.process_states(solo, k, k_mpc, velID, pyb_sim, interface, joystick, myController, pyb_feedback)

        if np.isnan(interface.lC[2, 0]):
            print("NaN value for the position of the center of mass. Simulation likely crashed. Ending loop.")
            break

        # Process footstep planner
        proc.process_footsteps_planner(k, k_mpc, pyb_sim, interface, joystick, fstep_planner)       
        

        if (k % k_mpc) == 0:
            time_mpc = time.time()
            # Run both algorithms
            proc.process_mpc(k, k_mpc, interface, joystick, fstep_planner, mpc_wrapper_ddp,
                             dt_mpc, ID_deb_lines)
                     
            logger_ddp.log_fstep_planner( k , fstep_planner)
           
            t_list_mpc[k] = time.time() - time_mpc            
            print( int(k/k_mpc))


        # Process MPC once every k_mpc iterations of TSID
        mpc_wrapper_ddp_2.updateProblem(k,fstep_planner.fsteps , fstep_planner.xref , interface.lC, interface.abg, interface.lV, interface.lW, joystick.v_ref)
        mpc_wrapper_ddp_2.ddp.solve([],[],50)
      
        f_applied = mpc_wrapper_ddp_2.get_latest_result()

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
                print('NaN value in feedforward torque. Ending loop.')
                break

            # Process PyBullet
            proc.process_pybullet(pyb_sim, k, envID, jointTorques)

        # Call logger object to log various parameters
        logger_ddp.call_log_functions(k, pyb_sim, joystick, fstep_planner, interface, mpc_wrapper_ddp, myController,
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

    finished = False 

    if k == N_SIMULATION - 1 : 
        finished = True 
    
    print(finished)

    print("END")

    pyb.disconnect()

    return logger_ddp 


"""# Display what has been logged by the logger
logger.plot_graphs(enable_multiprocessing=False)

quit()

# Display duration of MPC block and Inverse Dynamics block
plt.figure()
plt.plot(t_list_tsid, 'k+')
plt.title("Time TSID")
plt.show(block=True)"""
