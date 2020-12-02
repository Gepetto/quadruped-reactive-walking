# coding: utf8
# coding: utf8

import sys
import os 
from sys import argv
sys.path.insert(0, os.getcwd()) # adds current directory to python path

import numpy as np
import MPC_Wrapper 

import numpy as np
import matplotlib.pylab as plt
from TSID_Debug_controller_four_legs_fb_vel import controller, dt
from main import run_scenario
from IPython import embed

import crocoddyl_class.MPC_crocoddyl as MPC_crocoddyl
import FootstepPlanner
import random
import crocoddyl 

envID = 0
velID = 0

t = 0.0 
dt_mpc = 0.02  # Time step of the MPC
k_mpc = int(dt_mpc / dt)  # dt is dt_tsid, defined in the TSID controller script
n_periods = 1  # Number of periods in the prediction horizon
T_gait = 0.32  # Duration of one gait period
N_SIMULATION = 100  # number of simulated TSID time steps
# Which MPC solver you want to use
# True to have PA's MPC, to False to have Thomas's MPC
type_MPC = False

# Whether PyBullet feedback is enabled or not
pyb_feedback = True

#################
# RUN SCENARIOS #
#################

# Run a scenario and retrieve data thanks to the logger
logger_ddp  = run_scenario(envID, velID, dt_mpc, k_mpc, t, n_periods, T_gait, N_SIMULATION, type_MPC, pyb_feedback)


# Create footstep planner object
fstep_planner = FootstepPlanner.FootstepPlanner(dt_mpc, n_periods, T_gait)

mpc_wrapper_ddp = MPC_crocoddyl.MPC_crocoddyl(dt_mpc, T_gait, 1,  inner = True , linearModel = False)

a = 1
b = -1 

N_trial = 10


epsilon = 10e-6


################################################
## CHECK DERIVATIVE WITH NUM_DIFF 
#################################################


model_diff = crocoddyl.ActionModelNumDiff(mpc_wrapper_ddp.ListAction[0])
data = model_diff.createData()
dataCpp = mpc_wrapper_ddp.ListAction[0].createData()

# RUN CALC DIFF
def run_calcDiff_numDiff(epsilon) :
  Lx = 0
  Lx_err = 0
  Lu = 0
  Lu_err = 0
  Lxx = 0
  Lxx_err = 0
  Luu = 0
  Luu_err = 0
  Luu_noFri = 0
  Luu_err_noFri = 0
  Fx = 0
  Fx_err = 0 
  Fu = 0
  Fu_err = 0    

  for k in range(N_trial):    

    N_iteration = random.randint(0, int(N_SIMULATION/k_mpc) - 1  ) 
    N_model = random.randint(0, np.int(T_gait/dt_mpc)- 1 ) 

    x = a + (b-a)*np.random.rand(12)
    u = a + (b-a)*np.random.rand(12)

    mpc_wrapper_ddp.updateProblem(logger_ddp.fsteps[:,:,N_iteration] ,  logger_ddp.xref[:,:, N_iteration] )
    actionModel = mpc_wrapper_ddp.ListAction[N_model]
    model_diff = crocoddyl.ActionModelNumDiff(actionModel)
    
   
    # Run calc & calcDiff function : numDiff 
    
    model_diff.calc(data , x , u )
    model_diff.calcDiff(data , x , u )
    
    # Run calc & calcDiff function : c++
    actionModel.calc(dataCpp , x , u )
    actionModel.calcDiff(dataCpp , x , u )

    Lx +=  np.sum( abs((data.Lx - dataCpp.Lx )) >= epsilon  ) 
    Lx_err += np.sum( abs((data.Lx - dataCpp.Lx )) )  

    Lu +=  np.sum( abs((data.Lu - dataCpp.Lu )) >= epsilon  ) 
    Lu_err += np.sum( abs((data.Lu - dataCpp.Lu )) )  

    Lxx +=  np.sum( abs((data.Lxx - dataCpp.Lxx )) >= epsilon  ) 
    Lxx_err += np.sum( abs((data.Lxx - dataCpp.Lxx )) )  

    Luu +=  np.sum( abs((data.Luu - dataCpp.Luu )) >= epsilon  ) 
    Luu_err += np.sum( abs((data.Luu - dataCpp.Luu )) ) 

    Fx +=  np.sum( abs((data.Fx - dataCpp.Fx )) >= epsilon  ) 
    Fx_err += np.sum( abs((data.Fx - dataCpp.Fx )) )  

    Fu +=  np.sum( abs((data.Fu - dataCpp.Fu )) >= epsilon  ) 
    Fu_err += np.sum( abs((data.Fu - dataCpp.Fu )) )  

    # No friction cone : 
    actionModel.frictionWeights = 0.0    
    model_diff = crocoddyl.ActionModelNumDiff(actionModel)   
   
    # Run calc & calcDiff function : numDiff     
    model_diff.calc(data , x , u )
    model_diff.calcDiff(data , x , u )
    
    # Run calc & calcDiff function : c++
    actionModel.calc(dataCpp , x , u )
    actionModel.calcDiff(dataCpp , x , u )
    Luu_noFri +=  np.sum( abs((data.Luu - dataCpp.Luu )) >= epsilon  ) 
    Luu_err_noFri += np.sum( abs((data.Luu - dataCpp.Luu )) ) 


  
  Lx_err = Lx_err /N_trial
  Lu_err = Lu_err/N_trial
  Lxx_err = Lxx_err/N_trial    
  Luu_err = Luu_err/N_trial
  Fx_err = Fx_err/N_trial
  Fu_err = Fu_err/N_trial
  
  return Lx , Lx_err , Lu , Lu_err , Lxx , Lxx_err , Luu , Luu_err, Luu_noFri , Luu_err_noFri, Fx, Fx_err, Fu , Fu_err


Lx , Lx_err , Lu , Lu_err , Lxx , Lxx_err , Luu , Luu_err , Luu_noFri , Luu_err_noFri , Fx, Fx_err, Fu , Fu_err = run_calcDiff_numDiff(epsilon)

print("\n \n ------------------------------------------ " )
print(" Checking implementation of the derivatives ")
print(" Using crocoddyl NumDiff class")
print(" ------------------------------------------ " )

print("\n Luu is calculated with the residual cost and can not be exact")
print(" Because of the friction cost term")
print("\n")
print("Espilon : %f" %epsilon)
print("N_trial : %f" %N_trial)
print("\n")

if Fx == 0:  print("Fx : OK    (error : %f)" %Fx_err)
else :     print("Fx : NOT OK !!!   (error : %f)" %Fx_err)

if Fu == 0:  print("Fu : OK    (error : %f)" %Fu_err)
else :     print("Fu : NOT OK !!!   (error : %f)" %Fu_err)
if Lx == 0:  print("Lx : OK    (error : %f)" %Lx_err)
else :     print("Lx : NOT OK !!!    (error : %f)" %Lx_err )

if Lu == 0:  print("Lu : OK    (error : %f)" %Lu_err)
else :     print("Lu : NOT OK !!!    (error : %f)" %Lu_err)

if Lxx == 0:  print("Lxx : OK    (error : %f)" %Lxx_err)
else :     print("Lxx : NOT OK !!!   (error : %f)" %Lxx_err)

if Luu == 0:  print("Luu : OK    (error : %f)" %Luu_err)
else :     print("Luu : NOT OK !!!   (error : %f)" %Luu_err)

if Luu_noFri == 0:  print("Luu : OK    (error : %f) , no friction cone" %Luu_err_noFri)
else :     print("Luu : NOT OK !!!   (error : %f) , no friction cone" %Luu_err_noFri )



if Lx == 0 and Lu == 0 and Lxx == 0 and Luu_noFri == 0 and Fu == 0 and Fx == 0: print("\n      -->      Derivatives : OK")
else : print("\n         -->    Derivatives : NOT OK !!!")