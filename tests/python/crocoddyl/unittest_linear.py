# coding: utf8

import numpy as np

import quadruped_walkgen as quadruped_walkgen
import quadruped_reactive_walking as qrw
import crocoddyl 

#####################
# Select MPC type   #
#####################

model = quadruped_walkgen.ActionModelQuadruped()

# Tune the weights
# model.stateWeights = 2*np.ones(12)
# model.heuristicWeights = 2*np.ones(8)
# model.stepWeights = np.ones(8)

# Update the dynamic of the model 
fstep = np.random.rand(12).reshape((3,4))
xref = np.random.rand(12)
gait = np.random.randint(2, size=4)
model.updateModel(fstep, xref , gait)

################################################
## CHECK DERIVATIVE WITH NUM_DIFF 
#################################################

a = 1
b = -1 
N_trial = 50
epsilon = 10e-4

model_diff = crocoddyl.ActionModelNumDiff(model)
data_diff = model_diff.createData()
data = model.createData()

# RUN CALC DIFF
def run_calcDiff_numDiff(epsilon) :
  Lx = 0
  Lx_err = 0
  Lu = 0
  Lu_err = 0
  Lxu = 0
  Lxu_err = 0
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

    x = a + (b-a)*np.random.rand(12)
    u = a + (b-a)*np.random.rand(12)

    fstep = np.random.rand(12).reshape((3,4))
    xref = np.random.rand(12)
    gait = np.random.randint(2, size=4)
    model.updateModel(fstep, xref , gait)
    model_diff = crocoddyl.ActionModelNumDiff(model)     
   
    # Run calc & calcDiff function : numDiff     
    model_diff.calc(data_diff , x , u )
    model_diff.calcDiff(data_diff , x , u )
    
    # Run calc & calcDiff function : c++ model
    model.calc(data , x , u )
    model.calcDiff(data , x , u )

    Lx +=  np.sum( abs((data.Lx - data_diff.Lx )) >= epsilon  ) 
    Lx_err += np.sum( abs((data.Lx - data_diff.Lx )) )  

    Lu +=  np.sum( abs((data.Lu - data_diff.Lu )) >= epsilon  ) 
    Lu_err += np.sum( abs((data.Lu - data_diff.Lu )) )  

    Lxu +=  np.sum( abs((data.Lxu - data_diff.Lxu )) >= epsilon  ) 
    Lxu_err += np.sum( abs((data.Lxu - data_diff.Lxu )) )  

    Lxx +=  np.sum( abs((data.Lxx - data_diff.Lxx )) >= epsilon  ) 
    Lxx_err += np.sum( abs((data.Lxx - data_diff.Lxx )) )  

    Luu +=  np.sum( abs((data.Luu - data_diff.Luu )) >= epsilon  ) 
    Luu_err += np.sum( abs((data.Luu - data_diff.Luu )) ) 

    Fx +=  np.sum( abs((data.Fx - data_diff.Fx )) >= epsilon  ) 
    Fx_err += np.sum( abs((data.Fx - data_diff.Fx )) )  

    Fu +=  np.sum( abs((data.Fu - data_diff.Fu )) >= epsilon  ) 
    Fu_err += np.sum( abs((data.Fu - data_diff.Fu )) )  
  
  Lx_err = Lx_err /N_trial
  Lu_err = Lu_err/N_trial
  Lxx_err = Lxx_err/N_trial    
  Luu_err = Luu_err/N_trial
  Fx_err = Fx_err/N_trial
  Fu_err = Fu_err/N_trial
  
  return Lx , Lx_err , Lu , Lu_err , Lxu , Lxu_err, Lxx , Lxx_err , Luu , Luu_err, Luu_noFri , Luu_err_noFri, Fx, Fx_err, Fu , Fu_err


Lx, Lx_err, Lu, Lu_err, Lxu, Lxu_err, Lxx, Lxx_err , Luu , Luu_err , Luu_noFri , Luu_err_noFri , Fx, Fx_err, Fu , Fu_err = run_calcDiff_numDiff(epsilon)

print("\n \n ------------------------------------------ " )
print(" Checking implementation of the derivatives ")
print(" Using crocoddyl NumDiff class")
print(" ------------------------------------------ " )

print("\n Luu and Lxx are calculated with the residual cost and cannot be exact")
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

if Lxu == 0:  print("Lxu : OK    (error : %f)" %Lxu_err)
else :     print("Lxu : NOT OK !!!    (error : %f)" %Lxu_err)

if Lxx == 0:  print("Lxx : OK    (error : %f)" %Lxx_err)
else :     print("Lxx : NOT OK !!!   (error : %f)" %Lxx_err)

if Luu == 0:  print("Luu : OK    (error : %f)" %Luu_err)
else :     print("Luu : NOT OK !!!   (error : %f)" %Luu_err)

if Lx == 0 and Lu == 0 and Fx == 0 and Fu == 0:
  print("\n      -->      Derivatives 1st order : OK")
else: 
  print("\n         -->    Derivatives : NOT OK !!!")