# coding: utf8
# coding: utf8

import sys
import os 
from sys import argv
sys.path.insert(0, os.getcwd()) # adds current directory to python path


import numpy as np
import matplotlib.pylab as plt

####################
# Recovery of Data
####################

folder_name = ""
pathIn = "crocoddyl_eval/test_4/log_eval/"
res = np.load(pathIn + folder_name + "results_wyaw_all_false.npy" , allow_pickle=True )
# res1 = np.load(pathIn + folder_name + "results_osqp_wyaw.npy" , allow_pickle=True )

import numpy as np

X = np.linspace(1,-1,25)
# Y = np.linspace(-1,1,65)
W = np.linspace(-2.2,2.2,25)

def find_nearest(Vx , Vy):
    idx = (np.abs(X - Vx)).argmin()
    idy = (np.abs(W - Vy)).argmin()

    return idx , idy


XX , YY = np.meshgrid(X,W)
Z = np.zeros((XX.shape[0] , YY.shape[1]))
Z_osqp = np.zeros((XX.shape[0] , YY.shape[1]))
# plt.figure()

# for elt in res : 
#     if elt[0] == True : 
#         plt.plot(elt[1][0] , elt[1][1] , "bs" , markerSize= "13")
#     else :
#         pass

# plt.xlim([-1,1])
# plt.ylim([-1,1])

plt.figure()

for elt in res : 
    idx , idy = find_nearest(elt[1][0] , elt[1][5])
    Z[idx,idy] = elt[0]

plt.rc('text', usetex=True)
im = plt.imshow(Z ,cmap = plt.cm.binary , extent=(-2.2,2.2,-1,1))
plt.xlabel("Lateral Velocity $\dot{p_y} \hspace{2mm} [m.s^{-1}]$" , fontsize=12)
plt.ylabel("Forward Velocity $\dot{p_x} \hspace{2mm} [m.s^{-1}]$" , fontsize=12)
plt.title("Viable Operating Regions (DDP and foot optimization)" , fontsize=14)


# plt.figure()

# for elt in res1 : 
#     idx , idy = find_nearest(elt[1][0] , elt[1][5])
#     Z_osqp[idx,idy] = elt[0]

# plt.rc('text', usetex=True)
# im = plt.imshow(Z_osqp ,cmap = plt.cm.binary , extent=(-2.2,2.2,-1,1))
# plt.xlabel("Lateral Velocity $\dot{p_y} \hspace{2mm} [m.s^{-1}]$" , fontsize=12)
# plt.ylabel("Forward Velocity $\dot{p_x} \hspace{2mm} [m.s^{-1}]$" , fontsize=12)
# plt.title("Viable Operating Regions OSQP" , fontsize=14)




plt.show()