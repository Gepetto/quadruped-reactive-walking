# coding: utf8
# coding: utf8

import matplotlib.pylab as plt
import numpy as np
"""import sys
import os
from sys import argv
sys.path.insert(0, os.getcwd())  # adds current directory to python path"""


####################
# Recovery of Data
####################

folder_name = ""
pathIn = "crocoddyl_eval/test_4/log_eval/"
#res = np.load(pathIn + folder_name + "results_wyaw_all_false.npy", allow_pickle=True)

N_lin = 6
X = np.linspace(1.4, -1.4, N_lin)
Y = np.linspace(-1.4, 1.4, N_lin)
W = np.linspace(-2.2, 2.2, N_lin)


def find_nearest(A, B, C, D):
    idx = (np.abs(A - B)).argmin()
    idy = (np.abs(C - D)).argmin()
    return idx, idy

res = np.load(pathIn + folder_name + "results_osqp_vy.npy", allow_pickle=True)
plt.figure()
for elt in res :
    if elt[0] == True :
        plt.plot(elt[1][0] , elt[1][1] , "bs" , markerSize= "13")
    else :
        plt.plot(elt[1][0] , elt[1][1] , "rs" , markerSize= "13")
print(res)
plt.xlim([-1.5,1.5])
plt.ylim([-1.5,1.5])
plt.show(block=True)

####################
# Plotting
####################

# Plotting Forward vel VS Lateral vel for OSQP
res_osqp_vy = np.load(pathIn + folder_name + "results_osqp_vy.npy", allow_pickle=True)
XX, YY = np.meshgrid(X, Y)
Z = np.zeros((XX.shape[0], YY.shape[1]))
Z_osqp_vy = np.zeros((XX.shape[0], YY.shape[1]))
for elt in res_osqp_vy:
    idx, idy = find_nearest(X, elt[1][0], Y, elt[1][1])
    Z_osqp_vy[idx, idy] = elt[0]


plt.figure()
plt.rc('text', usetex=True)
im = plt.imshow(Z_osqp_vy, cmap=plt.cm.binary, extent=(-1, 1, -1, 1))
plt.xlabel("Lateral Velocity $\dot{p_y} \hspace{2mm} [m.s^{-1}]$", fontsize=12)
plt.ylabel("Forward Velocity $\dot{p_x} \hspace{2mm} [m.s^{-1}]$", fontsize=12)
plt.title("Viable Operating Regions OSQP", fontsize=14)

# Plotting Forward vel VS Angular vel for OSQP
"""
res_osqp_wyaw = np.load(pathIn + folder_name + "results_osqp_wyaw.npy", allow_pickle=True)
XX, YY = np.meshgrid(X, W)
Z = np.zeros((XX.shape[0], YY.shape[1]))
Z_osqp_wyaw = np.zeros((XX.shape[0], YY.shape[1]))
for elt in res_osqp_wyaw:
    idx, idy = find_nearest(X, elt[1][0], W, elt[1][5])
    Z_osqp_wyaw[idx, idy] = elt[0]

plt.figure()
plt.rc('text', usetex=True)
im = plt.imshow(Z_osqp_vy, cmap=plt.cm.binary, extent=(-2.2, 2.2, -1, 1))
plt.xlabel("Angular Velocity $\dot{p_y} \hspace{2mm} [rad.s^{-1}]$", fontsize=12)
plt.ylabel("Forward Velocity $\dot{p_x} \hspace{2mm} [m.s^{-1}]$", fontsize=12)
plt.title("Viable Operating Regions OSQP", fontsize=14)
"""

plt.show()
