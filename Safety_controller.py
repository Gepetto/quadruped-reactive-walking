# coding: utf8


########################################################################
#                                                                      #
#                    Control mode : tau = - D * v^                     #
#                                                                      #
########################################################################


import numpy as np


########################################################################
#                    Class for a Safety Controller                     #
########################################################################

class controller:

    def __init__(self, qdes, vdes):
        self.error = False
        self.qdes = qdes
        self.vdes = vdes

    ####################################################################
    #                      Torque Control method                       #
    ####################################################################
    def control(self, qmes, vmes):

        # D Torque controller,
        D = 0.05
        tau = np.array(-D * vmes)

        # Saturation to limit the maximal torque
        t_max = 1.0
        tau = np.maximum(np.minimum(tau, t_max * np.ones((8, 1))), -t_max * np.ones((8, 1)))

        return tau.flatten()


class controller_12dof:

    def __init__(self):
        self.error = False

    ####################################################################
    #                      Torque Control method                       #
    ####################################################################
    def control(self, qmes12, vmes12, t, solo):

        # D Torque controller,
        D = 0.2
        torques12 = -D * vmes12[6:]

        #torques8 = np.concatenate((torques12[1:3], torques12[4:6], torques12[7:9], torques12[10:12]))

        # Saturation to limit the maximal torque
        t_max = 1.0
        tau = np.maximum(np.minimum(torques12, t_max * np.ones((12, 1))), -t_max * np.ones((12, 1)))

        return tau
