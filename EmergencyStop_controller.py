# coding: utf8


########################################################################
#                                                                      #
#                      Control law : tau = 0                           #
#                                                                      #
########################################################################


import numpy as np


########################################################################
#              Class for an Emergency Stop Controller                  #
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

        # Torque controller
        tau = np.zeros((8, 1))

        return tau.flatten()


class controller_12dof:

    def __init__(self):
        self.error = False

    ####################################################################
    #                      Torque Control method                       #
    ####################################################################
    def control(self, qmes12, vmes12, t, solo):

        # Torque controller
        tau = np.zeros((12, 1))

        return tau.flatten()
