# coding: utf8

import numpy as np


class ContactSequencer:
    """Create and update the contact sequence
    """

    def __init__(self, dt):

        # Time step
        self.dt = dt

        # Stance duration
        self.t_stance = 0.16

        # Gait duration
        self.T_gait = 0.32

        # Order of feet is FL, FR, HL, HR
        self.phases = np.array([[0, 0.5, 0.5, 0]])

        # Contact sequence
        self.S = np.zeros((np.int(self.T_gait/self.dt), 4), dtype=np.int)

        # Create contact sequence
        self.createSequence()

    def createSequence(self, t=0):
        """Returns the sequence of footholds from time t to time t+T_gait with step dt and a phase offset.
        The output is a matrix of size N by 4 with N the number of time steps (around T_gait / dt). Each column
        corresponds to one foot with 1 if it touches the ground or 0 otherwise.

        Keyword arguments:
        t -- current time
        dt -- time step
        T_gait -- period of the current gait
        phases -- phase offset for each foot compared to the default sequence
        """

        t_seq = np.linspace(t, t+self.T_gait-self.dt, int(np.round(self.T_gait/self.dt))).reshape((-1, 1))
        phases_seq = (np.hstack((t_seq, t_seq, t_seq, t_seq)) - self.phases * np.float(self.T_gait)) * 2 * np.pi / self.T_gait
        self.S = (np.sin(phases_seq) >= 0).astype(int)

        # To have a four feet stance phase at the start
        # Likely due to numerical effect we don't have it
        self.S[0, :] = np.ones((1, 4), dtype=np.int)
        self.S[np.int(self.S.shape[0]*0.5), :] = np.ones((1, 4), dtype=np.int)

        # For 4 feet stance phase
        # self.S = np.matrix(np.ones(self.S.shape))

        return 0

    def updateSequence(self):

        self.S = np.roll(self.S, -1, axis=0)

        return 0
