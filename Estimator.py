# coding: utf8

import numpy as np


class Estimator:
    """State estimator with a complementary filter

    Args:
        dt (float): Time step of the MPC
        n_steps (int): Number of time steps in one gait cycle
        k_mpc (int): Number of inv dyn time step for one iteration of the MPC
        T_gait (float): Duration of one period of gait
        multiprocessing (bool): Enable/Disable running the MPC with another process
    """

    def __init__(self, dt):

        # Sample frequency
        self.dt = dt
        self.k = 2 / self.dt  # Gain of bilinear transform

        # Cut frequency of the high-pass filter for the acceleration data of the IMU
        self.f_HP = 50
        self.tau_HP = 1 / (2 * np.pi * self.f_HP)

        # IMU data
        # Linear acceleration (gravity debiased)
        # Angular velocity (gyroscopes)
        # Angular position (estimation of IMU)
        self.IMU_data = np.zeros((12, 1))  # Current data
        self.IMU_data_prev = np.zeros((12, 1))  # Data of the previous time step

        # Forward Kinematics data
        # Linear velocity and position
        # Angular velocity and position
        self.FK_data = np.zeros((12, 1))  # Current data
        self.FK_data_prev = np.zeros((12, 1))  # Data of the previous time step

        # Filtered quantities (output)
        self.filt_data_IMU = np.zeros((12, 1))  # Filtered data (for IMU)
        self.filt_data_FK = np.zeros((12, 1))  # Filtered data (for FK)
        self.filt_data = np.zeros((12, 1))  # Sum of both filtered data

    def get_data_IMU(self):

        return 0

    def get_data_FK(self):

        return 0

    def filter_linear_vel(self):
        """Run the complementary filter to get the estimated linear velocity
        """

        # Filtered data of IMU for the linear velocity using the debiased linear acceleration IMU data
        self.filt_data_IMU[6:9, 0] = ((1 - self.k * self.tau_HP) / (1 + self.k * self.tau_HP)) * self.filt_data_IMU[6:9, 0] \
            + (self.tau_HP / (1 + self.k * self.tau_HP)) * (self.IMU_data[6:9, 0] + self.IMU_data_prev[6:9, 0])

        # Filtered data of FK for the linear velocity using the linear velocity FK data
        self.filt_data_FK[6:9, 0] = ((1 - self.k * self.tau_HP) / (1 + self.k * self.tau_HP)) * self.filt_data_FK[6:9, 0] \
            + (1 / (1 + self.k * self.tau_HP)) * (self.FK_data[6:9, 0] + self.FK_data_prev[6:9, 0])

        # Filtered linear velocity
        self.filt_data[6:9, 0] = self.filt_data_IMU[6:9, 0] + self.filt_data_FK[6:9, 0]

        return 0

    def run_filter(self):

        # Angular position
        self.filt_data[3:6, 0] = self.alpha * self.IMU_ang_pos \
            + (1 - self.alpha) * self.FK_data[3:6, 0]

        # Linear velocity
        self.filt_data[6:9, 0] = self.alpha * (self.filt_data[6:9, 0] + self.IMU_lin_acc * self.dt) \
            + (1 - self.alpha) * self.FK_data[6:9, 0]

        # Angular velocity
        self.filt_data[9:12, 0] = self.alpha * self.IMU_ang_vel \
            + (1 - self.alpha) * self.FK_data[9:12, 0]

        return 0
