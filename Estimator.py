# coding: utf8

import numpy as np
import pinocchio as pin
import pybullet as pyb


class Estimator:
    """State estimator with a complementary filter

    Args:
        dt (float): Time step of the estimator update
    """

    def __init__(self, dt, N_simulation):

        # Sample frequency
        self.dt = dt

        # Cut frequency (fc should be < than 1/dt)
        self.fc = 400

        # Filter coefficient (0 < alpha < 1)
        self.alpha = self.dt * self.fc

        # IMU data
        self.IMU_lin_acc = np.zeros((3, ))  # Linear acceleration (gravity debiased)
        self.IMU_ang_vel = np.zeros((3, ))  # Angular velocity (gyroscopes)
        self.IMU_ang_pos = np.zeros((4, ))  # Angular position (estimation of IMU)

        # Forward Kinematics data
        self.FK_lin_vel = np.zeros((3, ))  # Linear velocity
        # self.FK_ang_vel = np.zeros((3, ))  # Angular velocity
        # self.FK_ang_pos = np.zeros((3, ))  # Angular position

        # Filtered quantities (output)
        # self.filt_data = np.zeros((12, ))  # Sum of both filtered data
        self.filt_lin_vel = np.zeros((3, ))  # Linear velocity
        self.filt_ang_vel = np.zeros((3, ))  # Angular velocity
        self.filt_ang_pos = np.zeros((4, ))  # Angular position

        # Various matrices
        self.q_FK = np.zeros((19, 1))
        self.q_FK[:7, 0] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.v_FK = np.zeros((18, 1))
        self.indexes = [10, 18, 26, 34]  #  Indexes of feet frames
        self.actuators_pos = np.zeros((12, ))
        self.actuators_vel = np.zeros((12, ))

        # Logging
        self.log_v_truth = np.zeros((3, N_simulation))
        self.log_v_est = np.zeros((3, 4, N_simulation))
        #self.log_Fv1F = np.zeros((3, 4, N_simulation))
        #self.log_alpha = np.zeros((3, N_simulation))

        self.log_filt_lin_vel = np.zeros((3, N_simulation))
        self.log_filt_lin_vel_bis = np.zeros((3, N_simulation))
        self.rotated_FK = np.zeros((3, N_simulation))

        self.k_log = 0

        self.prev = np.zeros((3, ))

    def get_data_IMU(self, robotId):
        """Get data from the IMU (linear acceleration, angular velocity and position)
        """

        # Position and orientation of the trunk (PyBullet world frame)
        baseState = pyb.getBasePositionAndOrientation(robotId)

        # Linear and angular velocity of the trunk (PyBullet world frame)
        baseVel = pyb.getBaseVelocity(robotId)

        # Transform from world frame to base frame in PyBullet world
        self.quat_oMb = np.array(baseState[1])
        rot_oMb = pin.Quaternion(np.array([baseState[1]]).transpose()).toRotationMatrix()
        self.oMb = pin.SE3(rot_oMb, np.array([baseState[0]]).transpose())

        # Get roll pitch yaw orientation in PyBullet world then remove yaw component
        # self.oMb = pin.SE3(pin.Quaternion(np.array([baseState[1]]).transpose()).toRotationMatrix(), np.zeros((3, 1)))
        """RPY_pyb = pin.rpy.matrixToRpy(pin.Quaternion(np.array([baseState[1]]).transpose()).toRotationMatrix())
        self.IMU_ang_pos[:] = np.array(pyb.getQuaternionFromEuler(np.array([RPY_pyb[0],
                                                                            RPY_pyb[1],
                                                                            0.0])))
        rot = pin.SE3(pin.utils.rotate('z', RPY_pyb[2]), np.zeros((3, 1)))
        tmp = rot.rotation.transpose() @ np.array([baseVel[0]]).transpose()"""

        # Position of the trunk (PyBullet world frame)
        self.cheat_lin_pos = np.array(baseState[0])

        # Linear acceleration of the trunk (PyBullet base frame)
        tmp = (self.oMb.rotation.transpose() @ np.array([baseVel[0]]).transpose()).ravel()
        self.IMU_lin_acc[:] = (tmp.ravel() - self.prev) / self.dt
        self.prev[:] = tmp.ravel()

        # Angular velocity of the trunk (PyBullet base frame)
        self.IMU_ang_vel[:] = (self.oMb.rotation.transpose() @ np.array([baseVel[1]]).transpose()).ravel()

        # Angular position of the trunk (PyBullet local frame)
        RPY_pyb = pin.rpy.matrixToRpy(rot_oMb)
        self.IMU_ang_pos[:] = np.array(pyb.getQuaternionFromEuler(np.array([RPY_pyb[0], RPY_pyb[1], 0.0])))

        return 0

    def get_data_joints(self, robotId):
        """Get the angular position and velocity of the 12 DoF
        """

        revoluteJointIndices = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
        jointStates = pyb.getJointStates(robotId, revoluteJointIndices)  # State of all joints

        self.actuators_pos[:] = np.array([state[0] for state in jointStates])
        self.actuators_vel[:] = np.array([state[1] for state in jointStates])

        return 0

    def get_data_FK(self, feet_status):
        """Get data from the forward kinematics (linear velocity, angular velocity and position)

        Args:
            feet_status (4x0 numpy array): Current contact state of feet
        """

        # Update estimator FK model
        self.q_FK[7:, 0] = self.actuators_pos  # Position of actuators
        self.v_FK[6:, 0] = self.actuators_vel  # Velocity of actuators
        # self.v_FK[0:3, 0] = self.filt_lin_vel[:]  #  Linear velocity of base (in base frame)
        # self.v_FK[3:6, 0] = self.filt_ang_vel[:]  #  Angular velocity of base (in base frame)

        pin.forwardKinematics(self.model, self.data, self.q_FK, self.v_FK)

        # Get estimated velocity from updated model
        cpt = 0
        vel_est = np.zeros((3, ))
        for i in (np.where(feet_status == 1))[0]:
            vel_estimated_baseframe = self.BaseVelocityFromKinAndIMU(self.indexes[i])

            self.log_v_est[:, i, self.k_log] = vel_estimated_baseframe[0:3, 0]
            # self.log_Fv1F[:, i, self.k_log] = _Fv1F[0:3]

            cpt += 1
            vel_est += vel_estimated_baseframe[:, 0]
        self.FK_lin_vel = vel_est / cpt

        self.k_log += 1

        return 0

    def run_filter(self, k, feet_status, robotId, data=None, model=None):
        """Run the complementary filter to get the filtered quantities

        Args:
            k (int): Number of inv dynamics iterations since the start of the simulation
            feet_status (4x0 numpy array): Current contact state of feet
        """

        # TSID needs to run at least once
        if k == 0:
            return 0

        # Retrieve model during first run
        if k == 1:
            self.data = data
            self.model = model

        # Update IMU data
        self.get_data_IMU(robotId)

        # Update joints data
        self.get_data_joints(robotId)

        # Update FK data
        self.get_data_FK(feet_status)

        # Angular position of the trunk (PyBullet local frame)
        """self.filt_data[3:6] = self.alpha * self.IMU_ang_pos \
            + (1 - self.alpha) * self.FK_ang_pos"""
        self.filt_ang_pos[:] = self.IMU_ang_pos

        # Linear velocity of the trunk (PyBullet base frame)
        self.filt_lin_vel[:] = self.alpha * (self.filt_lin_vel[:] + self.IMU_lin_acc * self.dt) \
            + (1 - self.alpha) * self.FK_lin_vel

        self.log_filt_lin_vel[:, self.k_log] = self.filt_lin_vel[:]
        """beta = 475 / 500
        self.log_filt_lin_vel_bis[:, self.k_log] = beta * (self.filt_lin_vel[:] + self.IMU_lin_acc * self.dt) \
            + (1 - beta) * (self.oMb.rotation @ np.array([self.FK_lin_vel]).transpose()).ravel()
        self.rotated_FK[:, self.k_log] = (self.oMb.rotation @ np.array([self.FK_lin_vel]).transpose()).ravel()

        tmp = (self.filt_lin_vel[:] + self.IMU_lin_acc * self.dt)
        self.log_alpha[:, self.k_log] = beta * (self.oMb.rotation.transpose() @ np.array([tmp]).transpose()).ravel() \
            + (1 - beta) * (np.array([self.FK_lin_vel]).transpose()).ravel()"""

        # Angular velocity of the trunk (PyBullet base frame)
        """self.filt_data[9:12] = self.alpha * self.IMU_ang_vel \
            + (1 - self.alpha) * self.FK_ang_vel"""
        self.filt_ang_vel[:] = self.IMU_ang_vel

        return 0

    def cross3(self, left, right):
        """Numpy is inefficient for this

        Args:
            left (3x0 array): left term of the cross product
            right (3x0 array): right term of the cross product
        """
        return np.array([[left[1] * right[2] - left[2] * right[1]],
                         [left[2] * right[0] - left[0] * right[2]],
                         [left[0] * right[1] - left[1] * right[0]]])

    def BaseVelocityFromKinAndIMU(self, contactFrameId):
        """Estimate the velocity of the base with forward kinematics using a contact point
        that is supposed immobile in world frame

        Args:
            contactFrameId (int): ID of the contact point frame (foot frame)
        """

        frameVelocity = pin.getFrameVelocity(self.model, self.data, contactFrameId, pin.ReferenceFrame.LOCAL)
        framePlacement = pin.updateFramePlacement(self.model, self.data, contactFrameId)

        # Angular velocity of the base wrt the world in the base frame (Gyroscope)
        _1w01 = self.IMU_ang_vel.reshape((3, 1))
        # Linear velocity of the foot wrt the base in the base frame
        _Fv1F = frameVelocity.linear
        # Level arm between the base and the foot
        _1F = framePlacement.translation
        # Orientation of the foot wrt the base
        _1RF = framePlacement.rotation
        # Linear velocity of the base from wrt world in the base frame
        _1v01 = self.cross3(_1F.ravel(), _1w01.ravel()) - (_1RF @ _Fv1F.reshape((3, 1)))

        return _1v01
