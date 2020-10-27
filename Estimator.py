# coding: utf8

import numpy as np
import pinocchio as pin
from matplotlib import pyplot as plt


class Estimator:
    """State estimator with a complementary filter

    Args:
        dt (float): Time step of the estimator update
    """

    def __init__(self, dt, N_simulation):

        # Sample frequency
        self.dt = dt

        # Cut frequency (fc should be < than 1/dt)
        fc = 10.0
        y = 1 - np.cos(2*np.pi*fc*dt)
        self.alpha_v = -y+np.sqrt(y*y+2*y)

        # Filter coefficient (0 < alpha < 1) for IMU with FK
        self.alpha = 0.98  # 0.98  # self.dt * self.fc

        # IMU data
        # Linear acceleration (gravity debiased)
        self.IMU_lin_acc = np.zeros((3, ))
        self.IMU_ang_vel = np.zeros((3, ))  # Angular velocity (gyroscopes)
        # Angular position (estimation of IMU)
        self.IMU_ang_pos = np.zeros((4, ))

        # Forward Kinematics data
        self.FK_lin_vel = np.zeros((3, ))  # Linear velocity
        # self.FK_ang_vel = np.zeros((3, ))  # Angular velocity
        # self.FK_ang_pos = np.zeros((3, ))  # Angular position
        self.FK_h = 0.22294615

        # Filtered quantities (output)
        # self.filt_data = np.zeros((12, ))  # Sum of both filtered data
        self.filt_lin_vel = np.zeros((3, ))  # Linear velocity
        self.filt_lin_pos = np.zeros((3, ))  # Linear position
        self.filt_ang_vel = np.zeros((3, ))  # Angular velocity
        self.filt_ang_pos = np.zeros((4, ))  # Angular position
        self.q_filt = np.zeros((19, 1))
        self.v_filt = np.zeros((18, 1))

        # Various matrices
        self.q_FK = np.zeros((19, 1))
        self.q_FK[:7, 0] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.v_FK = np.zeros((18, 1))
        self.indexes = [10, 18, 26, 34]  #  Indexes of feet frames
        self.actuators_pos = np.zeros((12, ))
        self.actuators_vel = np.zeros((12, ))

        # Transform between the base frame and the IMU frame
        self._1Mi = pin.SE3(pin.Quaternion(np.array([[0.0, 0.0, 0.0, 1.0]]).transpose()),
                            np.array([0.1163, 0.0, 0.02]))

        # Logging
        self.log_v_truth = np.zeros((3, N_simulation))
        self.log_v_est = np.zeros((3, 4, N_simulation))
        # self.log_Fv1F = np.zeros((3, 4, N_simulation))
        # self.log_alpha = np.zeros((3, N_simulation))
        self.log_h_est = np.zeros((4, N_simulation))

        self.log_filt_lin_vel = np.zeros((3, N_simulation))
        self.log_filt_lin_vel_bis = np.zeros((3, N_simulation))
        self.rotated_FK = np.zeros((3, N_simulation))

        self.contactStatus = np.zeros(4)
        self.k_since_contact = np.zeros(4)

        self.k_log = 0

    def get_data_IMU(self, device, q):
        """Get data from the IMU (linear acceleration, angular velocity and position)
        """

        # Linear acceleration of the trunk (PyBullet base frame)
        self.IMU_lin_acc[:] = device.baseLinearAcceleration

        # Angular velocity of the trunk (PyBullet base frame)
        self.IMU_ang_vel[:] = device.baseAngularVelocity

        # Angular position of the trunk (PyBullet local frame)
        if q is not None:
            RPY = self.quaternionToRPY(device.baseOrientation)
            RPY_simu = self.quaternionToRPY(q[3:7])
            self.IMU_ang_pos[:] = self.EulerToQuaternion([RPY[0],
                                                          RPY[1],
                                                          RPY[2]])
        else:
            RPY = self.quaternionToRPY(device.baseOrientation)
            self.IMU_ang_pos[:] = self.EulerToQuaternion([RPY[0],
                                                          RPY[1],
                                                          RPY[2]])

        return 0

    def get_data_joints(self, device):
        """Get the angular position and velocity of the 12 DoF
        """

        self.actuators_pos[:] = device.q_mes
        self.actuators_vel[:] = device.v_mes

        return 0

    def get_data_FK(self, feet_status):
        """Get data from the forward kinematics (linear velocity, angular velocity and position)

        Args:
            feet_status (4x0 numpy array): Current contact state of feet
        """

        # Save contact status sent to the estimator for logging purpose
        self.contactStatus[:] = feet_status

        # Update estimator FK model
        self.q_FK[7:, 0] = self.actuators_pos  # Position of actuators
        self.v_FK[6:, 0] = self.actuators_vel  # Velocity of actuators
        # self.v_FK[0:3, 0] = self.filt_lin_vel[:]  #  Linear velocity of base (in base frame)
        # self.v_FK[3:6, 0] = self.filt_ang_vel[:]  #  Angular velocity of base (in base frame)

        # Update orientation of the robot with IMU data
        self.q_FK[3:7, 0] = self.IMU_ang_pos

        pin.forwardKinematics(self.model, self.data, self.q_FK, self.v_FK)

        # Get estimated velocity from updated model
        cpt = 0
        vel_est = np.zeros((3, ))
        h_est = 0.0
        for i in (np.where(feet_status == 1))[0]:
            if self.k_since_contact[i] >= 3:
                vel_estimated_baseframe, h_estimated = self.BaseVelocityFromKinAndIMU(self.indexes[i])

                self.log_v_est[:, i, self.k_log] = vel_estimated_baseframe[0:3, 0]
                self.log_h_est[i, self.k_log] = h_estimated
                # self.log_Fv1F[:, i, self.k_log] = _Fv1F[0:3]

                cpt += 1
                vel_est += vel_estimated_baseframe[:, 0]
                h_est += h_estimated
        if cpt > 0:
            self.FK_lin_vel = vel_est / cpt
            self.FK_h = h_est / cpt

        self.k_log += 1

        return 0

    def run_filter(self, k, feet_status, device, q=None, data=None, model=None, joystick=None):
        """Run the complementary filter to get the filtered quantities

        Args:
            k (int): Number of inv dynamics iterations since the start of the simulation
            feet_status (4x0 numpy array): Current contact state of feet
        """

        # Retrieve model during first run
        if k == 1:
            self.data = data
            self.model = model

        # Update IMU data
        self.get_data_IMU(device, q)

        # Update joints data
        self.get_data_joints(device)

        # TSID needs to run at least once for forward kinematics
        if k > 0:

            self.k_since_contact += feet_status  # Increment feet in stance phase
            self.k_since_contact *= feet_status  # Reset feet in swing phase

            # Update FK data
            self.get_data_FK(feet_status)

        # Linear position of the trunk
        # TODO: Position estimation
        self.filt_lin_pos[2] = self.FK_h  # 0.2027682

        # Angular position of the trunk (PyBullet local frame)
        """self.filt_data[3:6] = self.alpha * self.IMU_ang_pos \
            + (1 - self.alpha) * self.FK_ang_pos"""
        self.filt_ang_pos[:] = self.IMU_ang_pos

        # Linear velocity of the trunk (PyBullet base frame)
        if k > 0:
            """# Get previous base vel wrt world in base frame into IMU frame
            i_filt_lin_vel = self.filt_lin_vel[:] + self.cross3(self._1Mi.translation.ravel(), self.IMU_ang_vel).ravel()

            # Merge IMU base vel wrt world in IMU frame with FK base vel wrt world in IMU frame
            i_merged_lin_vel = self.alpha * (i_filt_lin_vel + self.IMU_lin_acc * self.dt) + (1 - self.alpha) * self.FK_lin_vel

            # Get merged base vel wrt world in IMU frame into base frame
            self.filt_lin_vel[:] = i_merged_lin_vel + self.cross3(-self._1Mi.translation.ravel(), self.IMU_ang_vel).ravel()"""

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

        # Two vectors that store all data about filtered q and v
        self.q_filt[0:3, 0] = self.filt_lin_pos
        self.q_filt[3:7, 0] = self.filt_ang_pos
        self.q_filt[7:, 0] = self.actuators_pos

        # self.v_filt[0:3, 0] = self.filt_lin_vel
        self.v_filt[0:3, 0] = (1 - self.alpha_v) * self.v_filt[0:3, 0] + self.alpha_v * self.filt_lin_vel

        self.v_filt[3:6, 0] = self.filt_ang_vel
        self.v_filt[6:, 0] = self.actuators_vel

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
        # print("Foot ", contactFrameId, " | ", framePlacement.translation)

        # Angular velocity of the base wrt the world in the base frame (Gyroscope)
        _1w01 = self.IMU_ang_vel.reshape((3, 1))
        # Linear velocity of the foot wrt the base in the foot frame
        _Fv1F = frameVelocity.linear
        # Level arm between the base and the foot
        _1F = np.array(framePlacement.translation)
        # Orientation of the foot wrt the base
        _1RF = framePlacement.rotation
        # Linear velocity of the base wrt world in the base frame
        # print(_1F.ravel())
        # print(_1w01.ravel())
        _1v01 = self.cross3(_1F.ravel(), _1w01.ravel()) - (_1RF @ _Fv1F.reshape((3, 1)))

        # IMU and base frames have the same orientation
        _iv0i = _1v01 + self.cross3(self._1Mi.translation.ravel(), _1w01.ravel())

        return np.array(_1v01), (-framePlacement.translation[2])

    def EulerToQuaternion(self, roll_pitch_yaw):
        roll, pitch, yaw = roll_pitch_yaw
        sr = np.sin(roll/2.)
        cr = np.cos(roll/2.)
        sp = np.sin(pitch/2.)
        cp = np.cos(pitch/2.)
        sy = np.sin(yaw/2.)
        cy = np.cos(yaw/2.)
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        qw = cr * cp * cy + sr * sp * sy
        return [qx, qy, qz, qw]

    def quaternionToRPY(self, quat):
        qx = quat[0]
        qy = quat[1]
        qz = quat[2]
        qw = quat[3]

        rotateXa0 = 2.0*(qy*qz + qw*qx)
        rotateXa1 = qw*qw - qx*qx - qy*qy + qz*qz
        rotateX = 0.0

        if (rotateXa0 != 0.0) and (rotateXa1 != 0.0):
            rotateX = np.arctan2(rotateXa0, rotateXa1)

        rotateYa0 = -2.0*(qx*qz - qw*qy)
        rotateY = 0.0
        if (rotateYa0 >= 1.0):
            rotateY = np.pi/2.0
        elif (rotateYa0 <= -1.0):
            rotateY = -np.pi/2.0
        else:
            rotateY = np.arcsin(rotateYa0)

        rotateZa0 = 2.0*(qx*qy + qw*qz)
        rotateZa1 = qw*qw + qx*qx - qy*qy - qz*qz
        rotateZ = 0.0
        if (rotateZa0 != 0.0) and (rotateZa1 != 0.0):
            rotateZ = np.arctan2(rotateZa0, rotateZa1)

        return np.array([[rotateX], [rotateY], [rotateZ]])

    def plot_graphs(self):

        NN = self.log_v_est.shape[2]
        avg = np.zeros((3, NN))
        for m in range(NN):
            tmp_cpt = 0
            tmp_sum = np.zeros((3, 1))
            for j in range(4):
                if np.any(np.abs(self.log_v_est[:, j, m]) > 1e-2):
                    tmp_cpt += 1
                    tmp_sum[:, 0] = tmp_sum[:, 0] + self.log_v_est[:, j, m].ravel()
            if tmp_cpt > 0:
                avg[:, m:(m+1)] = tmp_sum / tmp_cpt

        plt.figure()
        for i in range(3):
            if i == 0:
                ax0 = plt.subplot(3, 1, i+1)
            else:
                plt.subplot(3, 1, i+1, sharex=ax0)
            for j in range(4):
                plt.plot(self.log_v_est[i, j, :], linewidth=3)
                # plt.plot(-myController.log_Fv1F[i, j, :], linewidth=3, linestyle="--")
            plt.plot(avg[i, :], color="rebeccapurple", linewidth=3, linestyle="--")
            plt.plot(self.log_v_truth[i, :], "k", linewidth=3, linestyle="--")
            plt.plot(self.log_filt_lin_vel[i, :], color="darkgoldenrod", linewidth=3, linestyle="--")
            plt.legend(["FL", "FR", "HL", "HR", "Avg", "Truth", "Filtered"])
            # plt.xlim([14000, 15000])
        plt.suptitle("Estimation of the linear velocity of the trunk (in base frame)")

        """plt.figure()
        for i in range(3):
            plt.subplot(3, 1, i+1)
            plt.plot(self.log_filt_lin_vel[i, :], color="red", linewidth=3)
            plt.plot(self.log_filt_lin_vel_bis[i, :], color="forestgreen", linewidth=3)
            plt.plot(self.rotated_FK[i, :], color="blue", linewidth=3)
            plt.legend(["alpha = 1.0", "alpha = 450/500"])
        plt.suptitle("Estimation of the velocity of the base")"""

        """plt.figure()
        for i in range(3):
            plt.subplot(3, 1, i+1)
            for j in range(4):
                plt.plot(logger.feet_vel[i, j, :], linewidth=3)
        plt.suptitle("Velocity of feet over time")"""

        plt.show(block=True)

        return 0
