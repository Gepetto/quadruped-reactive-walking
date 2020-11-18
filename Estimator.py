# coding: utf8

import numpy as np
import pinocchio as pin
from example_robot_data import load


class ComplementaryFilter:
    """Simple complementary filter

    Args:
        dt (float): time step of the filter [s]
        fc (float): cut frequency of the filter [Hz]
    """

    def __init__(self, dt, fc):

        self.dt = dt

        y = 1 - np.cos(2*np.pi*fc*dt)
        self.alpha = -y+np.sqrt(y*y+2*y)

        self.HP_x = np.zeros(3)
        self.LP_x = np.zeros(3)
        self.filt_x = np.zeros(3)

    def compute(self, x, dx, alpha=None):
        """Run one step of complementary filter

        Args:
            x (N by 1 array): quantity handled by the filter
            dx (N by 1 array): derivative of the quantity
            alpha (float): optional, overwrites the fc of the filter
        """

        # Update alpha value if the user desires it
        if alpha is not None:
            self.alpha = alpha

        # Process high pass filter
        self.HP_x[:] = self.alpha * (self.HP_x + dx * self.dt)

        # Process low pass filter
        self.LP_x[:] = self.alpha * self.LP_x + (1.0 - self.alpha) * x

        # Add both
        self.filt_x[:] = self.HP_x + self.LP_x

        return self.filt_x


class Estimator:
    """State estimator with a complementary filter

    Args:
        dt (float): Time step of the estimator update
        N_simulation (int): maximum number of iterations of the main control loop
        h_init (float): initial height of the robot base
    """

    def __init__(self, dt, N_simulation, h_init=0.22294615):

        # Sample frequency
        self.dt = dt

        # Filtering estimated linear velocity
        fc = 10.0  # Cut frequency
        y = 1 - np.cos(2*np.pi*fc*dt)
        self.alpha_v = -y+np.sqrt(y*y+2*y)

        # Filtering velocities used for security checks
        fc = 6.0
        y = 1 - np.cos(2*np.pi*fc*dt)
        self.alpha_secu = -y+np.sqrt(y*y+2*y)

        # Complementary filters for linear velocity and position
        self.filter_xyz_vel = ComplementaryFilter(dt, 3.0)
        self.filter_xyz_pos = ComplementaryFilter(dt, 500.0)

        # IMU data
        self.IMU_lin_acc = np.zeros((3, ))  # Linear acceleration (gravity debiased)
        self.IMU_ang_vel = np.zeros((3, ))  # Angular velocity (gyroscopes)
        self.IMU_ang_pos = np.zeros((4, ))  # Angular position (estimation of IMU)

        # Forward Kinematics data
        self.FK_lin_vel = np.zeros((3, ))  # Linear velocity
        self.FK_h = h_init  # Default base height of the FK
        self.FK_xyz = np.array([0.0, 0.0, self.FK_h])
        self.xyz_mean_feet = np.zeros(3)
        self.filter_xyz_pos.LP_x[2] = self.FK_h

        # Boolean to disable FK and FG near contact switches
        self.close_from_contact = False
        self.contactStatus = np.zeros(4)
        self.k_since_contact = np.zeros(4)

        # Load the URDF model to get Pinocchio data and model structures
        robot = load('solo12')
        self.data = robot.data.copy()  # for velocity estimation (forward kinematics)
        self.model = robot.model.copy()  # for velocity estimation (forward kinematics)
        self.data_for_xyz = robot.data.copy()  # for position estimation (forward geometry)
        self.model_for_xyz = robot.model.copy()  # for position estimation (forward geometry)

        # High pass linear velocity (filtered IMU velocity)
        self.HP_lin_vel = np.zeros((3, ))
        # Low pass linear velocity (filtered FK velocity)
        self.LP_lin_vel = np.zeros((3, ))
        self.o_filt_lin_vel = np.zeros((3, 1))  # Linear velocity (world frame)
        self.filt_lin_vel = np.zeros((3, ))  # Linear velocity (base frame)
        self.filt_lin_pos = np.zeros((3, ))  # Linear position
        self.filt_ang_vel = np.zeros((3, ))  # Angular velocity
        self.filt_ang_pos = np.zeros((4, ))  # Angular position
        self.q_filt = np.zeros((19, 1))
        self.v_filt = np.zeros((18, 1))
        self.v_secu = np.zeros((12, ))

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

        # Logging matrices
        self.log_v_truth = np.zeros((3, N_simulation))
        self.log_v_est = np.zeros((3, 4, N_simulation))
        self.log_h_est = np.zeros((4, N_simulation))
        self.log_alpha = np.zeros(N_simulation)
        self.log_HP_lin_vel = np.zeros((3, N_simulation))
        self.log_IMU_lin_vel = np.zeros((3, N_simulation))
        self.log_IMU_lin_acc = np.zeros((3, N_simulation))
        self.log_LP_lin_vel = np.zeros((3, N_simulation))
        self.log_FK_lin_vel = np.zeros((3, N_simulation))
        self.log_o_filt_lin_vel = np.zeros((3, N_simulation))
        self.log_filt_lin_vel = np.zeros((3, N_simulation))
        self.log_filt_lin_vel_bis = np.zeros((3, N_simulation))
        self.rotated_FK = np.zeros((3, N_simulation))
        self.k_log = 0

    def get_data_IMU(self, device):
        """Get data from the IMU (linear acceleration, angular velocity and position)

        Args:
            device (object): Interface with the masterboard or the simulation
        """

        # Linear acceleration of the trunk (base frame)
        self.IMU_lin_acc[:] = device.baseLinearAcceleration

        # Angular velocity of the trunk (base frame)
        self.IMU_ang_vel[:] = device.baseAngularVelocity

        # Angular position of the trunk (local frame)
        self.RPY = self.quaternionToRPY(device.baseOrientation)
        if (self.k_log == 0):
            self.offset_yaw_IMU = self.RPY[2]
        self.RPY[2] -= self.offset_yaw_IMU  # Remove initial offset of IMU
        self.IMU_ang_pos[:] = self.EulerToQuaternion([self.RPY[0],
                                                      self.RPY[1],
                                                      self.RPY[2]])

        return 0

    def get_data_joints(self, device):
        """Get the angular position and velocity of the 12 DoF

        Args:
            device (object): Interface with the masterboard or the simulation
        """

        self.actuators_pos[:] = device.q_mes
        self.actuators_vel[:] = device.v_mes

        return 0

    def get_data_FK(self, feet_status):
        """Get data with forward kinematics and forward geometry
        (linear velocity, angular velocity and position)

        Args:
            feet_status (4x0 numpy array): Current contact state of feet
        """

        # Update estimator FK model
        self.q_FK[7:, 0] = self.actuators_pos  # Position of actuators
        self.v_FK[6:, 0] = self.actuators_vel  # Velocity of actuators
        # Position and orientation of the base remain at 0
        # Linear and angular velocities of the base remain at 0

        # Update model used for the forward kinematics
        self.q_FK[3:7, 0] = np.array([0.0, 0.0, 0.0, 1.0])
        pin.forwardKinematics(self.model, self.data, self.q_FK, self.v_FK)
        pin.updateFramePlacements(self.model, self.data)

        # Update model used for the forward geometry
        self.q_FK[3:7, 0] = self.IMU_ang_pos[:]
        pin.forwardKinematics(self.model_for_xyz, self.data_for_xyz, self.q_FK)

        # Get estimated velocity from updated model
        cpt = 0
        vel_est = np.zeros((3, ))
        xyz_est = np.zeros((3, ))
        for i in (np.where(feet_status == 1))[0]:  # Consider only feet in contact
            if self.k_since_contact[i] >= 16:  # Security margin after the contact switch

                # Estimated velocity of the base using the considered foot
                vel_estimated_baseframe = self.BaseVelocityFromKinAndIMU(self.indexes[i])

                # Estimated position of the base using the considered foot
                framePlacement = pin.updateFramePlacement(
                    self.model_for_xyz, self.data_for_xyz, self.indexes[i])
                xyz_estimated = -framePlacement.translation

                # Logging
                self.log_v_est[:, i, self.k_log] = vel_estimated_baseframe[0:3, 0]
                self.log_h_est[i, self.k_log] = xyz_estimated[2]

                # Increment counter and add estimated quantities to the storage variables
                cpt += 1
                vel_est += vel_estimated_baseframe[:, 0]  # Linear velocity
                xyz_est += xyz_estimated  # Position

        # If at least one foot is in contact, we do the average of feet results
        if cpt > 0:
            self.FK_lin_vel = vel_est / cpt
            self.FK_xyz = xyz_est / cpt

        return 0

    def get_xyz_feet(self, feet_status, goals):
        """Get average position of feet in contact with the ground

        Args:
            feet_status (4x0 array): Current contact state of feet
            goals (3x4 array): Target locations of feet on the ground
        """

        cpt = 0
        xyz_feet = np.zeros(3)
        for i in (np.where(feet_status == 1))[0]:  # Consider only feet in contact
            cpt += 1
            xyz_feet += goals[:, i]
        # If at least one foot is in contact, we do the average of feet results
        if cpt > 0:
            self.xyz_mean_feet = xyz_feet / cpt

        return 0

    def run_filter(self, k, feet_status, device, goals, remaining_steps=0):
        """Run the complementary filter to get the filtered quantities

        Args:
            k (int): Number of inv dynamics iterations since the start of the simulation
            feet_status (4x0 array): Current contact state of feet
            device (object): Interface with the masterboard or the simulation
            goals (3x4 array): Target locations of feet on the ground
            remaining_steps (int): Remaining MPC steps for the current gait phase
        """

        # Update IMU data
        self.get_data_IMU(device)

        # Angular position of the trunk
        self.filt_ang_pos[:] = self.IMU_ang_pos

        # Angular velocity of the trunk
        self.filt_ang_vel[:] = self.IMU_ang_vel

        # Update joints data
        self.get_data_joints(device)

        # Update nb of iterations since contact
        self.k_since_contact += feet_status  # Increment feet in stance phase
        self.k_since_contact *= feet_status  # Reset feet in swing phase

        # Update forward kinematics data
        self.get_data_FK(feet_status)

        # Update forward geometry data
        self.get_xyz_feet(feet_status, goals)

        # Tune alpha depending on the state of the gait (close to contact switch or not)
        a = np.ceil(np.max(self.k_since_contact)/10) - 1
        b = remaining_steps
        n = 1  # Nb of steps of margin around contact switch
        v = 0.96  # Minimum alpha value
        c = ((a + b) - 2 * n) * 0.5
        if (a <= (n-1)) or (b <= n):  # If we are close from contact switch
            self.alpha = 1.0  # Only trust IMU data
            self.close_from_contact = True  # Raise flag
        else:
            self.alpha = v + (1 - v) * np.abs(c - (a - n)) / c
            self.close_from_contact = False  # Lower flag

        # Linear velocity of the trunk (base frame)
        self.filt_lin_vel[:] = self.filter_xyz_vel.compute(
            self.FK_lin_vel[:], self.IMU_lin_acc[:], alpha=self.alpha)

        # Taking into account lever arm effect due to the position of the IMU
        """# Get previous base vel wrt world in base frame into IMU frame
        i_filt_lin_vel = self.filt_lin_vel[:] + self.cross3(self._1Mi.translation.ravel(), self.IMU_ang_vel).ravel()

        # Merge IMU base vel wrt world in IMU frame with FK base vel wrt world in IMU frame
        i_merged_lin_vel = self.alpha * (i_filt_lin_vel + self.IMU_lin_acc * self.dt) + (1 - self.alpha) * self.FK_lin_vel

        # Get merged base vel wrt world in IMU frame into base frame
        self.filt_lin_vel[:] = i_merged_lin_vel + self.cross3(-self._1Mi.translation.ravel(), self.IMU_ang_vel).ravel()
        """

        # Linear velocity of the trunk (world frame)
        oRb = pin.Quaternion(np.array([self.IMU_ang_pos]).transpose()).toRotationMatrix()
        self.o_filt_lin_vel[:, 0:1] = oRb @ self.filt_lin_vel.reshape((3, 1))

        # Position of the trunk
        self.filt_lin_pos[:] = self.filter_xyz_pos.compute(
            self.FK_xyz[:] + self.xyz_mean_feet[:], self.o_filt_lin_vel.ravel(), alpha=0.995)

        # Logging
        self.log_alpha[self.k_log] = self.alpha
        self.contactStatus[:] = feet_status  # Save contact status sent to the estimator for logging
        self.log_IMU_lin_acc[:, self.k_log] = self.IMU_lin_acc[:]
        self.log_HP_lin_vel[:, self.k_log] = self.HP_lin_vel[:]
        self.log_LP_lin_vel[:, self.k_log] = self.LP_lin_vel[:]
        self.log_FK_lin_vel[:, self.k_log] = self.FK_lin_vel[:]
        self.log_filt_lin_vel[:, self.k_log] = self.filt_lin_vel[:]
        self.log_o_filt_lin_vel[:, self.k_log] = self.o_filt_lin_vel[:, 0]

        # Output filtered position vector (19 x 1)
        self.q_filt[0:3, 0] = self.filt_lin_pos
        self.q_filt[3:7, 0] = self.filt_ang_pos
        self.q_filt[7:, 0] = self.actuators_pos

        # Output filtered velocity vector (18 x 1)
        self.v_filt[0:3, 0] = (1 - self.alpha_v) * self.v_filt[0:3, 0] + self.alpha_v * self.filt_lin_vel
        self.v_filt[3:6, 0] = self.filt_ang_vel
        self.v_filt[6:, 0] = self.actuators_vel

        # Output filtered actuators velocity for security checks
        self.v_secu[:] = (1 - self.alpha_secu) * self.actuators_vel + self.alpha_secu * self.v_secu[:]

        # Increment iteration counter
        self.k_log += 1

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

        frameVelocity = pin.getFrameVelocity(
            self.model, self.data, contactFrameId, pin.ReferenceFrame.LOCAL)
        framePlacement = pin.updateFramePlacement(
            self.model, self.data, contactFrameId)

        # Angular velocity of the base wrt the world in the base frame (Gyroscope)
        _1w01 = self.IMU_ang_vel.reshape((3, 1))
        # Linear velocity of the foot wrt the base in the foot frame
        _Fv1F = frameVelocity.linear
        # Level arm between the base and the foot
        _1F = np.array(framePlacement.translation)
        # Orientation of the foot wrt the base
        _1RF = framePlacement.rotation
        # Linear velocity of the base wrt world in the base frame
        _1v01 = self.cross3(_1F.ravel(), _1w01.ravel()) - \
            (_1RF @ _Fv1F.reshape((3, 1)))

        # IMU and base frames have the same orientation
        _iv0i = _1v01 + \
            self.cross3(self._1Mi.translation.ravel(), _1w01.ravel())

        return np.array(_1v01)

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

        from matplotlib import pyplot as plt

        NN = self.log_v_est.shape[2]
        avg = np.zeros((3, NN))
        for m in range(NN):
            tmp_cpt = 0
            tmp_sum = np.zeros((3, 1))
            for j in range(4):
                if np.any(np.abs(self.log_v_est[:, j, m]) > 1e-2):
                    tmp_cpt += 1
                    tmp_sum[:, 0] = tmp_sum[:, 0] + \
                        self.log_v_est[:, j, m].ravel()
            if tmp_cpt > 0:
                avg[:, m:(m+1)] = tmp_sum / tmp_cpt

        plt.figure()
        for i in range(3):
            if i == 0:
                ax0 = plt.subplot(3, 1, i+1)
            else:
                plt.subplot(3, 1, i+1, sharex=ax0)
            for j in range(4):
                pass
                # plt.plot(self.log_v_est[i, j, :], linewidth=3)
                # plt.plot(-myController.log_Fv1F[i, j, :], linewidth=3, linestyle="--")
            # plt.plot(avg[i, :], color="rebeccapurple", linewidth=3, linestyle="--")
            plt.plot(self.log_v_truth[i, :], "k", linewidth=3, linestyle="--")
            plt.plot(self.log_alpha, color="k", linewidth=5)
            plt.plot(self.log_HP_lin_vel[i, :],
                     color="orange", linewidth=4, linestyle="--")
            plt.plot(self.log_LP_lin_vel[i, :],
                     color="violet", linewidth=4, linestyle="--")
            plt.plot(
                self.log_FK_lin_vel[i, :], color="royalblue", linewidth=3, linestyle="--")
            plt.plot(
                self.log_filt_lin_vel[i, :], color="darkgoldenrod", linewidth=3, linestyle="--")
            # plt.legend(["FL", "FR", "HL", "HR", "Avg", "Truth", "Filtered", "IMU", "FK"])
            plt.legend(["Truth", "alpha", "HP vel",
                        "LP vel", "FK vel", "Output vel"])
            # plt.xlim([14000, 15000])
        plt.suptitle(
            "Estimation of the linear velocity of the trunk (in base frame)")

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
