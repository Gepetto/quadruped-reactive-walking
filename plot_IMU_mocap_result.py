
import numpy as np
from matplotlib import pyplot as plt
import pinocchio as pin
import tsid as tsid


def linearly_interpolate_nans(y):
    # Fit a linear regression to the non-nan y values

    # Create X matrix for linreg with an intercept and an index
    X = np.vstack((np.ones(len(y)), np.arange(len(y))))

    # Get the non-NaN values of X and y
    X_fit = X[:, ~np.isnan(y)]
    y_fit = y[~np.isnan(y)].reshape(-1, 1)

    # Estimate the coefficients of the linear regression
    beta = np.linalg.lstsq(X_fit.T, y_fit)[0]

    # Fill in all the nan values using the predicted coefficients
    y.flat[np.isnan(y)] = np.dot(X[:, np.isnan(y)].T, beta)
    return y


def cross3(left, right):
    """Numpy is inefficient for this

    Args:
        left (3x0 array): left term of the cross product
        right (3x0 array): right term of the cross product
    """
    return np.array([[left[1] * right[2] - left[2] * right[1]],
                     [left[2] * right[0] - left[0] * right[2]],
                     [left[0] * right[1] - left[1] * right[0]]])


def BaseVelocityFromKinAndIMU(contactFrameId, model, data, IMU_ang_vel):
    """Estimate the velocity of the base with forward kinematics using a contact point
    that is supposed immobile in world frame

    Args:
        contactFrameId (int): ID of the contact point frame (foot frame)
    """

    frameVelocity = pin.getFrameVelocity(
        model, data, contactFrameId, pin.ReferenceFrame.LOCAL)
    framePlacement = pin.updateFramePlacement(
        model, data, contactFrameId)

    # Angular velocity of the base wrt the world in the base frame (Gyroscope)
    _1w01 = IMU_ang_vel.reshape((3, 1))
    # Linear velocity of the foot wrt the base in the base frame
    _Fv1F = frameVelocity.linear
    # Level arm between the base and the foot
    _1F = framePlacement.translation
    # Orientation of the foot wrt the base
    _1RF = framePlacement.rotation
    # Linear velocity of the base from wrt world in the base frame
    _1v01 = cross3(_1F.ravel(), _1w01.ravel()) - \
        (_1RF @ _Fv1F.reshape((3, 1)))

    return _1v01

#########
# START #
#########


# Load data file
data = np.load("../../Tests_Python/data.npz")

# Store content of data in variables

# From Mocap
mocapPosition = data['mocapPosition']  # Position
mocapOrientationQuat = data['mocapOrientationQuat']  # Orientation as quat
mocapOrientationMat9 = data['mocapOrientationMat9']  # as 3 by 3 matrix
mocapVelocity = data['mocapVelocity']  # Linear velocity
mocapAngularVelocity = data['mocapAngularVelocity']  # Angular velocity

# Fill NaN mocap values with linear interpolation
for i in range(3):
    mocapPosition[:, i] = linearly_interpolate_nans(mocapPosition[:, i])
    mocapVelocity[:, i] = linearly_interpolate_nans(mocapVelocity[:, i])
    mocapAngularVelocity[:, i] = linearly_interpolate_nans(
        mocapAngularVelocity[:, i])

# From IMU
baseOrientation = data['baseOrientation']  # Orientation as quat
baseLinearAcceleration = data['baseLinearAcceleration']  # Linear acceleration
baseAngularVelocity = data['baseAngularVelocity']  # Angular Vel

# IMU is upside down so we have to reorder the data
"""tmp = baseOrientation[:, 0].copy()
baseOrientation[:, 0] = baseOrientation[:, 1].copy()
baseOrientation[:, 1] = tmp
baseOrientation[:, 2] = - baseOrientation[:, 2].copy()
tmp = baseLinearAcceleration[:, 0].copy()
baseLinearAcceleration[:, 0] = baseLinearAcceleration[:, 1].copy()
baseLinearAcceleration[:, 1] = tmp
baseLinearAcceleration[:, 2] = - baseLinearAcceleration[:, 2].copy()
tmp = baseAngularVelocity[:, 0].copy()
baseAngularVelocity[:, 0] = baseAngularVelocity[:, 1].copy()
baseAngularVelocity[:, 1] = tmp
baseAngularVelocity[:, 2] = - baseAngularVelocity[:, 2].copy()"""

# From actuators
torquesFromCurrentMeasurment = data['torquesFromCurrentMeasurment']  # Torques
q_mes = data['q_mes']  # Angular positions
v_mes = data['v_mes']  # Angular velocities

# Creating time vector
Nlist = np.where(mocapPosition[:, 0] == 0.0)[0]
if len(Nlist) > 0:
    N = Nlist[0]
else:
    N = mocapPosition.shape[0]
Tend = N * 0.001
t = np.linspace(0, Tend, N+1, endpoint=True)
t = t[:-1]

# Parameters
dt = 0.001
lwdth = 2

###############
# ORIENTATION #
###############

mocapRPY = np.zeros((N, 3))
imuRPY = np.zeros((N, 3))
for i in range(N):
    mocapRPY[i, :] = pin.rpy.matrixToRpy(mocapOrientationMat9[i, :, :])
    imuRPY[i, :] = pin.rpy.matrixToRpy(pin.Quaternion(
        baseOrientation[i:(i+1), :].transpose()).toRotationMatrix())

fig = plt.figure()
# Roll orientation
ax0 = plt.subplot(2, 1, 1)
plt.plot(t, mocapRPY[:N, 0], "darkorange", linewidth=lwdth)
plt.plot(t, imuRPY[:N, 0], "royalblue", linewidth=lwdth)
plt.ylabel("Roll")
plt.legend(["Mocap", "IMU"], prop={'size': 8})
# Pitch orientation
ax1 = plt.subplot(2, 1, 2, sharex=ax0)
plt.plot(t, mocapRPY[:N, 1], "darkorange", linewidth=lwdth)
plt.plot(t, imuRPY[:N, 1], "royalblue", linewidth=lwdth)
plt.ylabel("Pitch")
plt.xlabel("Time [s]")

###################
# LINEAR VELOCITY #
###################
mocapBaseLinearVelocity = np.zeros((N, 3))
imuBaseLinearVelocity = np.zeros((N, 3))
for i in range(N):
    mocapBaseLinearVelocity[i, :] = ((mocapOrientationMat9[i, :, :]).transpose() @
                                     (mocapVelocity[i:(i+1), :]).transpose()).ravel()
    if i == 0:
        imuBaseLinearVelocity[i, :] = mocapBaseLinearVelocity[0, :]
    else:
        imuBaseLinearVelocity[i, :] = imuBaseLinearVelocity[i -
                                                            1, :] + dt * baseLinearAcceleration[i-1, :]

fig = plt.figure()
# X linear velocity
ax0 = plt.subplot(3, 1, 1)
plt.plot(t, mocapBaseLinearVelocity[:N, 0], "darkorange", linewidth=lwdth)
plt.plot(t, imuBaseLinearVelocity[:N, 0], "royalblue", linewidth=lwdth)
plt.ylabel("$\dot x$ [m/s]")
plt.legend(["Mocap", "IMU"], prop={'size': 8})
# Y linear velocity
ax1 = plt.subplot(3, 1, 2, sharex=ax0)
plt.plot(t, mocapBaseLinearVelocity[:N, 1], "darkorange", linewidth=lwdth)
plt.plot(t, imuBaseLinearVelocity[:N, 1], "royalblue", linewidth=lwdth)
plt.ylabel("$\dot y$ [m/s]")
# Z linear velocity
ax1 = plt.subplot(3, 1, 3, sharex=ax0)
plt.plot(t, mocapBaseLinearVelocity[:N, 2], "darkorange", linewidth=lwdth)
plt.plot(t, imuBaseLinearVelocity[:N, 2], "royalblue", linewidth=lwdth)
plt.ylabel("$\dot z$ [m/s]")
plt.xlabel("Time [s]")

######################
# ANGULAR VELOCITIES #
######################
mocapBaseAngularVelocity = np.zeros(mocapAngularVelocity.shape)
for i in range(N):
    mocapBaseAngularVelocity[i, :] = ((mocapOrientationMat9[i, :, :]).transpose() @
                                      (mocapAngularVelocity[i:(i+1), :]).transpose()).ravel()
fig = plt.figure()
# Angular velocity X subplot
ax0 = plt.subplot(3, 1, 1)
plt.plot(t, mocapBaseAngularVelocity[:N, 0], "darkorange", linewidth=lwdth)
plt.plot(t, baseAngularVelocity[:N, 0], "royalblue", linewidth=lwdth)
plt.ylabel("$\dot \phi$ [rad/s]")
plt.legend(["Mocap", "IMU"], prop={'size': 8})
# Angular velocity Y subplot
ax1 = plt.subplot(3, 1, 2, sharex=ax0)
plt.plot(t, mocapBaseAngularVelocity[:N, 1], "darkorange", linewidth=lwdth)
plt.plot(t, baseAngularVelocity[:N, 1], "royalblue", linewidth=lwdth)
plt.ylabel("$\dot \\theta$ [rad/s]")
# Angular velocity Z subplot
ax2 = plt.subplot(3, 1, 3, sharex=ax0)
plt.plot(t, mocapBaseAngularVelocity[:N, 2], "darkorange", linewidth=lwdth)
plt.plot(t, baseAngularVelocity[:N, 2], "royalblue", linewidth=lwdth)
plt.ylabel("$\dot \psi$ [rad/s]")
plt.xlabel("Time [s]")

#######################
# LINEAR ACCELERATION #
#######################

fig = plt.figure()
# X linear acc
ax0 = plt.subplot(3, 1, 1)
plt.plot(t, baseLinearAcceleration[:N, 0], "royalblue", linewidth=lwdth)
plt.ylabel("$\ddot x$ [m/s^2]")
plt.legend(["IMU"], prop={'size': 8})
# Y linear acc
ax1 = plt.subplot(3, 1, 2, sharex=ax0)
plt.plot(t, baseLinearAcceleration[:N, 1], "royalblue", linewidth=lwdth)
plt.ylabel("$\ddot y$ [m/s^2]")
# Z linear acc
ax1 = plt.subplot(3, 1, 3, sharex=ax0)
plt.plot(t, baseLinearAcceleration[:N, 2], "royalblue", linewidth=lwdth)
plt.ylabel("$\ddot z$ [m/s^2]")
plt.xlabel("Time [s]")

#############
# ACTUATORS #
#############

index = [1, 5, 2, 6, 3, 7, 4, 8]
plt.figure()
for i in range(8):
    if i == 0:
        ax0 = plt.subplot(2, 4, index[i])
    else:
        plt.subplot(2, 4, index[i], sharex=ax0)
    plt.plot(
        t, torquesFromCurrentMeasurment[:N, i], "forestgreen", linewidth=lwdth)

    if (i % 2 == 1):
        plt.xlabel("Time [s]")
    if i <= 1:
        plt.ylabel("Torques [N.m]")

contact_state = np.zeros((N, 4))
margin = 25
treshold = 0.4
for i in range(4):
    state = 0
    front_up = 0
    front_down = 0
    for j in range(N):
        if (state == 0) and (np.abs(torquesFromCurrentMeasurment[j, 2*i+1]) >= treshold):
            state = 1
            front_up = j
        if (state == 1) and (np.abs(torquesFromCurrentMeasurment[j, 2*i+1]) < treshold):
            state = 0
            front_down = j
            l = np.min((front_up+margin, N))
            u = np.max((front_down-margin, 0))
            contact_state[l:u, i] = 1

plt.figure()
for i in range(4):
    if i == 0:
        ax0 = plt.subplot(1, 4, i+1)
    else:
        plt.subplot(1, 4, i+1, sharex=ax0)
    plt.plot(t, torquesFromCurrentMeasurment[:N, 2*i+1])
    plt.plot(t, contact_state[:N, i])
    plt.legend(["Torque", "Estimated contact state"], prop={'size': 8})

"""fig = plt.figure()
# Angular velocity X subplot
ax0 = plt.subplot(3, 1, 1)
plt.plot(t, torquesFromCurrentMeasurment[:N,
                                         0], "forestgreen", linewidth=lwdth)
plt.ylabel("Torques [N.m]")
# Angular velocity Y subplot
ax1 = plt.subplot(3, 1, 2, sharex=ax0)
plt.plot(t, q_mes[:N, 0], "forestgreen", linewidth=lwdth)
plt.ylabel("Angular position [rad]")
# Angular velocity Z subplot
ax2 = plt.subplot(3, 1, 3, sharex=ax0)
plt.plot(t, v_mes[:N, 2], "forestgreen", linewidth=lwdth)
plt.ylabel("Angular velocity [rad/s]")
plt.xlabel("Time [s]")"""


# Set the paths where the urdf and srdf file of the robot are registered
modelPath = "/opt/openrobots/share/example-robot-data/robots"
urdf = modelPath + "/solo_description/robots/solo.urdf"
srdf = modelPath + "/solo_description/srdf/solo.srdf"
vector = pin.StdVec_StdString()
vector.extend(item for item in modelPath)

# Create the robot wrapper from the urdf model (which has no free flyer) and add a free flyer
robot = tsid.RobotWrapper(
    urdf, vector, pin.JointModelFreeFlyer(), False)
model = robot.model()

# Creation of the Invverse Dynamics HQP problem using the robot
# accelerations (base + joints) and the contact forces
invdyn = tsid.InverseDynamicsFormulationAccForce(
    "tsid", robot, False)

# Compute the problem data with a solver based on EiQuadProg
t0 = 0.0
q_FK = np.zeros((15, 1))
q_FK[:7, 0] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
v_FK = np.zeros((14, 1))
invdyn.computeProblemData(t0, q_FK, v_FK)
data = invdyn.data()

indexes = [8, 14, 20, 26]
alpha = 0.98
filteredLinearVelocity = np.zeros((N, 3))
"""for a in range(len(model.frames)):
    print(a)
    print(model.frames[a])"""
FK_lin_vel_log = np.nan * np.zeros((N, 3))
rms_x = []
rms_y = []
rms_z = []
alphas = [0.97]  # [0.01*i for i in range(100)]
i_not_nan = np.where(np.logical_not(np.isnan(mocapBaseLinearVelocity[:, 0])))
i_not_nan = (i_not_nan[0])[(i_not_nan[0] < 9600)]
for alpha in alphas:
    filteredLinearVelocity = np.zeros((N, 3))
    for i in range(N):
        # Update estimator FK model
        q_FK[7:, 0] = q_mes[i, :]  # Position of actuators
        v_FK[6:, 0] = v_mes[i, :]  # Velocity of actuators

        pin.forwardKinematics(model, data, q_FK, v_FK)

        # Get estimated velocity from updated model
        cpt = 0
        vel_est = np.zeros((3, ))
        for j in (np.where(contact_state[i, :] == 1))[0]:
            vel_estimated_baseframe = BaseVelocityFromKinAndIMU(
                indexes[j], model, data, baseAngularVelocity[i, :])

            cpt += 1
            vel_est += vel_estimated_baseframe[:, 0]
        if cpt > 0:
            FK_lin_vel = vel_est / cpt  # average of all feet in contact

            filteredLinearVelocity[i, :] = alpha * (filteredLinearVelocity[i-1, :] + baseLinearAcceleration[i, :] * dt) \
                + (1 - alpha) * FK_lin_vel
            FK_lin_vel_log[i, :] = FK_lin_vel
        else:
            filteredLinearVelocity[i, :] = filteredLinearVelocity[i -
                                                                  1, :] + baseLinearAcceleration[i, :] * dt
    rms_x.append(
        np.sqrt(np.mean(np.square(filteredLinearVelocity[i_not_nan, 0] - mocapBaseLinearVelocity[i_not_nan, 0]))))
    rms_y.append(
        np.sqrt(np.mean(np.square(filteredLinearVelocity[i_not_nan, 1] - mocapBaseLinearVelocity[i_not_nan, 1]))))
    rms_z.append(
        np.sqrt(np.mean(np.square(filteredLinearVelocity[i_not_nan, 2] - mocapBaseLinearVelocity[i_not_nan, 2]))))

plt.figure()
plt.plot(alphas, rms_x)
plt.plot(alphas, rms_y)
plt.plot(alphas, rms_z)
plt.legend(["RMS X", "RMS Y", "RMS Z"], prop={'size': 8})
plt.xlabel("Alpha")
plt.ylabel("RMS erreur en vitesse")

plt.figure()
plt.plot(t, filteredLinearVelocity[:N, 0], linewidth=3)
plt.plot(t, mocapBaseLinearVelocity[:N, 0], linewidth=3)
plt.plot(t, FK_lin_vel_log[:N, 0], color="rebeccapurple", linestyle="--")
"""plt.plot(t, baseLinearAcceleration[:N, 0], linestyle="--")"""
plt.legend(["Filtered", "Mocap", "FK"], prop={'size': 8})
plt.show()
