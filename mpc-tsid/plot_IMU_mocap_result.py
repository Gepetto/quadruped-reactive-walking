
import glob
import numpy as np
from matplotlib import pyplot as plt
import pinocchio as pin
import tsid as tsid
from IPython import embed

import plot_utils

"""import matplotlib as matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42"""

# Transform between the base frame and the IMU frame
_1Mi = pin.SE3(pin.Quaternion(np.array([[0.0, 0.0, 0.0, 1.0]]).transpose()),
               np.array([0.1163, 0.0, 0.02]))


def EulerToQuaternion(roll_pitch_yaw):
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


def quaternionToRPY(quat):
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
    _1F = np.array(framePlacement.translation)
    # Orientation of the foot wrt the base
    _1RF = framePlacement.rotation
    # Linear velocity of the base from wrt world in the base frame
    _1v01 = cross3(_1F.ravel(), _1w01.ravel()) - (_1RF @ _Fv1F.reshape((3, 1)))

    # IMU and base frames have the same orientation
    _iv0i = _1v01 + cross3(_1Mi.translation.ravel(), _1w01.ravel())

    return _1v01, np.array(_iv0i)

#########
# START #
#########


on_solo8 = False

"""for name in np.sort(glob.glob('./*.npz')):
    print(name)"""
last_date = np.sort(glob.glob('./*.npz'))[-1][-20:]
print(last_date)
last_date = "2020_11_02_13_25.npz"
# last_date = "2020_11_02_13_25.npz"
# Load data file
data = np.load("data_" + last_date)

# Store content of data in variables

# From Mocap
mocapPosition = data['mocapPosition']  # Position
mocapOrientationQuat = data['mocapOrientationQuat']  # Orientation as quat
mocapOrientationMat9 = data['mocapOrientationMat9']  # as 3 by 3 matrix
mocapVelocity = data['mocapVelocity']  # Linear velocity
mocapAngularVelocity = data['mocapAngularVelocity']  # Angular velocity

cheatLinearVelocity = data["baseLinearVelocity"]
cheatForce = data["appliedForce"]
cheatPos = data["dummyPos"]

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
# Acceleration with gravity vector
baseAccelerometer = data['baseAccelerometer']
print(baseAngularVelocity)
# From actuators
torquesFromCurrentMeasurment = data['torquesFromCurrentMeasurment']  # Torques
q_mes = data['q_mes']  # Angular positions
v_mes = data['v_mes']  # Angular velocities

data_control = np.load("data_control_" + last_date)

log_tau_ff = data_control['log_tau_ff']  # Position
log_qdes = data_control['log_qdes']  # Orientation as quat
log_vdes = data_control['log_vdes']  # as 3 by 3 matrix
log_q = data_control['log_q']
log_dq = data_control['log_dq']

# From estimator
if data['estimatorVelocity'] is not None:
    estimatorHeight = data['estimatorHeight']
    estimatorVelocity = data['estimatorVelocity']
    contactStatus = data['contactStatus']
    referenceVelocity = np.round(data['referenceVelocity'], 3)
    log_xfmpc = data['logXFMPC']

# Creating time vector
Nlist = np.where(mocapPosition[:, 0] == 0.0)[0]
if len(Nlist) > 0:
    N = Nlist[0]
else:
    N = mocapPosition.shape[0]
if N == 0:
    N = baseOrientation.shape[0]
N = baseOrientation.shape[0]
Tend = N * 0.002
t = np.linspace(0, Tend, N+1, endpoint=True)
t = t[:-1]

# Parameters
dt = 0.0020
lwdth = 2

#######
#######

imuRPY = np.zeros((N, 3))
vx_ref = np.zeros((N, 1))
vy_ref = np.zeros((N, 1))
vx_est = np.zeros((N, 1))
vy_est = np.zeros((N, 1))
for i in range(N):
    imuRPY[i, :] = pin.rpy.matrixToRpy(pin.Quaternion(
        baseOrientation[i:(i+1), :].transpose()).toRotationMatrix())

    c = np.cos(imuRPY[i, 2])
    s = np.sin(imuRPY[i, 2])
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    v = R.transpose() @ log_xfmpc[i:(i+1), 6:9].transpose()
    # v = pin.Quaternion(baseOrientation[i:(i+1), :].transpose()).toRotationMatrix().transpose() @ log_xfmpc[i:(i+1), 6:9].transpose()
    vx_ref[i] = v[0]
    vy_ref[i] = v[1]

    v_est = log_dq[0:3, i:(i+1)]
    vx_est[i] = v_est[0]
    vy_est[i] = v_est[1]

plot_forces = True

c = ["royalblue", "forestgreen"]
lwdth = 1
velID = 4
# embed()
# HEIGHT / ROLL / PITCH FIGURE
fig1 = plt.figure(figsize=(7, 4))
offset_h = cheatPos[0, 2] - log_xfmpc[0, 2]
# Height subplot
ax0 = plt.subplot(3, 1, 1)
plt.plot(t[:-1], cheatPos[:-1, 2] - offset_h, color=c[0], linewidth=lwdth)
plt.plot(t[:-1], log_q[2, :-1], color="darkgreen", linewidth=lwdth)
plt.plot(t[:-1], log_xfmpc[:-1, 2],
         "darkorange", linewidth=lwdth, linestyle="--")
plt.ylabel("Height [m]", fontsize=14)
plt.legend(["Ground truth", "Estimated", "Command"], prop={'size': 10}, loc=2)


# Roll subplot
ax1 = plt.subplot(3, 1, 2, sharex=ax0)
plt.plot(t[:-1], imuRPY[:-1, 0], color=c[0], linewidth=lwdth)
plt.plot(t[:-1], log_q[3, :-1], color="darkgreen", linewidth=lwdth)
plt.plot(t[:-1], log_xfmpc[:-1, 3], "darkorange",
         linewidth=lwdth, linestyle="--")
plt.ylabel("Roll [rad]", fontsize=14)

if plot_forces:
    ax1bis = ax1.twinx()
    ax1bis.set_ylabel("$F_y$ [N]", color='k', fontsize=14)
    ax1bis.plot(t[:-1], cheatForce[:-1, 1], color="darkviolet",
                linewidth=lwdth, linestyle="--")
    ax1bis.tick_params(axis='y', labelcolor='k')
    ax1bis.legend(["Force"], prop={'size': 10}, loc=1)

# Pitch subplot
ax2 = plt.subplot(3, 1, 3, sharex=ax0)
plt.plot(t[:-1], imuRPY[:-1, 1], color=c[0], linewidth=lwdth)
plt.plot(t[:-1], log_q[4, :-1], color="darkgreen", linewidth=lwdth)
plt.plot(t[:-1], log_xfmpc[:-1, 4], "darkorange",
         linewidth=lwdth, linestyle="--")
plt.xlabel("Time [s]", fontsize=16)
plt.ylabel("Pitch [rad]", fontsize=14)

if plot_forces:
    ax2bis = ax2.twinx()
    ax2bis.set_ylabel("$F_x$ [N]", color='k', fontsize=14)
    ax2bis.plot(t[:-1], cheatForce[:-1, 0], color="darkviolet",
                linewidth=lwdth, linestyle="--")
    ax2bis.tick_params(axis='y', labelcolor='k')
    ax2bis.legend(["Force"], prop={'size': 10}, loc=1)

for ax in [ax0, ax1, ax2]:
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=10)

plt.savefig("/home/palex/Documents/Travail/Article_10_2020/solopython_02_11_2020ter/Figures/H_R_P_vele" +
            str(velID)+ last_date[:-4] +".eps", dpi="figure", bbox_inches="tight")
plt.savefig("/home/palex/Documents/Travail/Article_10_2020/solopython_02_11_2020ter/Figures/H_R_P_vele" +
            str(velID)+ last_date[:-4] +".png", dpi=800, bbox_inches="tight")

# VX / VY / WYAW FIGURE
fig2 = plt.figure(figsize=(7, 4))
# Forward velocity subplot
ax0 = plt.subplot(3, 1, 1)
plt.plot(t[:-1], cheatLinearVelocity[:-1, 0], color=c[0], linewidth=lwdth)
plt.plot(t[:-1], vx_est[:-1], color="darkgreen", linewidth=lwdth)
plt.plot(t[:-1], vx_ref[:-1], "darkorange", linewidth=lwdth, linestyle="--")
plt.ylabel("$\dot x$ [m/s]", fontsize=14)
ax0.legend(["Ground truth", "Estimated", "Command"], prop={'size': 10}, loc=2)

if plot_forces:
    ax0bis = ax0.twinx()
    ax0bis.set_ylabel("$F_x$ [N]", color='k', fontsize=14)
    ax0bis.plot(t[:-1], cheatForce[:-1, 0], color="darkviolet",
                linewidth=lwdth, linestyle="--")
    ax0bis.tick_params(axis='y', labelcolor='k')
    ax0bis.legend(["Force"], prop={'size': 10}, loc=1)

# Lateral velocity subplot
ax1 = plt.subplot(3, 1, 2, sharex=ax0)
plt.plot(t[:-1], cheatLinearVelocity[:-1, 1], color=c[0], linewidth=lwdth)
plt.plot(t[:-1], vy_est[:-1], color="darkgreen", linewidth=lwdth)
plt.plot(t[:-1], vy_ref[:-1], "darkorange", linewidth=lwdth, linestyle="--")
plt.ylabel("$\dot y$ [m/s]", fontsize=14)
#ax1.legend(["Ground truth", "Estimated", "Command"], prop={'size': 10}, loc=2)

if plot_forces:
    ax1bis = ax1.twinx()
    ax1bis.set_ylabel("$F_y$ [N]", color='k', fontsize=14)
    ax1bis.plot(t[:-1], cheatForce[:-1, 1], color="darkviolet",
                linewidth=lwdth, linestyle="--")
    ax1bis.tick_params(axis='y', labelcolor='k')
    ax1bis.legend(["Force"], prop={'size': 10}, loc=1)

# Angular velocity yaw subplot
ax2 = plt.subplot(3, 1, 3, sharex=ax0)
plt.plot(t[:-1], baseAngularVelocity[:-1, 2], color=c[0], linewidth=lwdth)
plt.plot(t[:-1], log_dq[5, :-1], color="darkgreen", linewidth=lwdth)
plt.plot(t[:-1], log_xfmpc[:-1, 11], "darkorange",
         linewidth=lwdth, linestyle="--")
plt.ylabel("$\dot \omega_z$ [rad/s]", fontsize=16)
plt.xlabel("Time [s]", fontsize=14)
#ax2.legend(["Ground truth", "Estimated", "Command"], prop={'size': 10}, loc=2)

for ax in [ax0, ax1, ax2]:
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=10)

plt.savefig("/home/palex/Documents/Travail/Article_10_2020/solopython_02_11_2020ter/Figures/Vx_Vy_Wyaw_vele" +
            str(velID)+ last_date[:-4] + ".eps", dpi="figure", bbox_inches="tight")
plt.savefig("/home/palex/Documents/Travail/Article_10_2020/solopython_02_11_2020ter/Figures/Vx_Vy_Wyaw_vele" +
            str(velID)+ last_date[:-4] + ".png", dpi=800, bbox_inches="tight")

plt.show(block=True)

########
########

# embed()
# X_F_MPC
index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
lgd1 = ["Ctct force X", "Ctct force Y", "Ctct force Z"]
lgd2 = ["FL", "FR", "HL", "HR"]
plt.figure()
for i in range(12):
    if i == 0:
        ax0 = plt.subplot(3, 4, index[i])
    else:
        plt.subplot(3, 4, index[i], sharex=ax0)
    plt.plot(log_xfmpc[:, i], "b", linewidth=2)
    # plt.ylabel(lgd[i])
plt.suptitle("b_xfmpc")

plt.figure()
for i in range(12):
    if i == 0:
        ax0 = plt.subplot(3, 4, index[i])
    else:
        plt.subplot(3, 4, index[i], sharex=ax0)

    h1, = plt.plot(log_xfmpc[:, 12+i], "b", linewidth=5)

    plt.xlabel("Time [s]")
    plt.ylabel(lgd1[i % 3]+" "+lgd2[int(i/3)])

    if (i % 3) == 2:
        plt.ylim([-1.0, 15.0])
    else:
        plt.ylim([-1.5, 1.5])

plt.suptitle("b_xfmpc forces")

# plt.show(block=True)


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

# embed()

###################
# LINEAR VELOCITY #
###################
mocapBaseLinearVelocity = np.zeros((N, 3))
imuBaseLinearVelocity = np.zeros((N, 3))
for i in range(N):
    mocapBaseLinearVelocity[i, :] = ((mocapOrientationMat9[i, :, :]) @
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
if data['estimatorVelocity'] is not None:
    plt.plot(t, estimatorVelocity[:N, 0], "darkgreen", linewidth=lwdth)
plt.plot(t, referenceVelocity[:N, 0], color="darkviolet", linewidth=lwdth)
plt.ylabel("$\dot x$ [m/s]")
plt.legend(["Mocap", "IMU", "Estimator", "Reference"], prop={'size': 8})
# Y linear velocity
ax1 = plt.subplot(3, 1, 2, sharex=ax0)
plt.plot(t, mocapBaseLinearVelocity[:N, 1], "darkorange", linewidth=lwdth)
plt.plot(t, imuBaseLinearVelocity[:N, 1], "royalblue", linewidth=lwdth)
if data['estimatorVelocity'] is not None:
    plt.plot(t, estimatorVelocity[:N, 1], "darkgreen", linewidth=lwdth)
plt.plot(t, referenceVelocity[:N, 1], color="darkviolet", linewidth=lwdth)
plt.ylabel("$\dot y$ [m/s]")
# Z linear velocity
ax1 = plt.subplot(3, 1, 3, sharex=ax0)
plt.plot(t, mocapBaseLinearVelocity[:N, 2], "darkorange", linewidth=lwdth)
plt.plot(t, imuBaseLinearVelocity[:N, 2], "royalblue", linewidth=lwdth)
if data['estimatorVelocity'] is not None:
    plt.plot(t, estimatorVelocity[:N, 2], "darkgreen", linewidth=lwdth)
plt.plot(t, referenceVelocity[:N, 2], color="darkviolet", linewidth=lwdth)
plt.ylabel("$\dot z$ [m/s]")
plt.xlabel("Time [s]")

######################
# ANGULAR VELOCITIES #
######################
mocapBaseAngularVelocity = np.zeros(mocapAngularVelocity.shape)
for i in range(N):
    #mocapBaseAngularVelocity[i, :] = ((mocapOrientationMat9[i, :, :]) @ (mocapAngularVelocity[i:(i+1), :]).transpose()).ravel()
    mocapBaseAngularVelocity[i, :] = (
        mocapAngularVelocity[i:(i+1), :]).transpose().ravel()

fig = plt.figure()
# Angular velocity X subplot
ax0 = plt.subplot(3, 1, 1)
plt.plot(t, mocapBaseAngularVelocity[:N, 0], "darkorange", linewidth=lwdth)
plt.plot(t, baseAngularVelocity[:N, 0], "royalblue", linewidth=lwdth*2)
plt.plot(t, referenceVelocity[:N, 3], color="darkviolet", linewidth=lwdth)
plt.ylabel("$\dot \phi$ [rad/s]")
plt.legend(["Mocap", "IMU", "Reference"], prop={'size': 8})
# Angular velocity Y subplot
ax1 = plt.subplot(3, 1, 2, sharex=ax0)
plt.plot(t, mocapBaseAngularVelocity[:N, 1], "darkorange", linewidth=lwdth)
plt.plot(t, baseAngularVelocity[:N, 1], "royalblue", linewidth=lwdth)
plt.plot(t, referenceVelocity[:N, 4], color="darkviolet", linewidth=lwdth)
plt.ylabel("$\dot \\theta$ [rad/s]")
# Angular velocity Z subplot
ax2 = plt.subplot(3, 1, 3, sharex=ax0)
plt.plot(t, mocapBaseAngularVelocity[:N, 2], "darkorange", linewidth=lwdth)
plt.plot(t, baseAngularVelocity[:N, 2], "royalblue", linewidth=lwdth)
plt.plot(t, referenceVelocity[:N, 5], color="darkviolet", linewidth=lwdth)
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

####################
# JOYSTICK CONTROL #
####################

fig = plt.figure()
# X linear velocity
ax0 = plt.subplot(3, 1, 1)
plt.plot(t, mocapBaseLinearVelocity[:N, 0], "darkorange", linewidth=lwdth)
if data['estimatorVelocity'] is not None:
    plt.plot(t, estimatorVelocity[:N, 0], "darkgreen", linewidth=lwdth)
plt.plot(t, referenceVelocity[:N, 0], color="darkviolet", linewidth=lwdth)
plt.ylabel("$\dot x$ [m/s]")
plt.legend(["Mocap", "Estimator", "Reference"], prop={'size': 8})
# Y linear velocity
ax1 = plt.subplot(3, 1, 2, sharex=ax0)
plt.plot(t, mocapBaseLinearVelocity[:N, 1], "darkorange", linewidth=lwdth)
if data['estimatorVelocity'] is not None:
    plt.plot(t, estimatorVelocity[:N, 1], "darkgreen", linewidth=lwdth)
plt.plot(t, referenceVelocity[:N, 1], color="darkviolet", linewidth=lwdth)
plt.ylabel("$\dot y$ [m/s]")
# Z linear velocity
ax1 = plt.subplot(3, 1, 3, sharex=ax0)
plt.plot(t, mocapBaseAngularVelocity[:N, 2], "darkorange", linewidth=lwdth)
plt.plot(t, baseAngularVelocity[:N, 2], "royalblue", linewidth=lwdth)
plt.plot(t, referenceVelocity[:N, 5], color="darkviolet", linewidth=lwdth)
plt.ylabel("$\dot \psi$ [rad/s]")
plt.xlabel("Time [s]")
plt.suptitle("Tracking of the velocity command sent to the robot")

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
if on_solo8:
    urdf = modelPath + "/solo_description/robots/solo.urdf"
else:
    urdf = modelPath + "/solo_description/robots/solo12.urdf"
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
if on_solo8:
    q_FK = np.zeros((15, 1))
else:
    q_FK = np.zeros((19, 1))
q_FK[:7, 0] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
RPY = quaternionToRPY(baseOrientation.ravel())
IMU_ang_pos = np.zeros(4)
IMU_ang_pos[:] = EulerToQuaternion([RPY[0], RPY[1], 0.0])
q_FK[3:7, 0] = IMU_ang_pos

v_FK = np.zeros((q_FK.shape[0] - 1, 1))
invdyn.computeProblemData(t0, q_FK, v_FK)
data = invdyn.data()

# Feet indexes
if on_solo8:
    indexes = [8, 14, 20, 26]  # solo8
else:
    indexes = [10, 18, 26, 34]  # solo12
alpha = 0.98
filteredLinearVelocity = np.zeros((N, 3))
"""for a in range(len(model.frames)):
    print(a)
    print(model.frames[a])"""
FK_lin_vel_log = np.nan * np.zeros((N, 3))
iFK_lin_vel_log = np.nan * np.zeros((N, 3))
rms_x = []
rms_y = []
rms_z = []
irms_x = []
irms_y = []
irms_z = []
alphas = [0.97]  # [0.01*i for i in range(100)]
i_not_nan = np.where(np.logical_not(np.isnan(mocapBaseLinearVelocity[:, 0])))
i_not_nan = (i_not_nan[0])[(i_not_nan[0] < 9600)]
for alpha in alphas:
    print(alpha)
    filteredLinearVelocity = np.zeros((N, 3))
    ifilteredLinearVelocity = np.zeros((N, 3))
    for i in range(N):
        # Update estimator FK model
        q_FK[7:, 0] = q_mes[i, :]  # Position of actuators
        v_FK[6:, 0] = v_mes[i, :]  # Velocity of actuators

        pin.forwardKinematics(model, data, q_FK, v_FK)

        # Get estimated velocity from updated model
        cpt = 0
        vel_est = np.zeros((3, ))
        ivel_est = np.zeros((3, ))
        for j in (np.where(contactStatus[i, :] == 1))[0]:
            vel_estimated_baseframe, _iv0i = BaseVelocityFromKinAndIMU(
                indexes[j], model, data, baseAngularVelocity[i, :])

            cpt += 1
            vel_est += vel_estimated_baseframe[:, 0]
            ivel_est = ivel_est + _iv0i.ravel()
        if cpt > 0:
            FK_lin_vel = vel_est / cpt  # average of all feet in contact
            iFK_lin_vel = ivel_est / cpt

            filteredLinearVelocity[i, :] = alpha * (filteredLinearVelocity[i-1, :] +
                                                    baseLinearAcceleration[i, :] * dt) + (1 - alpha) * FK_lin_vel
            FK_lin_vel_log[i, :] = FK_lin_vel

            # Get previous base vel wrt world in base frame into IMU frame
            i_filt_lin_vel = ifilteredLinearVelocity[i-1, :] + \
                cross3(_1Mi.translation.ravel(),
                       baseAngularVelocity[i, :]).ravel()

            # Merge IMU base vel wrt world in IMU frame with FK base vel wrt world in IMU frame
            i_merged_lin_vel = alpha * \
                (i_filt_lin_vel +
                 baseLinearAcceleration[i, :] * dt) + (1 - alpha) * iFK_lin_vel.ravel()
            """print("##")
            print(filteredLinearVelocity[i, :])
            print(i_merged_lin_vel)"""
            # Get merged base vel wrt world in IMU frame into base frame
            ifilteredLinearVelocity[i, :] = np.array(
                i_merged_lin_vel + cross3(-_1Mi.translation.ravel(), baseAngularVelocity[i, :]).ravel())
            #print(ifilteredLinearVelocity[i, :])
            """if np.array_equal(filteredLinearVelocity[i, :], ifilteredLinearVelocity[i, :]):
                print("Same values")
                
            else:
                print("Different")
                print(filteredLinearVelocity[i, :])
                print(ifilteredLinearVelocity[i, :])"""

        else:
            filteredLinearVelocity[i, :] = filteredLinearVelocity[i -
                                                                  1, :] + baseLinearAcceleration[i, :] * dt

            # Get previous base vel wrt world in base frame into IMU frame
            i_filt_lin_vel = ifilteredLinearVelocity[i-1, :] + \
                cross3(_1Mi.translation.ravel(),
                       baseAngularVelocity[i, :]).ravel()
            # Merge IMU base vel wrt world in IMU frame with FK base vel wrt world in IMU frame
            i_merged_lin_vel = i_filt_lin_vel + \
                baseLinearAcceleration[i, :] * dt
            # Get merged base vel wrt world in IMU frame into base frame
            ifilteredLinearVelocity[i, :] = i_merged_lin_vel + \
                cross3(-_1Mi.translation.ravel(),
                       baseAngularVelocity[i, :]).ravel()

    rms_x.append(
        np.sqrt(np.mean(np.square(filteredLinearVelocity[i_not_nan, 0] - mocapBaseLinearVelocity[i_not_nan, 0]))))
    rms_y.append(
        np.sqrt(np.mean(np.square(filteredLinearVelocity[i_not_nan, 1] - mocapBaseLinearVelocity[i_not_nan, 1]))))
    rms_z.append(
        np.sqrt(np.mean(np.square(filteredLinearVelocity[i_not_nan, 2] - mocapBaseLinearVelocity[i_not_nan, 2]))))
    irms_x.append(
        np.sqrt(np.mean(np.square(ifilteredLinearVelocity[i_not_nan, 0] - mocapBaseLinearVelocity[i_not_nan, 0]))))
    irms_y.append(
        np.sqrt(np.mean(np.square(ifilteredLinearVelocity[i_not_nan, 1] - mocapBaseLinearVelocity[i_not_nan, 1]))))
    irms_z.append(
        np.sqrt(np.mean(np.square(ifilteredLinearVelocity[i_not_nan, 2] - mocapBaseLinearVelocity[i_not_nan, 2]))))

plt.figure()
plt.plot(alphas, rms_x)
plt.plot(alphas, rms_y)
plt.plot(alphas, rms_z)
plt.plot(alphas, irms_x)
plt.plot(alphas, irms_y)
plt.plot(alphas, irms_z)
plt.legend(["RMS X", "RMS Y", "RMS Z", "New RMS X",
            "New RMS Y", "New RMS Z"], prop={'size': 8})
plt.xlabel("Alpha")
plt.ylabel("RMS erreur en vitesse")

fc = 10
y = 1 - np.cos(2*np.pi*fc*dt)
alpha_v = -y+np.sqrt(y*y+2*y)
lowpass_ifilteredLinearVelocity = np.zeros(ifilteredLinearVelocity.shape)
lowpass_ifilteredLinearVelocity[0, :] = ifilteredLinearVelocity[0, :]
for k in range(1, N):
    lowpass_ifilteredLinearVelocity[k, :] = (
        1 - alpha_v) * lowpass_ifilteredLinearVelocity[k-1, :] + alpha_v * ifilteredLinearVelocity[k, :]


plt.figure()
plt.plot(t, filteredLinearVelocity[:N, 0], linewidth=3)
plt.plot(t, ifilteredLinearVelocity[:N, 0], linewidth=3, linestyle="--")
plt.plot(t, mocapBaseLinearVelocity[:N, 0], linewidth=3)
plt.plot(
    t, lowpass_ifilteredLinearVelocity[:N, 0], color="darkviolet", linewidth=3)
# plt.plot(t, FK_lin_vel_log[:N, 0], color="darkviolet", linestyle="--")
"""plt.plot(t, baseLinearAcceleration[:N, 0], linestyle="--")"""
plt.legend(["Filtered", "New Filtered", "Mocap"], prop={'size': 8})
# plt.show()


data_control = np.load("data_control_" + last_date)

log_tau_ff = data_control['log_tau_ff']  # Position
log_qdes = data_control['log_qdes']  # Orientation as quat
log_vdes = data_control['log_vdes']  # as 3 by 3 matrix
log_q = data_control['log_q']

index12 = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
plt.figure()
for i in range(12):
    if i == 0:
        ax0 = plt.subplot(3, 4, index12[i])
    else:
        plt.subplot(3, 4, index12[i], sharex=ax0)
    plt.plot(t, log_qdes[i, :], color='r', linewidth=lwdth)
    plt.plot(t, q_mes[:, i], color='b', linewidth=lwdth)
    plt.legend(["Qdes", "Qmes"], prop={'size': 8})

plt.figure()
for i in range(12):
    if i == 0:
        ax0 = plt.subplot(3, 4, index12[i])
    else:
        plt.subplot(3, 4, index12[i], sharex=ax0)
    plt.plot(t, log_vdes[i, :], color='r', linewidth=lwdth)
    plt.plot(t, v_mes[:, i], color='b', linewidth=lwdth)
    plt.legend(["Vdes", "Vmes"], prop={'size': 8})

# Z linear velocity
plt.figure()
for i in range(12):
    if i == 0:
        ax0 = plt.subplot(3, 4, index12[i])
    else:
        plt.subplot(3, 4, index12[i], sharex=ax0)
    plt.plot(t, log_tau_ff[i, :], color='r', linewidth=lwdth)
    plt.plot(t, torquesFromCurrentMeasurment[:, i], color='b', linewidth=lwdth)
    plt.legend(["Tau_FF", "TAU"], prop={'size': 8})

plt.show(block=True)
