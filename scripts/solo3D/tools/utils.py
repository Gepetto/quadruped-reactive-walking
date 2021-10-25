import numpy as np

def EulerToQuaternion(roll_pitch_yaw):
    roll, pitch, yaw = roll_pitch_yaw
    sr = np.sin(roll / 2.)
    cr = np.cos(roll / 2.)
    sp = np.sin(pitch / 2.)
    cp = np.cos(pitch / 2.)
    sy = np.sin(yaw / 2.)
    cy = np.cos(yaw / 2.)
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return [qx, qy, qz, qw]


def inertiaTranslation(inertia, vect, mass):
    ''' Translation of the inertia matrix using Parallel Axis theorem
    Translation from frame expressed in G to frame expressed in O.
    Args :
    - inertia (Array 3x3): initial inertia matrix expressed in G
    - vect (Array 3x)    : translation vector to apply (OG) (!warning sign)
    - mass (float)       : mass 
    '''
    inertia_o = inertia.copy()

    # Diagonal coeff :
    inertia_o[0, 0] += mass * (vect[1]**2 + vect[2]**2)  # Ixx_o = Ixx_g + m*(yg**2 + zg**2)
    inertia_o[1, 1] += mass * (vect[0]**2 + vect[2]**2)  # Iyy_o = Iyy_g + m*(xg**2 + zg**2)
    inertia_o[2, 2] += mass * (vect[0]**2 + vect[1]**2)  # Izz_o = Iyy_g + m*(xg**2 + zg**2)

    inertia_o[0, 1] += mass * vect[0] * vect[1]  # Ixy_o = Ixy_g + m*xg*yg
    inertia_o[0, 2] += mass * vect[0] * vect[2]  # Ixz_o = Ixz_g + m*xg*zg
    inertia_o[1, 2] += mass * vect[1] * vect[2]  # Iyz_o = Iyz_g + m*yg*zg

    inertia_o[1, 0] = inertia_o[0, 1]  # Ixy_o = Iyx_o
    inertia_o[2, 0] = inertia_o[0, 2]  # Ixz_o = Izx_o
    inertia_o[2, 1] = inertia_o[1, 2]  # Iyz_o = Izy_o

    return inertia_o


def quaternionToRPY(quat):
    """Quaternion (4 x 0) to Roll Pitch Yaw (3 x 1)"""

    qx = quat[0]
    qy = quat[1]
    qz = quat[2]
    qw = quat[3]

    rotateXa0 = 2.0 * (qy * qz + qw * qx)
    rotateXa1 = qw * qw - qx * qx - qy * qy + qz * qz
    rotateX = 0.0

    if (rotateXa0 != 0.0) and (rotateXa1 != 0.0):
        rotateX = np.arctan2(rotateXa0, rotateXa1)

    rotateYa0 = -2.0 * (qx * qz - qw * qy)
    rotateY = 0.0
    if (rotateYa0 >= 1.0):
        rotateY = np.pi / 2.0
    elif (rotateYa0 <= -1.0):
        rotateY = -np.pi / 2.0
    else:
        rotateY = np.arcsin(rotateYa0)

    rotateZa0 = 2.0 * (qx * qy + qw * qz)
    rotateZa1 = qw * qw + qx * qx - qy * qy - qz * qz
    rotateZ = 0.0
    if (rotateZa0 != 0.0) and (rotateZa1 != 0.0):
        rotateZ = np.arctan2(rotateZa0, rotateZa1)

    return np.array([[rotateX], [rotateY], [rotateZ]])
