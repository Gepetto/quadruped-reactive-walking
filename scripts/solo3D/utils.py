import numpy as np
from scipy.spatial import ConvexHull


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

def getAllSurfacesDict_inner(all_surfaces, margin):
    '''
    Computes the inner vertices of the given convex surface, with a margin.
    Args :
    - all_surfaces : Dictionary containing the surface vertices, normal and name.
    - margin : (float) margin in m
    Returns :
    - New dictionnary with inner vertices
    '''

    all_names = []
    surfaces = []
    for name_surface in all_surfaces :
        vertices = order(np.array(all_surfaces.get(name_surface)[0]))
        ineq_inner, ineq_inner_vect, normal = compute_inner_inequalities(vertices, margin)
        vertices_inner = compute_inner_vertices(vertices, ineq_inner, ineq_inner_vect )

        # Save inner vertices
        all_names.append(name_surface)
        surfaces.append((vertices_inner.tolist(), normal.tolist()))

    surfaces_dict = dict(zip(all_names, surfaces))
    return surfaces_dict

def norm(sq):
    """ Computes b=norm
"""
    cr = np.cross(sq[2] - sq[0], sq[1] - sq[0])
    return np.abs(cr / np.linalg.norm(cr))


def order(vertices, method="convexHull"):
    """" Order the array of vertice in counterclock wise using convex Hull method  
"""
    if len(vertices) <= 3:
        return 0
    v = np.unique(vertices, axis=0)
    n = norm(v[:3])
    y = np.cross(n, v[1] - v[0])
    y = y / np.linalg.norm(y)
    c = np.dot(v, np.c_[v[1] - v[0], y])
    if method == "convexHull":
        h = ConvexHull(c)
        vert = v[h.vertices]
    else:
        mean = np.mean(c, axis=0)
        d = c - mean
        s = np.arctan2(d[:, 0], d[:, 1])
        vert = v[np.argsort(s)]

    return vert


def compute_inner_inequalities(vertices, margin):
    """Compute surface inequalities from the vertices list with a margin, update self.ineq_inner, 
    self.ineq_vect_inner
ineq_iner X <= ineq_vect_inner
the last row contains the equality vector
Keyword arguments:
Vertice of the surface  = [[x1 ,y1 ,z1 ]
                            [x2 ,y2 ,z2 ]
                                ...      ]]
                                """
    nb_vert = vertices.shape[0]

    # Computes normal surface
    S_normal = np.cross(vertices[0, :] - vertices[1, :], vertices[0, :] - vertices[2, :])
    if S_normal @ np.array([0., 0., 1.]) < 0.: # Check orientation of the normal
        S_normal = -S_normal

    normal = S_normal / np.linalg.norm(S_normal)

    ineq_inner = np.zeros((nb_vert + 1, 3))
    ineq_vect_inner = np.zeros((nb_vert + 1))

    ineq_inner[-1, :] = normal
    ineq_vect_inner[-1] = -(-normal[0] * vertices[0, 0] - normal[1] * vertices[0, 1] - normal[2] * vertices[0, 2])

    for i in range(nb_vert):

        if i < nb_vert - 1:
            AB = vertices[i, :] - vertices[i + 1, :]
        else:
            AB = vertices[i, :] - vertices[0, :]  # last point of the list with first

        n_plan = np.cross(AB, normal)
        n_plan = n_plan / np.linalg.norm(n_plan)

        # normal = [a,b,c].T
        # To keep the half space in the direction of the normal :
        # ax + by + cz + d >= 0
        # - [a,b,c] * X <= d

        # Take a point M along the normal of the plan, from a distance margin
        # OM = OA + AM = OA + margin*n_plan

        M = vertices[i, :] + margin * n_plan

        # Create the parallel plan that pass trhough M
        ineq_inner[i, :] = -np.array([n_plan[0], n_plan[1], n_plan[2]])
        ineq_vect_inner[i] = -n_plan[0] * M[0] - n_plan[1] * M[1] - n_plan[2] * M[2]

    return ineq_inner, ineq_vect_inner, normal


def compute_inner_vertices( vertices, ineq_inner, ineq_vect_inner):
    """" Compute the list of vertice defining the inner surface :
    update self.vertices_inner = = [[x1 ,y1 ,z1 ]    shape((nb vertice , 3))
                                    [x2 ,y2 ,z2 ]
                                        ...      ]]
    """
    S_inner = []
    nb_vert = vertices.shape[0]

    # P = np.array([a,b,c,d]) , (Plan) ax + by + cz + d = 0
    P_normal = np.zeros(4)
    P_normal[:3] = ineq_inner[-1, :]
    P_normal[-1] = -ineq_vect_inner[-1]

    P1, P2 = np.zeros(4), np.zeros(4)

    for i in range(nb_vert):
        if i < nb_vert - 1:
            P1[:3], P2[:3] = ineq_inner[i, :], ineq_inner[i + 1, :]
            P1[-1], P2[-1] = -ineq_vect_inner[i], -ineq_vect_inner[i + 1]

            A, B = plane_intersect(P1, P2)
            S_inner.append(LinePlaneCollision(P_normal, A, B))
        else:
            P1[:3], P2[:3] = ineq_inner[i, :], ineq_inner[0, :]
            P1[-1], P2[-1] = -ineq_vect_inner[i], -ineq_vect_inner[0]

            A, B = plane_intersect(P1, P2)
            S_inner.append(LinePlaneCollision(P_normal, A, B))

    vertices_inner = np.array(S_inner)
    return vertices_inner


def plane_intersect(P1, P2):
    """ Get the intersection between 2 plan, return Point and direction
:param P1,P2: Plan equalities 
              np.array([a,b,c,d])
              ax + by + cz + d = 0
Returns : 1 point and 1 direction vect of the line of intersection, np.arrays, shape (3,)
"""

    P1_normal, P2_normal = P1[:3], P2[:3]

    aXb_vec = np.cross(P1_normal, P2_normal)

    A = np.array([P1_normal, P2_normal, aXb_vec])
    d = np.array([-P1[3], -P2[3], 0.]).reshape(3, 1)

    # could add np.linalg.det(A) == 0 test to prevent linalg.solve throwing error

    p_inter = np.linalg.solve(A, d).T

    return p_inter[0], (p_inter + aXb_vec)[0]


def LinePlaneCollision(P, A, B, epsilon=1e-6):
    """ Get the intersection point between 1 plane and 1 line
:param P: Plane equality
              np.array([a,b,c,d])
              ax + by + cz + d = 0
param A,B : 2 points defining the line np.arrays, shape(3,)
Returns : 1 point,  np.array, shape (3,)
"""
    plane_normal = P[:3]
    if P[0] == 0:
        if P[1] == 0:
            planePoint = np.array([0, 0, -P[-1] / P[2]])  # a,b = 0 --> z = -d/c
        else:
            planePoint = np.array([0, -P[-1] / P[1], 0])  # a,c = 0 --> y = -d/b
    else:
        planePoint = np.array([-P[-1] / P[0], 0., 0])  # b,c = 0 --> x = -d/a

    rayDirection = A - B
    ndotu = plane_normal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        raise RuntimeError("no intersection or line is within plane")

    w = A - planePoint
    si = -plane_normal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint
    return Psi
