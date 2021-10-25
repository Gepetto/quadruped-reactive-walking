import pickle
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import math
from scipy.spatial import ConvexHull


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


class Surface():

    def __init__(self,
                 vertices=np.zeros((3, 3)),
                 vertices_inner=None,
                 ineq=None,
                 ineq_vect=None,
                 normal=None,
                 order_bool=False,
                 margin=0.):
        # Inital surface equations
        self.vertices = vertices
        self.ineq = ineq
        self.ineq_vect = ineq_vect
        # Inner surface equations
        self.vertices_inner = vertices_inner
        self.ineq_inner = ineq
        self.ineq_vect_inner = ineq_vect

        self.normal = normal
        self.order_bool = False
        self.margin = margin

    def compute_all_inequalities(self):
        """" Compute all inequalities, update inner vertices...
    """
        self.compute_inequalities()
        self.compute_inner_inequalities()
        self.compute_inner_vertices()
        return 0

    def norm(self, sq):
        """" norm ...
    """
        cr = np.cross(sq[2] - sq[0], sq[1] - sq[0])
        return np.abs(cr / np.linalg.norm(cr))

    def order(self, method="convexHull"):
        """" Order the array of vertice in counterclock wise using convex Hull method
    """
        if not (self.order_bool):
            if len(self.vertices) <= 3:
                return 0
            v = np.unique(self.vertices, axis=0)
            n = self.norm(v[:3])
            y = np.cross(n, v[1] - v[0])
            y = y / np.linalg.norm(y)
            c = np.dot(v, np.c_[v[1] - v[0], y])
            if method == "convexHull":
                h = ConvexHull(c)
                self.vertices = v[h.vertices]
            else:
                mean = np.mean(c, axis=0)
                d = c - mean
                s = np.arctan2(d[:, 0], d[:, 1])
                self.vertices = v[np.argsort(s)]
        self.order_bool = True
        return 0

    def compute_inequalities(self):
        """Compute surface inequalities from the vertices list
        --> update self.ineq, self.normal,self.ineq_vect
    S x <= d
    the last row contains the equality vector
    Vertice of the surface = [[x1 ,y1 ,z1 ]
                                [x2 ,y2 ,z2 ]
                                   ...      ]]
                                   """
        vert = self.vertices
        nb_vert = vert.shape[0]
        # In .obj, by convention, the direction of the normal AB cross AC
        # is outside the object
        # We know that only the 3 first point are in good size
        S_normal = np.cross(vert[0, :] - vert[1, :], vert[0, :] - vert[2, :])
        self.normal = S_normal / np.linalg.norm(S_normal)

        self.ineq = np.zeros((nb_vert + 1, 3))
        self.ineq_vect = np.zeros((nb_vert + 1))

        self.ineq[-1, :] = self.normal
        self.ineq_vect[-1] = -(-self.normal[0] * vert[0, 0] - self.normal[1] * vert[0, 1] -
                               self.normal[2] * vert[0, 2])

        # Order the whole list (convex hull on 2D order in counterclock wise)
        # Not ordering the list for the previous step is importamt since the 3 first points comes from
        # the .obj, and with the convex, we obtain the proper direction for the normal
        self.order()
        self.order_bool = True
        vert = self.vertices

        for i in range(nb_vert):

            if i < nb_vert - 1:
                AB = vert[i, :] - vert[i + 1, :]
            else:
                AB = vert[i, :] - vert[0, :]  # last point of the list with first

            n_plan = np.cross(AB, self.normal)
            n_plan = n_plan / np.linalg.norm(n_plan)

            # normal = [a,b,c].T
            # To keep the half space in the direction of the normal :
            # ax + by + cz + d >= 0
            # - [a,b,c] * X <= d

            self.ineq[i, :] = -np.array([n_plan[0], n_plan[1], n_plan[2]])
            self.ineq_vect[i] = -n_plan[0] * vert[i, 0] - n_plan[1] * vert[i, 1] - n_plan[2] * vert[i, 2]

        return 0

    def compute_inner_inequalities(self):
        """Compute surface inequalities from the vertices list with a margin, update self.ineq_inner, 
        self.ineq_vect_inner
    ineq_iner X <= ineq_vect_inner
    the last row contains the equality vector
    Keyword arguments:
    Vertice of the surface  = [[x1 ,y1 ,z1 ]
                              [x2 ,y2 ,z2 ]
                                  ...      ]]
                                   """
        if self.ineq is None or self.normal is None or self.ineq_vect is None:
            self.compute_inequalities()

        nb_vert = self.vertices.shape[0]
        self.ineq_inner = np.zeros((nb_vert + 1, 3))
        self.ineq_vect_inner = np.zeros((nb_vert + 1))

        # same normal vector
        self.ineq_inner[-1, :] = self.ineq[-1, :]
        self.ineq_vect_inner[-1] = self.ineq_vect[-1]

        for i in range(nb_vert):

            if i < nb_vert - 1:
                AB = self.vertices[i, :] - self.vertices[i + 1, :]
            else:
                AB = self.vertices[i, :] - self.vertices[0, :]  # last point of the list with first

            n_plan = np.cross(AB, self.normal)
            n_plan = n_plan / np.linalg.norm(n_plan)

            # normal = [a,b,c].T
            # To keep the half space in the direction of the normal :
            # ax + by + cz + d >= 0
            # - [a,b,c] * X <= d

            # Take a point M along the normal of the plan, from a distance margin
            # OM = OA + AM = OA + margin*n_plan

            M = self.vertices[i, :] + self.margin * n_plan

            # Create the parallel plan that pass trhough M
            self.ineq_inner[i, :] = -np.array([n_plan[0], n_plan[1], n_plan[2]])
            self.ineq_vect_inner[i] = -n_plan[0] * M[0] - n_plan[1] * M[1] - n_plan[2] * M[2]

        return 0

    def compute_inner_vertices(self):
        """" Compute the list of vertice defining the inner surface :
        update self.vertices_inner = = [[x1 ,y1 ,z1 ]    shape((nb vertice , 3))
                                        [x2 ,y2 ,z2 ]
                                          ...      ]]
    """
        S_inner = []
        nb_vert = self.vertices.shape[0]

        if self.ineq is None or self.normal is None or self.ineq_vect is None:
            self.compute_inequalities()
            self.compute_inner_inequalities()

        # P = np.array([a,b,c,d]) , (Plan) ax + by + cz + d = 0
        P_normal = np.zeros(4)
        P_normal[:3] = self.ineq[-1, :]
        P_normal[-1] = -self.ineq_vect[-1]

        P1, P2 = np.zeros(4), np.zeros(4)

        for i in range(nb_vert):
            if i < nb_vert - 1:
                P1[:3], P2[:3] = self.ineq_inner[i, :], self.ineq_inner[i + 1, :]
                P1[-1], P2[-1] = -self.ineq_vect_inner[i], -self.ineq_vect_inner[i + 1]

                A, B = plane_intersect(P1, P2)
                S_inner.append(LinePlaneCollision(P_normal, A, B))
            else:
                P1[:3], P2[:3] = self.ineq_inner[i, :], self.ineq_inner[0, :]
                P1[-1], P2[-1] = -self.ineq_vect_inner[i], -self.ineq_vect_inner[0]

                A, B = plane_intersect(P1, P2)
                S_inner.append(LinePlaneCollision(P_normal, A, B))

        self.vertices_inner = np.array(S_inner)
        return 0

    def isPointInside(self, pt, epsilon=10e-4):
        """" Compute if pt is inside the surface
        Params : pt : np.array, shape(3,) , [x,y,z]
                 epsilon : float, for the equality
        return : Bool, inside surface or not
    """

        if abs(np.dot(self.ineq[-1, :], pt) - self.ineq_vect[-1]) < epsilon:
            # inside the plan
            Sx = np.dot(self.ineq[:-1, :], pt)
            return np.sum(Sx <= self.ineq_vect[:-1]) == len(self.vertices)

        else:
            return False

    def isInsideIneq(self, pt, epsilon=10e-4):

        Sx = np.dot(self.ineq[:-1, :], pt)

        return np.sum(Sx <= self.ineq_vect[:-1]) == len(self.vertices)
