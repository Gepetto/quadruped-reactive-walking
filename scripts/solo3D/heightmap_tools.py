import numpy as np
import hppfcl
import pickle
import ctypes

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


class MapHeader(ctypes.Structure):
    _fields_ = [
        ("size_x", ctypes.c_int),
        ("size_y", ctypes.c_int),
        ("x_init", ctypes.c_double),
        ("x_end", ctypes.c_double),
        ("y_init", ctypes.c_double),
        ("y_end", ctypes.c_double),
    ]


class Heightmap:

    def __init__(self, n_x, n_y, x_lim, y_lim):
        """
        :param n_x number of samples in x
        :param n_y number of samples in y
        :param x_lim bounds in x
        :param y_lim bounds in y
        """
        self.n_x = n_x
        self.n_y = n_y

        self.x = np.linspace(x_lim[0], x_lim[1], n_x)
        self.y = np.linspace(y_lim[0], y_lim[1], n_y)

        self.z = np.zeros((n_x, n_y))

    def save_pickle(self, filename):
        filehandler = open(filename, 'wb')
        pickle.dump(self, filehandler)

    def save_binary(self, filename):
        """
        Save heightmap matrix as binary file.
        Args :
        - filename (str) : name of the file saved.
        """

        arr_bytes = self.z.astype(ctypes.c_double).tobytes()
        h = MapHeader(self.n_x, self.n_y, self.x[0], self.x[-1], self.y[0], self.y[-1])
        h_bytes = bytearray(h)

        with open(filename, "ab") as f:
            f.truncate(0)
            f.write(h_bytes)
            f.write(arr_bytes)

    def build_from_fit(self, fit):
        """
        Build the heightmap and return it
        For each slot in the grid create a vertical segment and check its collisions with the 
        affordances until one is found
        :param affordances list of affordances
        """
        for i in range(2):
            for j in range(2):
                self.z[i, j] = fit[0] * self.x[i] + fit[1] * self.y[j] + fit[2]

    def build(self, affordances):
        """
        Build the heightmap and return it
        For each slot in the grid create a vertical segment and check its collisions with the 
        affordances until one is found
        :param affordances list of affordances
        """
        last_z = 0.
        for i in range(self.n_x):
            for j in range(self.n_y):
                p1 = np.array([self.x[i], self.y[j], -1.])
                p2 = np.array([self.x[i], self.y[j], 10.])
                segment = np.array([p1, p2])
                fcl_segment = convex(segment, [0, 1, 0])

                intersections = []
                for affordance in affordances:
                    fcl_affordance = affordance_to_convex(affordance)
                    if distance(fcl_affordance, fcl_segment) < 0:
                        for triangle_list in affordance:
                            triangle = [np.array(p) for p in triangle_list]
                            if intersect_line_triangle(segment, triangle):
                                intersections.append(get_point_intersect_line_triangle(segment, triangle)[2])

                if len(intersections) != 0:
                    self.z[i, j] = np.max(np.array(intersections))
                    last_z = self.z[i, j]
                else:
                    self.z[i, j] = last_z

    def plot(self, alpha=1., ax=None):
        """
        Plot the heightmap
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        i = 0
        if alpha != 1.:
            i = 1

        xv, yv = np.meshgrid(self.x, self.y, sparse=False, indexing='ij')
        ax.plot_surface(xv, yv, self.z, color=COLORS[i], alpha=alpha)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_zlim([np.min(self.z), np.max(self.z) + 1.])

        return ax


def affordance_to_convex(affordance):
    """
    Creates a hpp-FCL convex object with an affordance
    """
    vertices = hppfcl.StdVec_Vec3f()
    faces = hppfcl.StdVec_Triangle()
    for triangle_list in affordance:
        [vertices.append(np.array(p)) for p in triangle_list]
        faces.append(hppfcl.Triangle(0, 1, 2))
    return hppfcl.Convex(vertices, faces)


def convex(points, indices):
    """
    Creates a hpp-FCL convex object with a list of points and three indices of the vertices of the
    triangle (or segment)
    """
    vertices = hppfcl.StdVec_Vec3f()
    faces = hppfcl.StdVec_Triangle()
    vertices.extend(points)
    faces.append(hppfcl.Triangle(indices[0], indices[1], indices[2]))
    return hppfcl.Convex(vertices, faces)


def distance(object1, object2):
    """
    Returns the distance between object1 and object2
    """
    guess = np.array([1., 0., 0.])
    support_hint = np.array([0, 0], dtype=np.int32)

    shape = hppfcl.MinkowskiDiff()
    shape.set(object1, object2, hppfcl.Transform3f(), hppfcl.Transform3f())
    gjk = hppfcl.GJK(150, 1e-8)
    gjk.evaluate(shape, guess, support_hint)
    return gjk.distance


# Method to intersect triangle and segment
def signed_tetra_volume(a, b, c, d):
    return np.sign(np.dot(np.cross(b - a, c - a), d - a) / 6.0)


def intersect_line_triangle(segment, triangle):
    s1 = signed_tetra_volume(segment[0], triangle[0], triangle[1], triangle[2])
    s2 = signed_tetra_volume(segment[1], triangle[0], triangle[1], triangle[2])

    if s1 != s2:
        s3 = signed_tetra_volume(segment[0], segment[1], triangle[0], triangle[1])
        s4 = signed_tetra_volume(segment[0], segment[1], triangle[1], triangle[2])
        s5 = signed_tetra_volume(segment[0], segment[1], triangle[2], triangle[0])

        if s3 == s4 and s4 == s5:
            return True
        else:
            return False
    else:
        return False


def get_point_intersect_line_triangle(segment, triangle):
    s1 = signed_tetra_volume(segment[0], triangle[0], triangle[1], triangle[2])
    s2 = signed_tetra_volume(segment[1], triangle[0], triangle[1], triangle[2])

    if s1 != s2:
        s3 = signed_tetra_volume(segment[0], segment[1], triangle[0], triangle[1])
        s4 = signed_tetra_volume(segment[0], segment[1], triangle[1], triangle[2])
        s5 = signed_tetra_volume(segment[0], segment[1], triangle[2], triangle[0])

        if s3 == s4 and s4 == s5:
            n = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            t = np.dot(triangle[0] - segment[0], n) / np.dot(segment[1] - segment[0], n)
            return segment[0] + t * (segment[1] - segment[0])
        else:
            return np.zeros(3)
    else:
        return np.zeros(3)
