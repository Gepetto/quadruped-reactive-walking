import cython
import numpy as np
cimport numpy as np

cdef class Foot_trajectory_generator:
    cdef public float h, x1, y1, time_adaptative_disabled
    cdef public list lastCoeffs
    cdef public float coeff_acc_x_lin_a, coeff_acc_x_lin_b, coeff_acc_y_lin_a, coeff_acc_y_lin_b
    
    
    @cython.locals(h=float, adaptative_mode=bint, den=float,
                   cx5=float, dx5=float, cx4=float, dx4=float, cx3=float, dx3=float, cx2=float, dx2=float, cx1=float, dx1=float, cx0=float,
                   cy5=float, dy5=float, cy4=float, dy4=float, cy3=float, dy3=float, cy2=float, dy2=float, cy1=float, dy1=float, cy0=float,
                   Az6=float, Az5=float, Az4=float, Az3=float, ev=float,
                   z0=float, dz0=float, ddz0=float)
    cpdef list get_next_foot(self, float x0, float dx0, float ddx0, float y0, float dy0, float ddy0, float x1, float y1, float t0, float t1, float dt)

    