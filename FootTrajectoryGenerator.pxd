import cython
import numpy as np
cimport numpy as np

cdef class Foot_trajectory_generator:
    cdef public double h, x1, y1, time_adaptative_disabled
    cdef public list lastCoeffs_x, lastCoeffs_y
    
    
    @cython.locals(h=double, adaptative_mode=bint, den=double,
                   Ax5=double, Ax4=double, Ax3=double, Ax2=double, Ax1=double, Ax0=double,
                   Ay5=double, Ay4=double, Ay3=double, Ay2=double, Ay1=double, Ay0=double,
                   Az6=double, Az5=double, Az4=double, Az3=double, ev=double,
                   z0=double, dz0=double, ddz0=double)
    cpdef list get_next_foot(self, double x0, double dx0, double ddx0, double y0, double dy0, double ddy0, double x1, double y1, double t0, double t1, double dt)

    