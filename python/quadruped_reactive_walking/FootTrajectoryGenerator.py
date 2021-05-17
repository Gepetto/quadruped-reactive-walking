# coding: utf8

import numpy as np


class FootTrajectoryGenerator:
    """A foot trajectory generator that handles the generation of a 3D trajectory
    with a 5th order polynomial to lead each foot from its location at the start of
    its swing phase to its final location that has been decided by the FootstepPlanner

    :param shoulders: A 2 by 4 numpy array, the position of shoulders in local frame
    :param dt: A float, time step of the contact sequence
    """

    def __init__(self, dt):

        # Position of shoulders in local frame
        self.shoulders = np.array(
            [[0.1946, 0.1946, -0.1946, -0.1946], [0.14695, -0.14695, 0.14695, -0.14695]])

        # Time step of the trajectory generator
        self.dt = dt

        # Desired (x, y) position of footsteps without lock mechanism before impact
        # Received from the FootstepPlanner
        # self.footsteps = self.shoulders.copy()

        # Desired (x, y) position of footsteps with lock mechanism before impact
        R = np.array([[0.0, -1.0], [1.0, 0.0]])
        self.footsteps_lock = R @ self.shoulders.copy()

        # Desired footsteps with lock in world frame for visualisation purpose
        self.footsteps_lock_world = self.footsteps_lock.copy()

        # Desired position, velocity and acceleration of feet in 3D, in local frame
        self.desired_pos = np.vstack((R @ self.shoulders, np.zeros((1, 4))))
        self.desired_vel = np.zeros(self.desired_pos.shape)
        self.desired_acc = np.zeros(self.desired_pos.shape)

        # Desired 3D position in world frame for visualisation purpose
        self.desired_pos_world = self.desired_pos.copy()

        # Maximum height at which the robot should lift its feet during swing phase
        self.max_height_feet = 0.02

        # Lock target positions of footholds before touchdown
        self.t_lock_before_touchdown = 0.01

        # Foot trajectory generator objects (one for each foot)
        self.ftgs = [Foot_trajectory_generator(
            self.max_height_feet, self.t_lock_before_touchdown) for i in range(4)]

        # Initialization of ftgs objects
        for i in range(4):
            self.ftgs[i].x1 = self.desired_pos[0, i]
            self.ftgs[i].y1 = self.desired_pos[1, i]

        self.flag_initialisation = False

    def update_desired_feet_pos(self, sequencer, fstep_planner, mpc):

        # Initialisation of rotation from local frame to world frame
        c, s = np.cos(mpc.q_w[5, 0]), np.sin(mpc.q_w[5, 0])
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        # Initialisation of trajectory parameters
        x0 = 0.0
        dx0 = 0.0
        ddx0 = 0.0

        y0 = 0.0
        dy0 = 0.0
        ddy0 = 0.0

        z0 = 0.0
        dz0 = 0.0
        ddz0 = 0.0

        # The swing phase lasts T seconds
        t1 = sequencer.T_gait - sequencer.t_stance

        # For each foot
        for i in range(4):
            # Time remaining before touchdown
            index = (np.where(sequencer.S[:, i] == True))[0][0]
            t0 = t1 - index * sequencer.dt

            # Current position of the foot
            x0 = self.desired_pos[0, i]
            y0 = self.desired_pos[1, i]

            # Target position of the foot
            x1 = fstep_planner.footsteps[0, i]
            y1 = fstep_planner.footsteps[1, i]

            # Update if the foot is in swing phase or is going to leave the ground
            if ((sequencer.S[0, i] == True) and (sequencer.S[1, i] == False)):
                t0 = 0

            if (t0 != t1) and (t0 != (t1 - sequencer.dt)):

                # Get desired 3D position
                [x0, dx0, ddx0,  y0, dy0, ddy0,  z0, dz0, ddz0, gx1, gy1] = (self.ftgs[i]).get_next_foot(
                    x0, self.desired_vel[0, i], self.desired_acc[0, i],
                    y0, self.desired_vel[1, i], self.desired_acc[1, i],
                    x1, y1, t0,  t1, self.dt)

                if self.flag_initialisation:
                    # Retrieve result in terms of position, velocity and acceleration
                    self.desired_pos[:, i] = np.array([x0, y0, z0])
                    self.desired_vel[:, i] = np.array([dx0, dy0, dz0])
                    self.desired_acc[:, i] = np.array([ddx0, ddy0, ddz0])

                    # Update target position of the foot with lock
                    self.footsteps_lock[:, i] = np.array([gx1, gy1])

                    # Update variables in world frame
                    self.desired_pos_world[:, i:(i+1)] = np.vstack((mpc.q_w[0:2, 0:1], np.zeros((1, 1)))) + \
                        np.dot(R, self.desired_pos[:, i:(i+1)])
                    self.footsteps_lock_world[:, i:(i+1)] = mpc.q_w[0:2, 0:1] + \
                        np.dot(R[0:2, 0:2], self.footsteps_lock[:, i:(i+1)])
            else:
                self.desired_vel[:, i] = np.array([0.0, 0.0, 0.0])
                self.desired_acc[:, i] = np.array([0.0, 0.0, 0.0])

        if not self.flag_initialisation:
            self.flag_initialisation = True

        return 0

    def update_frame(self, vel):
        """As we are working in local frame, the footsteps drift backwards
        if the trunk is moving forwards as footsteps are not supposed to move
        in the world frame

        Keyword arguments:
        vel -- Current velocity vector of the flying base (6 by 1, linear and angular stacked)
        """

        # Displacement along x and y
        c, s = np.cos(- vel[5, 0] * self.dt), np.sin(- vel[5, 0] * self.dt)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        # Update desired 3D position
        self.desired_pos = np.dot(R, self.desired_pos -
                                  self.dt * np.vstack((np.tile(vel[0:2, 0:1], (1, 4)), np.zeros((1, 4)))))

        # Update desired 2D location of footsteps
        self.footsteps_lock = np.dot(R[0:2, 0:2], self.footsteps_lock
                                     - self.dt * np.tile(vel[0:2, 0:1], (1, 4)))

        return 0

    def update_viewer(self, viewer, initialisation):
        """Update display for visualization purpose

        Keyword arguments:
        :param viewer: A gepetto viewer object
        :param initialisation: A bool, is it the first iteration of the main loop
        """

        # Display locked target footholds with red spheres (gepetto gui)
        rgbt = [1.0, 0.0, 0.0, 0.5]
        for i in range(4):
            if initialisation:
                viewer.gui.addSphere("world/sphere"+str(i)+"_lock", .025, rgbt)  # .1 is the radius
            viewer.gui.applyConfiguration("world/sphere"+str(i)+"_lock",
                                          (self.footsteps_lock_world[0, i], self.footsteps_lock_world[1, i],
                                           0.0, 1., 0., 0., 0.))

        # Display desired 3D position of feet with magenta spheres (gepetto gui)
        rgbt = [1.0, 0.0, 1.0, 0.5]
        for i in range(4):
            if initialisation:
                viewer.gui.addSphere("world/sphere"+str(i)+"_des", .03, rgbt)  # .1 is the radius
            viewer.gui.applyConfiguration("world/sphere"+str(i)+"_des",
                                          (self.desired_pos_world[0, i], self.desired_pos_world[1, i],
                                           self.desired_pos_world[2, i], 1., 0., 0., 0.))

        return 0


# @thomasfla's trajectory generator

class Foot_trajectory_generator(object):
    '''This class provide adaptative 3d trajectory for a foot from (x0,y0) to (x1,y1) using polynoms

    A foot trajectory generator that handles the generation of a 3D trajectory
    with a 5th order polynomial to lead each foot from its location at the start of
    its swing phase to its final location that has been decided by the FootstepPlanner

    Args:
        - h (float): the height at which feet should be raised at the apex of the wing phase
        - time_adaptative_disabled (float): how much time before touchdown is the desired position locked
    '''

    def __init__(self, h=0.03, time_adaptative_disabled=0.200, x_init=0.0, y_init=0.0):
        # maximum heigth for the z coordonate
        self.h = h

        # when there is less than this time for the trajectory to finish, disable adaptative (using last computed coefficients)
        # this parameter should always be a positive number less than the durration of a step
        self.time_adaptative_disabled = time_adaptative_disabled

        # memory of the last coeffs
        self.lastCoeffs_x = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.lastCoeffs_y = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # self.lastCoeffs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        #                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.x1 = x_init
        self.y1 = y_init

        # express acceleration as: ddx0 = (coeff_acc_x_lin_a) * x1 + coeff_acc_x_lin_b
        #                         ddy0 = (coeff_acc_y_lin_a) * x1 + coeff_acc_y_lin_b
        # Remark : When the trajectory becomes non-adaptative coeff_acc_x_lin_a is = 0.0 and coeff_acc_y_lin_b contains the full information of the acceleration!

        #self.coeff_acc_x_lin_a = 0.0
        #self.coeff_acc_x_lin_b = 0.0

        #self.coeff_acc_y_lin_a = 0.0
        #self.coeff_acc_y_lin_b = 0.0

    def get_next_foot(self, x0, dx0, ddx0, y0, dy0, ddy0, x1, y1, t0, t1,  dt):
        '''how to reach a foot position (here using polynomials profiles)'''

        epsilon = 0.00
        t2 = t1
        t3 = t0
        t1 -= 2*epsilon
        t0 -= epsilon

        h = self.h
        adaptative_mode = (t1 - t0) > self.time_adaptative_disabled
        if(adaptative_mode):
            # compute polynoms coefficients for x and y
            Ax5 = (ddx0*t0**2 - 2*ddx0*t0*t1 - 6*dx0*t0 + ddx0*t1**2 + 6*dx0*t1 + 12 *
                   x0 - 12*x1)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
            Ax4 = (30*t0*x1 - 30*t0*x0 - 30*t1*x0 + 30*t1*x1 - 2*t0**3*ddx0 - 3*t1**3*ddx0 + 14*t0**2*dx0 - 16*t1**2*dx0 +
                   2*t0*t1*dx0 + 4*t0*t1**2*ddx0 + t0**2*t1*ddx0)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
            Ax3 = (t0**4*ddx0 + 3*t1**4*ddx0 - 8*t0**3*dx0 + 12*t1**3*dx0 + 20*t0**2*x0 - 20*t0**2*x1 + 20*t1**2*x0 - 20*t1**2*x1 + 80*t0*t1*x0 - 80*t0 *
                   t1*x1 + 4*t0**3*t1*ddx0 + 28*t0*t1**2*dx0 - 32*t0**2*t1*dx0 - 8*t0**2*t1**2*ddx0)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
            Ax2 = -(t1**5*ddx0 + 4*t0*t1**4*ddx0 + 3*t0**4*t1*ddx0 + 36*t0*t1**3*dx0 - 24*t0**3*t1*dx0 + 60*t0*t1**2*x0 + 60*t0**2*t1*x0 - 60*t0*t1 **
                    2*x1 - 60*t0**2*t1*x1 - 8*t0**2*t1**3*ddx0 - 12*t0**2*t1**2*dx0)/(2*(t0**2 - 2*t0*t1 + t1**2)*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
            Ax1 = -(2*t1**5*dx0 - 2*t0*t1**5*ddx0 - 10*t0*t1**4*dx0 + t0**2*t1**4*ddx0 + 4*t0**3*t1**3*ddx0 - 3*t0**4*t1**2*ddx0 - 16*t0**2 *
                    t1**3*dx0 + 24*t0**3*t1**2*dx0 - 60*t0**2*t1**2*x0 + 60*t0**2*t1**2*x1)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
            Ax0 = (2*x1*t0**5 - ddx0*t0**4*t1**3 - 10*x1*t0**4*t1 + 2*ddx0*t0**3*t1**4 + 8*dx0*t0**3*t1**3 + 20*x1*t0**3*t1**2 - ddx0*t0**2*t1**5 - 10*dx0*t0 **
                   2*t1**4 - 20*x0*t0**2*t1**3 + 2*dx0*t0*t1**5 + 10*x0*t0*t1**4 - 2*x0*t1**5)/(2*(t0**2 - 2*t0*t1 + t1**2)*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))

            Ay5 = (ddy0*t0**2 - 2*ddy0*t0*t1 - 6*dy0*t0 + ddy0*t1**2 + 6*dy0*t1 + 12 *
                   y0 - 12*y1)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
            Ay4 = (30*t0*y1 - 30*t0*y0 - 30*t1*y0 + 30*t1*y1 - 2*t0**3*ddy0 - 3*t1**3*ddy0 + 14*t0**2*dy0 - 16*t1**2*dy0 +
                   2*t0*t1*dy0 + 4*t0*t1**2*ddy0 + t0**2*t1*ddy0)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
            Ay3 = (t0**4*ddy0 + 3*t1**4*ddy0 - 8*t0**3*dy0 + 12*t1**3*dy0 + 20*t0**2*y0 - 20*t0**2*y1 + 20*t1**2*y0 - 20*t1**2*y1 + 80*t0*t1*y0 - 80*t0 *
                   t1*y1 + 4*t0**3*t1*ddy0 + 28*t0*t1**2*dy0 - 32*t0**2*t1*dy0 - 8*t0**2*t1**2*ddy0)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
            Ay2 = -(t1**5*ddy0 + 4*t0*t1**4*ddy0 + 3*t0**4*t1*ddy0 + 36*t0*t1**3*dy0 - 24*t0**3*t1*dy0 + 60*t0*t1**2*y0 + 60*t0**2*t1*y0 - 60*t0*t1 **
                    2*y1 - 60*t0**2*t1*y1 - 8*t0**2*t1**3*ddy0 - 12*t0**2*t1**2*dy0)/(2*(t0**2 - 2*t0*t1 + t1**2)*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
            Ay1 = -(2*t1**5*dy0 - 2*t0*t1**5*ddy0 - 10*t0*t1**4*dy0 + t0**2*t1**4*ddy0 + 4*t0**3*t1**3*ddy0 - 3*t0**4*t1**2*ddy0 - 16*t0**2 *
                    t1**3*dy0 + 24*t0**3*t1**2*dy0 - 60*t0**2*t1**2*y0 + 60*t0**2*t1**2*y1)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
            Ay0 = (2*y1*t0**5 - ddy0*t0**4*t1**3 - 10*y1*t0**4*t1 + 2*ddy0*t0**3*t1**4 + 8*dy0*t0**3*t1**3 + 20*y1*t0**3*t1**2 - ddy0*t0**2*t1**5 - 10*dy0*t0 **
                   2*t1**4 - 20*y0*t0**2*t1**3 + 2*dy0*t0*t1**5 + 10*y0*t0*t1**4 - 2*y0*t1**5)/(2*(t0**2 - 2*t0*t1 + t1**2)*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))

            # den = (2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
            # We are more interested in the expression of coefficients as linear fonction of the final position (x1,y1)
            # in fact: Ax5 = cx5*x1 + dx5
            #         Ax4 = cx4*x1 + dx4
            #         Ax3 = cx3*x1 + dx3
            #         Ax2 = cx2*x1 + dx2
            #         Ax1 = cx1*x1 + dx1
            #         Ax0 = cx0*x1 + dx0
            #       Same for Ay5..Ay0
            """cx5 = (-12)/den
            dx5 = (ddx0*t0**2 - 2*ddx0*t0*t1 - 6*dx0*t0 + ddx0*t1**2 + 6*dx0*t1 + 12*x0)/den

            cx4 = (30*t0 + 30*t1)/den
            dx4 = (- 30*t0*x0 - 30*t1*x0 - 2*t0**3*ddx0 - 3*t1**3*ddx0 + 14*t0**2*dx0 -
                   16*t1**2*dx0 + 2*t0*t1*dx0 + 4*t0*t1**2*ddx0 + t0**2*t1*ddx0)/den

            cx3 = (-20*t0**2 - 20*t1**2 - 80*t0*t1)/den
            dx3 = (t0**4*ddx0 + 3*t1**4*ddx0 - 8*t0**3*dx0 + 12*t1**3*dx0 + 20*t0**2*x0 + 20*t1**2*x0 + 80 *
                   t0*t1*x0 + 4*t0**3*t1*ddx0 + 28*t0*t1**2*dx0 - 32*t0**2*t1*dx0 - 8*t0**2*t1**2*ddx0)/den

            cx2 = -(- 60*t0*t1**2 - 60*t0**2*t1)/den
            dx2 = -(t1**5*ddx0 + 4*t0*t1**4*ddx0 + 3*t0**4*t1*ddx0 + 36*t0*t1**3*dx0 - 24*t0**3*t1 *
                    dx0 + 60*t0*t1**2*x0 + 60*t0**2*t1*x0 - 8*t0**2*t1**3*ddx0 - 12*t0**2*t1**2*dx0)/den

            cx1 = -(60*t0**2*t1**2)/den
            dx1 = -(2*t1**5*dx0 - 2*t0*t1**5*ddx0 - 10*t0*t1**4*dx0 + t0**2*t1**4*ddx0 + 4*t0**3*t1**3*ddx0 -
                    3*t0**4*t1**2*ddx0 - 16*t0**2*t1**3*dx0 + 24*t0**3*t1**2*dx0 - 60*t0**2*t1**2*x0)/den

            cx0 = (20*t0**3*t1**2 + 2*t0**5 - 10*t0**4*t1) / den
            dx0 = (- ddx0*t0**4*t1**3 + 2*ddx0*t0**3*t1**4 + 8*dx0*t0**3*t1**3 - ddx0*t0**2*t1**5 - 10 *
                   dx0*t0**2*t1**4 - 20*x0*t0**2*t1**3 + 2*dx0*t0*t1**5 + 10*x0*t0*t1**4 - 2*x0*t1**5)/den

            cy5 = (-12)/den
            dy5 = (ddy0*t0**2 - 2*ddy0*t0*t1 - 6*dy0*t0 + ddy0*t1**2 + 6*dy0*t1 + 12*y0)/den

            cy4 = (30*t0 + 30*t1)/den
            dy4 = (- 30*t0*y0 - 30*t1*y0 - 2*t0**3*ddy0 - 3*t1**3*ddy0 + 14*t0**2*dy0 -
                   16*t1**2*dy0 + 2*t0*t1*dy0 + 4*t0*t1**2*ddy0 + t0**2*t1*ddy0)/den

            cy3 = (-20*t0**2 - 20*t1**2 - 80*t0*t1)/den
            dy3 = (t0**4*ddy0 + 3*t1**4*ddy0 - 8*t0**3*dy0 + 12*t1**3*dy0 + 20*t0**2*y0 + 20*t1**2*y0 + 80 *
                   t0*t1*y0 + 4*t0**3*t1*ddy0 + 28*t0*t1**2*dy0 - 32*t0**2*t1*dy0 - 8*t0**2*t1**2*ddy0)/den

            cy2 = -(- 60*t0*t1**2 - 60*t0**2*t1)/den
            dy2 = -(t1**5*ddy0 + 4*t0*t1**4*ddy0 + 3*t0**4*t1*ddy0 + 36*t0*t1**3*dy0 - 24*t0**3*t1 *
                    dy0 + 60*t0*t1**2*y0 + 60*t0**2*t1*y0 - 8*t0**2*t1**3*ddy0 - 12*t0**2*t1**2*dy0)/den

            cy1 = -(60*t0**2*t1**2)/den
            dy1 = -(2*t1**5*dy0 - 2*t0*t1**5*ddy0 - 10*t0*t1**4*dy0 + t0**2*t1**4*ddy0 + 4*t0**3*t1**3*ddy0 -
                    3*t0**4*t1**2*ddy0 - 16*t0**2*t1**3*dy0 + 24*t0**3*t1**2*dy0 - 60*t0**2*t1**2*y0)/den

            cy0 = (20*t0**3*t1**2 + 2*t0**5 - 10*t0**4*t1) / den
            dy0 = (- ddy0*t0**4*t1**3 + 2*ddy0*t0**3*t1**4 + 8*dy0*t0**3*t1**3 - ddy0*t0**2*t1**5 - 10 *
                   dy0*t0**2*t1**4 - 20*y0*t0**2*t1**3 + 2*dy0*t0*t1**5 + 10*y0*t0*t1**4 - 2*y0*t1**5)/den"""

            # test should be zero : ok
            # ~ print Ax0 - (cx0*x1 + dx0)
            # ~ print Ax1 - (cx1*x1 + dx1)
            # ~ print Ax2 - (cx2*x1 + dx2)
            # ~ print Ax3 - (cx3*x1 + dx3)
            # ~ print Ax4 - (cx4*x1 + dx4)
            # ~ print Ax5 - (cx5*x1 + dx5)

            self.lastCoeffs_x = [Ax5, Ax4, Ax3, Ax2, Ax1, Ax0]  # save coeffs
            self.lastCoeffs_y = [Ay5, Ay4, Ay3, Ay2, Ay1, Ay0]

            # self.lastCoeffs = [cx5, cx4, cx3, cx2, cx1, cx0, dx5, dx4, dx3, dx2,
            #                   dx1, dx0, cy5, cy4, cy3, cy2, cy1, cy0, dy5, dy4, dy3, dy2, dy1, dy0]

            self.x1 = x1  # save last x1 value
            self.y1 = y1  # save last y1 value

        else:
            [Ax5, Ax4, Ax3, Ax2, Ax1, Ax0] = self.lastCoeffs_x  # use last coeffs
            [Ay5, Ay4, Ay3, Ay2, Ay1, Ay0] = self.lastCoeffs_y
            # [cx5, cx4, cx3, cx2, cx1, cx0, dx5, dx4, dx3, dx2, dx1, dx0, cy5, cy4,
            #    cy3, cy2, cy1, cy0, dy5, dy4, dy3, dy2, dy1, dy0] = self.lastCoeffs

        # coefficients for z (deterministic)
        Az6 = -h/((t2/2)**3*(t2 - t2/2)**3)
        Az5 = (3*t2*h)/((t2/2)**3*(t2 - t2/2)**3)
        Az4 = -(3*t2**2*h)/((t2/2)**3*(t2 - t2/2)**3)
        Az3 = (t2**3*h)/((t2/2)**3*(t2 - t2/2)**3)

        # get the next point
        ev = t0+dt
        evz = t3+dt
        x1 = self.x1
        y1 = self.y1
        """x0 = x1 * (cx0 + cx1*ev + cx2*ev**2 + cx3*ev**3 + cx4*ev**4 + cx5*ev**5) + \
            dx0 + dx1*ev + dx2*ev**2 + dx3*ev**3 + dx4*ev**4 + dx5*ev**5
        dx0 = x1 * (cx1 + 2*cx2*ev + 3*cx3*ev**2 + 4*cx4*ev**3 + 5*cx5*ev**4) + \
            dx1 + 2*dx2*ev + 3*dx3*ev**2 + 4*dx4*ev**3 + 5*dx5*ev**4
        ddx0 = x1 * (2*cx2 + 3*2*cx3*ev + 4*3*cx4*ev**2 + 5*4*cx5*ev**3) + \
            2*dx2 + 3*2*dx3*ev + 4*3*dx4*ev**2 + 5*4*dx5*ev**3

        y0 = y1 * (cy0 + cy1*ev + cy2*ev**2 + cy3*ev**3 + cy4*ev**4 + cy5*ev**5) + \
            dy0 + dy1*ev + dy2*ev**2 + dy3*ev**3 + dy4*ev**4 + dy5*ev**5
        dy0 = y1 * (cy1 + 2*cy2*ev + 3*cy3*ev**2 + 4*cy4*ev**3 + 5*cy5*ev**4) + \
            dy1 + 2*dy2*ev + 3*dy3*ev**2 + 4*dy4*ev**3 + 5*dy5*ev**4
        ddy0 = y1 * (2*cy2 + 3*2*cy3*ev + 4*3*cy4*ev**2 + 5*4*cy5*ev**3) + \
            2*dy2 + 3*2*dy3*ev + 4*3*dy4*ev**2 + 5*4*dy5*ev**3"""

        z0 = Az3*evz**3 + Az4*evz**4 + Az5*evz**5 + Az6*evz**6
        dz0 = 3*Az3*evz**2 + 4*Az4*evz**3 + 5*Az5*evz**4 + 6*Az6*evz**5
        ddz0 = 2*3*Az3*evz + 3*4*Az4*evz**2 + 4*5*Az5*evz**3 + 5*6*Az6*evz**4

        if (t3 < epsilon) or (t3 > (t2-epsilon)):
            return [x0, 0.0, 0.0,  y0, 0.0, 0.0,  z0, dz0, ddz0, self.x1, self.y1]
        else:
            x0 = Ax0 + Ax1*ev + Ax2*ev**2 + Ax3*ev**3 + Ax4*ev**4 + Ax5*ev**5
            dx0 = Ax1 + 2*Ax2*ev + 3*Ax3*ev**2 + 4*Ax4*ev**3 + 5*Ax5*ev**4
            ddx0 = 2*Ax2 + 3*2*Ax3*ev + 4*3*Ax4*ev**2 + 5*4*Ax5*ev**3

            y0 = Ay0 + Ay1*ev + Ay2*ev**2 + Ay3*ev**3 + Ay4*ev**4 + Ay5*ev**5
            dy0 = Ay1 + 2*Ay2*ev + 3*Ay3*ev**2 + 4*Ay4*ev**3 + 5*Ay5*ev**4
            ddy0 = 2*Ay2 + 3*2*Ay3*ev + 4*3*Ay4*ev**2 + 5*4*Ay5*ev**3

            return [x0, dx0, ddx0,  y0, dy0, ddy0,  z0, dz0, ddz0, self.x1, self.y1]

        # expression de ddx0 comme une fonction lineaire de x1:
        """if(adaptative_mode):
            self.coeff_acc_x_lin_a = 2*cx2 + 3*2*cx3*ev + 4*3*cx4*ev**2 + 5*4*cx5*ev**3
            self.coeff_acc_x_lin_b = 2*dx2 + 3*2*dx3*ev + 4*3*dx4*ev**2 + 5*4*dx5*ev**3
            self.coeff_acc_y_lin_a = 2*cy2 + 3*2*cy3*ev + 4*3*cy4*ev**2 + 5*4*cy5*ev**3
            self.coeff_acc_y_lin_b = 2*dy2 + 3*2*dy3*ev + 4*3*dy4*ev**2 + 5*4*dy5*ev**3
        else:
            self.coeff_acc_x_lin_a = 0.0
            self.coeff_acc_x_lin_b = x1 * (2*cx2 + 3*2*cx3*ev + 4*3*cx4*ev**2 + 5*4 *
                                           cx5*ev**3) + 2*dx2 + 3*2*dx3*ev + 4*3*dx4*ev**2 + 5*4*dx5*ev**3
            self.coeff_acc_y_lin_a = 0.0
            self.coeff_acc_y_lin_b = y1 * (2*cy2 + 3*2*cy3*ev + 4*3*cy4*ev**2 + 5*4 *
                                           cy5*ev**3) + 2*dy2 + 3*2*dy3*ev + 4*3*dy4*ev**2 + 5*4*dy5*ev**3"""

        # get the target point (usefull for inform the MPC when we are not adaptative anymore.
        # ev = t1
        # ~ x1  =Ax0 + Ax1*ev + Ax2*ev**2 + Ax3*ev**3 + Ax4*ev**4 + Ax5*ev**5
        # ~ dx1 =Ax1 + 2*Ax2*ev + 3*Ax3*ev**2 + 4*Ax4*ev**3 + 5*Ax5*ev**4
        # ~ ddx1=2*Ax2 + 3*2*Ax3*ev + 4*3*Ax4*ev**2 + 5*4*Ax5*ev**3

        # ~ y1  =Ay0 + Ay1*ev + Ay2*ev**2 + Ay3*ev**3 + Ay4*ev**4 + Ay5*ev**5
        # ~ dy1 =Ay1 + 2*Ay2*ev + 3*Ay3*ev**2 + 4*Ay4*ev**3 + 5*Ay5*ev**4
        # ~ ddy1=2*Ay2 + 3*2*Ay3*ev + 4*3*Ay4*ev**2 + 5*4*Ay5*ev**3

        # return [x0, dx0, ddx0,  y0, dy0, ddy0,  z0, dz0, ddz0, self.x1, self.y1]
