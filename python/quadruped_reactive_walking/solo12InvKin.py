
from example_robot_data import load
import time
import numpy as np
import pinocchio as pin
import libquadruped_reactive_walking as lrw

class Solo12InvKin:
    def __init__(self, dt):
        self.robot = load('solo12')
        self.dt = dt

        self.InvKinCpp = lrw.InvKin(dt)

        # Inputs to be modified bu the user before calling .compute
        self.feet_position_ref = [np.array([0.1946,   0.14695, 0.0191028]), np.array(
            [0.1946,  -0.14695, 0.0191028]), np.array([-0.1946,   0.14695, 0.0191028]),
            np.array([-0.1946,  -0.14695, 0.0191028])]
        self.feet_velocity_ref = [np.array([0., 0., 0.]), np.array(
            [0., 0., 0.]), np.array([0., 0., 0.]), np.array([0., 0., 0.])]
        self.feet_acceleration_ref = [np.array([0., 0., 0.]), np.array(
            [0., 0., 0.]), np.array([0., 0., 0.]), np.array([0., 0., 0.])]
        self.flag_in_contact = np.array([0, 1, 0, 1])
        self.base_orientation_ref = pin.utils.rpyToMatrix(0., 0., np.pi/6)
        self.base_angularvelocity_ref = np.array([0., 0., 0.])
        self.base_angularacceleration_ref = np.array([0., 0., 0.])
        self.base_position_ref = np.array([0., 0., 0.235])
        self.base_linearvelocity_ref = np.array([0., 0., 0.])
        self.base_linearacceleration_ref = np.array([0., 0., 0.])

        self.Kp_base_orientation = 100.0
        self.Kd_base_orientation = 2.0*np.sqrt(self.Kp_base_orientation)

        self.Kp_base_position = 100.0
        self.Kd_base_position = 2.0*np.sqrt(self.Kp_base_position)

        self.Kp_flyingfeet = 100.0
        self.Kd_flyingfeet = 2.0*np.sqrt(self.Kp_flyingfeet)

        self.x_ref = np.zeros((6, 1))
        self.x = np.zeros((6, 1))
        self.dx_ref = np.zeros((6, 1))
        self.dx = np.zeros((6, 1))

        # Matrices initialisation
        self.invJ = np.zeros((18, 18))

        self.cpp_posf = np.zeros((4, 3))
        self.cpp_vf = np.zeros((4, 3))
        self.cpp_wf = np.zeros((4, 3))
        self.cpp_af = np.zeros((4, 3))
        self.cpp_Jf = np.zeros((12, 18))

        self.cpp_posb = np.zeros((1, 3))
        self.cpp_rotb = np.zeros((3, 3))
        self.cpp_vb = np.zeros((1, 6))
        self.cpp_ab = np.zeros((1, 6))
        self.cpp_Jb = np.zeros((6, 18))

        self.cpp_ddq = np.zeros((18,))
        self.cpp_q_cmd = np.zeros((19,))
        self.cpp_dq_cmd = np.zeros((18,))

        # Get frame IDs
        FL_FOOT_ID = self.robot.model.getFrameId('FL_FOOT')
        FR_FOOT_ID = self.robot.model.getFrameId('FR_FOOT')
        HL_FOOT_ID = self.robot.model.getFrameId('HL_FOOT')
        HR_FOOT_ID = self.robot.model.getFrameId('HR_FOOT')
        self.BASE_ID = self.robot.model.getFrameId('base_link')
        self.foot_ids = np.array([FL_FOOT_ID, FR_FOOT_ID, HL_FOOT_ID, HR_FOOT_ID])

        def dinv(J, damping=1e-2):
            ''' Damped inverse '''
            U, S, V = np.linalg.svd(J)
            if damping == 0:
                Sinv = 1/S
            else:
                Sinv = S/(S**2+damping**2)
            return (V.T*Sinv)@U.T

        self.rmodel = self.robot.model
        self.rdata = self.robot.data
        self.i = 0

    def dinv(self, J, damping=1e-2):
        ''' Damped inverse '''
        U, S, V = np.linalg.svd(J)
        if damping == 0:
            Sinv = 1/S
        else:
            Sinv = S/(S**2+damping**2)
        return (V.T*Sinv)@U.T

    def cross3(self, left, right):
        """Numpy is inefficient for this"""
        return np.array([left[1] * right[2] - left[2] * right[1],
                         left[2] * right[0] - left[0] * right[2],
                         left[0] * right[1] - left[1] * right[0]])

    def refreshAndCompute(self, q, dq, x_cmd, contacts, planner):

        # Update contact status of the feet
        self.flag_in_contact[:] = contacts

        # Update position, velocity and acceleration references for the feet
        for i in range(4):
            self.feet_position_ref[i] = planner.goals[0:3, i]  # + np.array([0.0, 0.0, q[2, 0] - planner.h_ref])
            self.feet_velocity_ref[i] = planner.vgoals[0:3, i]
            self.feet_acceleration_ref[i] = planner.agoals[0:3, i]

        # Update model and data
        pin.computeJointJacobians(self.rmodel, self.rdata, q)
        pin.forwardKinematics(self.rmodel, self.rdata, q, dq, np.zeros(self.rmodel.nv))
        pin.updateFramePlacements(self.rmodel, self.rdata)

        # Get data required by IK with Pinocchio
        for i_ee in range(4):
            idx = int(self.foot_ids[i_ee])
            self.cpp_posf[i_ee, :] = self.rdata.oMf[idx].translation
            nu = pin.getFrameVelocity(self.rmodel, self.rdata, idx, pin.LOCAL_WORLD_ALIGNED)
            self.cpp_vf[i_ee, :] = nu.linear
            self.cpp_wf[i_ee, :] = nu.angular
            self.cpp_af[i_ee, :] = pin.getFrameAcceleration(self.rmodel, self.rdata, idx, pin.LOCAL_WORLD_ALIGNED).linear
            self.cpp_Jf[(3*i_ee):(3*(i_ee+1)), :] = pin.getFrameJacobian(self.robot.model, self.robot.data, idx, pin.LOCAL_WORLD_ALIGNED)[:3]

        self.cpp_posb[:] = self.rdata.oMf[self.BASE_ID].translation
        self.cpp_rotb[:, :] = self.rdata.oMf[self.BASE_ID].rotation
        nu = pin.getFrameVelocity(self.rmodel, self.rdata, self.BASE_ID, pin.LOCAL_WORLD_ALIGNED)
        self.cpp_vb[0, 0:3] = nu.linear
        self.cpp_vb[0, 3:6] = nu.angular
        acc = pin.getFrameAcceleration(self.rmodel, self.rdata, self.BASE_ID, pin.LOCAL_WORLD_ALIGNED)
        self.cpp_ab[0, 0:3] = acc.linear
        self.cpp_ab[0, 3:6] = acc.angular
        self.cpp_Jb[:, :] = pin.getFrameJacobian(self.robot.model, self.robot.data, self.BASE_ID, pin.LOCAL_WORLD_ALIGNED)

        self.cpp_ddq[:] = self.InvKinCpp.refreshAndCompute(np.array([x_cmd]), np.array([contacts]), planner.goals, planner.vgoals, planner.agoals,
                                                           self.cpp_posf, self.cpp_vf, self.cpp_wf, self.cpp_af, self.cpp_Jf,
                                                           self.cpp_posb, self.cpp_rotb, self.cpp_vb, self.cpp_ab, self.cpp_Jb)

        self.cpp_q_cmd[:] = pin.integrate(self.robot.model, q, self.InvKinCpp.get_q_step())
        self.cpp_dq_cmd[:] = self.InvKinCpp.get_dq_cmd()

        self.q_cmd = self.cpp_q_cmd
        self.dq_cmd = self.cpp_dq_cmd

        return self.cpp_ddq

    def compute(self, q, dq):
        # FEET
        Jfeet = []
        afeet = []
        self.pfeet_err = []
        vfeet_ref = []
        pin.computeJointJacobians(self.rmodel, self.rdata, q)
        pin.forwardKinematics(self.rmodel, self.rdata, q, dq, np.zeros(self.rmodel.nv))
        pin.updateFramePlacements(self.rmodel, self.rdata)

        for i_ee in range(4):
            idx = int(self.foot_ids[i_ee])
            pos = self.rdata.oMf[idx].translation
            nu = pin.getFrameVelocity(self.rmodel, self.rdata, idx, pin.LOCAL_WORLD_ALIGNED)
            ref = self.feet_position_ref[i_ee]
            vref = self.feet_velocity_ref[i_ee]
            aref = self.feet_acceleration_ref[i_ee]

            J1 = pin.getFrameJacobian(self.robot.model, self.robot.data, idx, pin.LOCAL_WORLD_ALIGNED)[:3]
            # J1 = pin.computeFrameJacobian(self.robot.model, self.robot.data, q, idx, pin.LOCAL_WORLD_ALIGNED)[:3]
            # print(np.array_equal(J1, J1b))
            e1 = ref-pos
            acc1 = -self.Kp_flyingfeet*(pos-ref) - self.Kd_flyingfeet*(nu.linear-vref) + aref
            if self.flag_in_contact[i_ee]:
                acc1 *= 1  # In contact = no feedback
            drift1 = np.zeros(3)
            drift1 += pin.getFrameAcceleration(self.rmodel, self.rdata, idx, pin.LOCAL_WORLD_ALIGNED).linear
            drift1 += self.cross3(nu.angular, nu.linear)
            acc1 -= drift1

            Jfeet.append(J1)
            afeet.append(acc1)

            self.pfeet_err.append(e1)
            vfeet_ref.append(vref)


        # BASE POSITION
        idx = self.BASE_ID
        pos = self.rdata.oMf[idx].translation
        nu = pin.getFrameVelocity(self.rmodel, self.rdata, idx, pin.LOCAL_WORLD_ALIGNED)
        ref = self.base_position_ref
        Jbasis = pin.getFrameJacobian(self.robot.model, self.robot.data, idx, pin.LOCAL_WORLD_ALIGNED)[:3]
        e_basispos = ref - pos
        accbasis = -self.Kp_base_position*(pos-ref) - self.Kd_base_position*(nu.linear-self.base_linearvelocity_ref)
        drift = np.zeros(3)
        drift += pin.getFrameAcceleration(self.rmodel, self.rdata, idx, pin.LOCAL_WORLD_ALIGNED).linear
        drift += self.cross3(nu.angular, nu.linear)
        accbasis -= drift

        self.x_ref[0:3, 0] = ref
        self.x[0:3, 0] = pos

        self.dx_ref[0:3, 0] = self.base_linearvelocity_ref
        self.dx[0:3, 0] = nu.linear

        # BASE ROTATION
        idx = self.BASE_ID

        rot = self.rdata.oMf[idx].rotation
        nu = pin.getFrameVelocity(self.rmodel, self.rdata, idx, pin.LOCAL_WORLD_ALIGNED)
        rotref = self.base_orientation_ref
        Jwbasis = pin.getFrameJacobian(self.robot.model, self.robot.data, idx, pin.LOCAL_WORLD_ALIGNED)[3:]
        e_basisrot = -rotref @ pin.log3(rotref.T@rot)
        accwbasis = -self.Kp_base_orientation * \
            rotref @ pin.log3(rotref.T@rot) - self.Kd_base_orientation*(nu.angular - self.base_angularvelocity_ref)
        drift = np.zeros(3)
        drift += pin.getFrameAcceleration(self.rmodel, self.rdata, idx, pin.LOCAL_WORLD_ALIGNED).angular
        accwbasis -= drift

        self.x_ref[3:6, 0] = np.zeros(3)
        self.x[3:6, 0] = np.zeros(3)

        self.dx_ref[3:6, 0] = self.base_angularvelocity_ref
        self.dx[3:6, 0] = nu.angular

        J = np.vstack([Jbasis, Jwbasis]+Jfeet)
        acc = np.concatenate([accbasis, accwbasis]+afeet)

        x_err = np.concatenate([e_basispos, e_basisrot]+self.pfeet_err)
        dx_ref = np.concatenate([self.base_linearvelocity_ref, self.base_angularvelocity_ref]+vfeet_ref)

        """import time
        tic = time.time()
        invR = J[0:3, 0:3].transpose()
        self.invJ[0:3, 0:3] = invR
        self.invJ[3:6, 3:6] = invR

        for i in range(4):
            inv = np.linalg.pinv(J[(6+3*i):(9+3*i), (6+3*i):(9+3*i)])
            self.invJ[(6+3*i):(9+3*i), 0:3] = - inv
            self.invJ[(6+3*i):(9+3*i), 3:6] = (- inv @ J[(6+3*i):(9+3*i), 3:6]) @ invR
            self.invJ[(6+3*i):(9+3*i), (6+3*i):(9+3*i)] = inv
        tac = time.time()"""

        print("J:")
        print(J)
        invJ = np.linalg.pinv(J)  # self.dinv(J)  # or np.linalg.inv(J) since full rank

        """toc = time.time()
        print("Old:", toc - tac)
        print("New:", tac - tic)"""

        print("invJ:")
        print(invJ)
        print("acc:")
        print(acc)
        
        ddq = invJ @ acc
        self.q_cmd = pin.integrate(self.robot.model, q, invJ @ x_err)
        self.dq_cmd = invJ @ dx_ref

        print("q_step")
        print(invJ @ x_err)
        print("dq_cmd:")
        print(self.dq_cmd)
        """from IPython import embed
        embed()"""

        return ddq


if __name__ == "__main__":
    USE_VIEWER = True
    print("test")
    dt = 0.002
    invKin = Solo12InvKin(dt)
    q = invKin.robot.q0.copy()
    q = np.array([[-3.87696007e-01],
                  [-4.62877770e+00],
                  [1.87606547e-01],
                  [1.32558492e-02],
                  [8.87905574e-03],
                  [-8.86025995e-01],
                  [4.63360961e-01],
                  [9.62126158e-03],
                  [6.06172292e-01],
                  [-1.48984107e+00],
                  [4.44117781e-03],
                  [1.08394553e+00],
                  [-1.40899150e+00],
                  [-5.22347798e-02],
                  [-4.24868613e-01],
                  [1.44182047e+00],
                  [4.41620770e-02],
                  [-9.76513563e-01],
                  [1.41483950e+00]])

    dq = np.array([[0.92799144],
                   [0.02038822],
                   [-0.10578672],
                   [1.29588322],
                   [-0.23417772],
                   [0.32688336],
                   [-1.60580342],
                   [4.67635444],
                   [-1.54127171],
                   [-1.63819893],
                   [7.81376752],
                   [-4.61388499],
                   [0.30138108],
                   [4.57546437],
                   [4.92438176],
                   [3.18059759],
                   [2.83654818],
                   [-5.17240673]])

    pin.forwardKinematics(invKin.rmodel, invKin.rdata, q, dq, np.zeros(invKin.rmodel.nv))
    pin.updateFramePlacements(invKin.rmodel, invKin.rdata)

    print("Initial positions")
    print("Base pos:", q[0:3].ravel())
    print("Base vel:", dq[0:3].ravel())
    for i_ee in range(4):
        idx = int(invKin.foot_ids[i_ee])
        pos = invKin.rdata.oMf[idx].translation
        print(i_ee, ": ", pos.ravel())
        invKin.feet_position_ref[i_ee] = np.array([pos[0], pos[1], 0.0])

    if USE_VIEWER:
        invKin.robot.initViewer(loadModel=True)
        invKin.robot.viewer.gui.setRefreshIsSynchronous(False)

        # invKin.robot.display(q)

    for i in range(1000):
        t = i*dt
        # set the references
        """invKin.feet_position_ref = [
            np.array([0.1946,   0.14695, 0.0]),
            np.array([0.1946,  -0.14695, 0.0]),
            np.array([-0.1946,   0.14695, 0.0]),
            np.array([-0.1946,  -0.14695, 0.0])]"""

        invKin.feet_velocity_ref = [
            np.array([0, 0, 0.]),
            np.array([0, 0, 0.]),
            np.array([0, 0, 0.]),
            np.array([0, 0, 0.])]

        invKin.feet_acceleration_ref = [
            np.array([0, 0, 0.]),
            np.array([0, 0, 0.]),
            np.array([0, 0, 0.]),
            np.array([0, 0, 0.])]

        invKin.base_position_ref[:] = np.array([0.0, 0.0, 0.223])
        invKin.base_orientation_ref[:] = np.array([0.0, 0.0, 0.0])
        invKin.base_linearvelocity_ref[:] = np.array([0.0, 0.0, 0.0])
        invKin.base_angularvelocity_ref[:] = np.array([0.0, 0.0, 0.0])

        ddq = invKin.compute(q, dq)
        dq = dq + dt*np.array([ddq]).T
        q = pin.integrate(invKin.robot.model, q, dq*dt)

        pin.forwardKinematics(invKin.rmodel, invKin.rdata, q, dq, np.zeros(invKin.rmodel.nv))
        pin.updateFramePlacements(invKin.rmodel, invKin.rdata)

        print("###")
        print("Base pos:", q[0:3].ravel())
        print("Base vel:", dq[0:3].ravel())
        for i_ee in range(4):
            idx = int(invKin.foot_ids[i_ee])
            pos = invKin.rdata.oMf[idx].translation
            print(i_ee, ": ", pos.ravel())

        if USE_VIEWER:
            invKin.robot.display(q)
