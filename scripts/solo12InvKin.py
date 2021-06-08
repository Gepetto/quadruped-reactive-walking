
# from example_robot_data import load
import numpy as np
import pinocchio as pin
import libquadruped_reactive_walking as lrw
from example_robot_data.robots_loader import Solo12Loader

class Solo12InvKin:
    def __init__(self, dt):

        # Robot model
        Solo12Loader.free_flyer = False
        self.robot = Solo12Loader().robot
        self.dt = dt

        # Inverse Kinematics solver in C++
        self.InvKinCpp = lrw.InvKin(dt)

        # Memory assignation for variables
        self.cpp_posf = np.zeros((4, 3))
        self.cpp_vf = np.zeros((4, 3))
        self.cpp_wf = np.zeros((4, 3))
        self.cpp_af = np.zeros((4, 3))
        self.cpp_Jf = np.zeros((12, 12))

        self.ddq_cmd = np.zeros((18,))
        self.dq_cmd = np.zeros((18,))
        self.q_cmd = np.zeros((19,))

        # Get frame IDs
        FL_FOOT_ID = self.robot.model.getFrameId('FL_FOOT')
        FR_FOOT_ID = self.robot.model.getFrameId('FR_FOOT')
        HL_FOOT_ID = self.robot.model.getFrameId('HL_FOOT')
        HR_FOOT_ID = self.robot.model.getFrameId('HR_FOOT')
        self.BASE_ID = FL_FOOT_ID # self.robot.model.getFrameId('base_link') TODO REMOVE
        self.foot_ids = np.array([FL_FOOT_ID, FR_FOOT_ID, HL_FOOT_ID, HR_FOOT_ID])

    def cross3(self, left, right):
        """Numpy is inefficient for this"""
        return np.array([left[1] * right[2] - left[2] * right[1],
                         left[2] * right[0] - left[0] * right[2],
                         left[0] * right[1] - left[1] * right[0]])

    def refreshAndCompute(self, q, dq, contacts, pgoals, vgoals, agoals):

        # Update model and data of the robot
        pin.computeJointJacobians(self.robot.model, self.robot.data, q)
        pin.forwardKinematics(self.robot.model, self.robot.data, q, dq, np.zeros(self.robot.model.nv))
        pin.updateFramePlacements(self.robot.model, self.robot.data)

        # Get data required by IK with Pinocchio
        for i_ee in range(4):
            idx = int(self.foot_ids[i_ee])
            self.cpp_posf[i_ee, :] = self.robot.data.oMf[idx].translation
            nu = pin.getFrameVelocity(self.robot.model, self.robot.data, idx, pin.LOCAL_WORLD_ALIGNED)
            self.cpp_vf[i_ee, :] = nu.linear
            self.cpp_wf[i_ee, :] = nu.angular
            self.cpp_af[i_ee, :] = pin.getFrameAcceleration(self.robot.model, self.robot.data, idx, pin.LOCAL_WORLD_ALIGNED).linear
            self.cpp_Jf[(3*i_ee):(3*(i_ee+1)), :] = pin.getFrameJacobian(self.robot.model, self.robot.data, idx, pin.LOCAL_WORLD_ALIGNED)[:3]

        # IK output for accelerations of actuators
        self.ddq_cmd[6:] = self.InvKinCpp.refreshAndCompute(np.array([contacts]), pgoals, vgoals, agoals,
                                                            self.cpp_posf, self.cpp_vf, self.cpp_wf,
                                                            self.cpp_af, self.cpp_Jf)

        self.dq_cmd[6:] = self.InvKinCpp.get_dq_cmd()  # IK output for velocities of actuators
        self.q_cmd[7:] = q[:, 0] + self.InvKinCpp.get_q_step()  # IK output for positions of actuators

        return self.ddq_cmd


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
