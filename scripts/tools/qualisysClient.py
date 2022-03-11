""" 
This client connects to a Qualisys (motion capture) server with an asyncronous subprocess and expose 6d position and velocity of a given body
Thomas FLAYOLS - LAAS CNRS
"""

import asyncio
from multiprocessing import Process, Lock
from multiprocessing.sharedctypes import Value, Array
from ctypes import c_double
import qtm
import numpy as np

import pinocchio
from pinocchio.utils import se3ToXYZQUAT
from pinocchio.explog import log


class QualisysClient():
    def __init__(self, ip="127.0.0.1", body_id=0):
        # shared c_double array
        self.shared_bodyPosition = Array(c_double, 3, lock=False)
        self.shared_bodyVelocity = Array(c_double, 3, lock=False)
        self.shared_bodyOrientationQuat = Array(c_double, 4, lock=False)
        self.shared_bodyOrientationMat9 = Array(c_double, 9, lock=False)
        self.shared_bodyAngularVelocity = Array(c_double, 3, lock=False)
        self.shared_timestamp = Value(c_double, lock=False)
        #self.shared_timestamp = -1
        args = (ip, body_id, self.shared_bodyPosition, self.shared_bodyVelocity,
                self.shared_bodyOrientationQuat, self.shared_bodyOrientationMat9,
                self.shared_bodyAngularVelocity, self.shared_timestamp)
        self.p = Process(target=self.qualisys_process, args=args)
        self.p.start()

    def stop(self):
        self.p.terminate()
        self.p.join()

    def getPosition(self):
        return np.array([self.shared_bodyPosition[0],
                         self.shared_bodyPosition[1],
                         self.shared_bodyPosition[2]])

    def getVelocity(self):
        return np.array([self.shared_bodyVelocity[0],
                         self.shared_bodyVelocity[1],
                         self.shared_bodyVelocity[2]])

    def getAngularVelocity(self):
        return np.array([self.shared_bodyAngularVelocity[0],
                         self.shared_bodyAngularVelocity[1],
                         self.shared_bodyAngularVelocity[2]])

    def getOrientationMat9(self):
        return np.array([[self.shared_bodyOrientationMat9[0], self.shared_bodyOrientationMat9[1], self.shared_bodyOrientationMat9[2]],
                         [self.shared_bodyOrientationMat9[3], self.shared_bodyOrientationMat9[4],
                             self.shared_bodyOrientationMat9[5]],
                         [self.shared_bodyOrientationMat9[6], self.shared_bodyOrientationMat9[7], self.shared_bodyOrientationMat9[8]]])

    def getOrientationQuat(self):
        return np.array([self.shared_bodyOrientationQuat[0],
                         self.shared_bodyOrientationQuat[1],
                         self.shared_bodyOrientationQuat[2],
                         self.shared_bodyOrientationQuat[3]])

    def qualisys_process(self, ip, body_id, shared_bodyPosition, shared_bodyVelocity,
                         shared_bodyOrientationQuat, shared_bodyOrientationMat9,
                         shared_bodyAngularVelocity, shared_timestamp):
        print("Qualisys process!")
        ''' This will run on a different process'''
        shared_timestamp.value = -1

        def on_packet(packet):
            """ Callback function that is called everytime a data packet arrives from QTM """
            position = packet.get_6d()[1][body_id][0]
            orientation = packet.get_6d()[1][body_id][1]
            timestamp = packet.timestamp * 1e-6

            # Get the last position and Rotation matrix from the shared memory.
            last_position = np.array(
                [shared_bodyPosition[0], shared_bodyPosition[1], shared_bodyPosition[2]])
            last_rotation = np.array([[shared_bodyOrientationMat9[0], shared_bodyOrientationMat9[1], shared_bodyOrientationMat9[2]],
                                      [shared_bodyOrientationMat9[3], shared_bodyOrientationMat9[4], shared_bodyOrientationMat9[5]],
                                      [shared_bodyOrientationMat9[6], shared_bodyOrientationMat9[7], shared_bodyOrientationMat9[8]]])
            last_se3 = pinocchio.SE3(last_rotation, last_position)

            # Get the new position and Rotation matrix from the motion capture.
            position = np.array([position.x, position.y, position.z]) * 1e-3
            rotation = np.array(orientation.matrix).reshape(3, 3).transpose()
            se3 = pinocchio.SE3(rotation, position)
            xyzquat = se3ToXYZQUAT(se3)

            # Get the position, Rotation matrix and Quaternion
            shared_bodyPosition[0] = xyzquat[0]
            shared_bodyPosition[1] = xyzquat[1]
            shared_bodyPosition[2] = xyzquat[2]
            shared_bodyOrientationQuat[0] = xyzquat[3]
            shared_bodyOrientationQuat[1] = xyzquat[4]
            shared_bodyOrientationQuat[2] = xyzquat[5]
            shared_bodyOrientationQuat[3] = xyzquat[6]

            shared_bodyOrientationMat9[0] = orientation.matrix[0]
            shared_bodyOrientationMat9[1] = orientation.matrix[3]
            shared_bodyOrientationMat9[2] = orientation.matrix[6]
            shared_bodyOrientationMat9[3] = orientation.matrix[1]
            shared_bodyOrientationMat9[4] = orientation.matrix[4]
            shared_bodyOrientationMat9[5] = orientation.matrix[7]
            shared_bodyOrientationMat9[6] = orientation.matrix[2]
            shared_bodyOrientationMat9[7] = orientation.matrix[5]
            shared_bodyOrientationMat9[8] = orientation.matrix[8]

            # Compute world velocity.
            if (shared_timestamp.value == -1):
                shared_bodyVelocity[0] = 0
                shared_bodyVelocity[1] = 0
                shared_bodyVelocity[2] = 0
                shared_bodyAngularVelocity[0] = 0.0
                shared_bodyAngularVelocity[1] = 0.0
                shared_bodyAngularVelocity[2] = 0.0
            else:
                dt = timestamp - shared_timestamp.value
                shared_bodyVelocity[0] = (position[0] - last_position[0])/dt
                shared_bodyVelocity[1] = (position[1] - last_position[1])/dt
                shared_bodyVelocity[2] = (position[2] - last_position[2])/dt
                bodyAngularVelocity = log(last_se3.rotation.T.dot(se3.rotation))/dt
                shared_bodyAngularVelocity[0] = bodyAngularVelocity[0]
                shared_bodyAngularVelocity[1] = bodyAngularVelocity[1]
                shared_bodyAngularVelocity[2] = bodyAngularVelocity[2]

            shared_timestamp.value = timestamp

        async def setup():
            """ Main function """
            connection = await qtm.connect(ip)
            if connection is None:
                print("no connection with qualisys!")
                return
            print("Connected")
            try:
                await connection.stream_frames(components=["6d"], on_packet=on_packet)
            except:
                print("connection with qualisys lost")

        asyncio.ensure_future(setup())
        asyncio.get_event_loop().run_forever()


def exampleOfUse():
    import time
    qc = QualisysClient(ip="140.93.16.160", body_id=0)
    for i in range(300):
        print(chr(27) + "[2J")
        print("position: ", qc.getPosition())
        print("quaternion: ", qc.getOrientationQuat())
        print("linear velocity: ", qc.getVelocity())
        print("angular velocity: ", qc.getAngularVelocity())
        time.sleep(0.3)
    print("killme!")


if __name__ == "__main__":
    exampleOfUse()
