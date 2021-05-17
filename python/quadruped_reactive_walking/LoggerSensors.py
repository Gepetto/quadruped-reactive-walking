'''This class will log 1d array in Nd matrix from device and qualisys object'''
import numpy as np
from datetime import datetime as datetime
from time import time


class LoggerSensors():
    def __init__(self, device, qualisys=None, logSize=60e3, ringBuffer=False):
        self.ringBuffer = ringBuffer
        logSize = np.int(logSize)
        self.logSize = logSize
        self.i = 0
        nb_motors = device.nb_motors

        # Allocate the data:
        # IMU and actuators:
        self.q_mes = np.zeros([logSize, nb_motors])
        self.v_mes = np.zeros([logSize, nb_motors])
        self.torquesFromCurrentMeasurment = np.zeros([logSize, nb_motors])
        self.baseOrientation = np.zeros([logSize, 4])
        self.baseAngularVelocity = np.zeros([logSize, 3])
        self.baseLinearAcceleration = np.zeros([logSize, 3])
        self.baseAccelerometer = np.zeros([logSize, 3])

        # Motion capture:
        self.mocapPosition = np.zeros([logSize, 3])
        self.mocapVelocity = np.zeros([logSize, 3])
        self.mocapAngularVelocity = np.zeros([logSize, 3])
        self.mocapOrientationMat9 = np.zeros([logSize, 3, 3])
        self.mocapOrientationQuat = np.zeros([logSize, 4])

        # Timestamps
        self.tstamps = np.zeros(logSize)

    def sample(self, device, qualisys=None):
        if (self.i >= self.logSize):
            if self.ringBuffer:
                self.i = 0
            else:
                return

        # Logging from the device (data coming from the robot)
        self.q_mes[self.i] = device.q_mes
        self.v_mes[self.i] = device.v_mes
        self.baseOrientation[self.i] = device.baseOrientation
        self.baseAngularVelocity[self.i] = device.baseAngularVelocity
        self.baseLinearAcceleration[self.i] = device.baseLinearAcceleration
        self.baseAccelerometer[self.i] = device.baseAccelerometer
        self.torquesFromCurrentMeasurment[self.i] = device.torquesFromCurrentMeasurment

        # Logging from qualisys (motion capture)
        if qualisys is not None:
            self.mocapPosition[self.i] = qualisys.getPosition()
            self.mocapVelocity[self.i] = qualisys.getVelocity()
            self.mocapAngularVelocity[self.i] = qualisys.getAngularVelocity()
            self.mocapOrientationMat9[self.i] = qualisys.getOrientationMat9()
            self.mocapOrientationQuat[self.i] = qualisys.getOrientationQuat()
        else:  # Logging from PyBullet simulator through fake device
            self.mocapPosition[self.i] = device.baseState[0]
            self.mocapVelocity[self.i] = device.baseVel[0]
            self.mocapAngularVelocity[self.i] = device.baseVel[1]
            self.mocapOrientationMat9[self.i] = device.rot_oMb
            self.mocapOrientationQuat[self.i] = device.baseState[1]

        # Logging timestamp
        self.tstamps[self.i] = time()

        self.i += 1

    def saveAll(self, fileName="dataSensors"):
        date_str = datetime.now().strftime('_%Y_%m_%d_%H_%M')

        np.savez(fileName + date_str + ".npz",
                 q_mes=self.q_mes,
                 v_mes=self.v_mes,
                 baseOrientation=self.baseOrientation,
                 baseAngularVelocity=self.baseAngularVelocity,
                 baseLinearAcceleration=self.baseLinearAcceleration,
                 baseAccelerometer=self.baseAccelerometer,
                 torquesFromCurrentMeasurment=self.torquesFromCurrentMeasurment,
                 mocapPosition=self.mocapPosition,
                 mocapVelocity=self.mocapVelocity,
                 mocapAngularVelocity=self.mocapAngularVelocity,
                 mocapOrientationMat9=self.mocapOrientationMat9,
                 mocapOrientationQuat=self.mocapOrientationQuat,
                 tstamps=self.tstamps)
