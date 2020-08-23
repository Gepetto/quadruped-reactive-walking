import numpy as np
import time
from loop import Loop


class SimulatorLoop(Loop):
    """
    Class used to call pybullet at a given frequency
    """
    def __init__(self, period, t_max):
        """
        Constructor
        :param period: the time step
        :param t_max: maximum simulation time
        """
        self.t = 0.0
        self.t_max = t_max
        self.period = period

    def trigger(self):
        super().__init__(self.period)

    def loop(self, signum, frame):
        self.t += self.period
        if self.t > self.t_max:
            self.stop()

        print("- Start loop -")
        print("- End loop -")


if __name__ == "__main__":

    # Start the control loop:
    sim_loop = SimulatorLoop(1.0, 10.0)
    sim_loop.trigger()

    print("-- FINAL --")
