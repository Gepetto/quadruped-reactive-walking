import signal, time
from abc import ABCMeta, abstractmethod

## Code copied from mlp to avoid large dependency. Should be put in smaller package instead


class Loop(metaclass=ABCMeta):
    """
    Astract Class to allow users to execute self.loop at a given frequency
    with a timer while self.run can do something else.
    """
    def __init__(self, period):
        self.period = period
        signal.signal(signal.SIGALRM, self.loop)
        signal.setitimer(signal.ITIMER_REAL, period, period)
        self.run()

    def stop(self):
        signal.setitimer(signal.ITIMER_REAL, 0)
        raise KeyboardInterrupt  # our self.run is waiting for this.

    def run(self):
        # Default implementation: don't do anything
        try:
            time.sleep(1e9)
        except KeyboardInterrupt:
            pass

    @abstractmethod
    def loop(self, signum, frame):
        ...
