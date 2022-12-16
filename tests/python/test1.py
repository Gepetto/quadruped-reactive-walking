import unittest

import quadruped_reactive_walking as qrw


class TestEstimator(unittest.TestCase):
    def test_constructor(self):
        qrw.Estimator()


if __name__ == "__main__":
    unittest.main()
