import unittest

import quadruped_reactive_walking as exa


class TestAdder(unittest.TestCase):
    def test_adder_integers(self):
        self.assertEqual(exa.add(4, 3), 7)
        self.assertEqual(exa.sub(4, 3), 1)


if __name__ == '__main__':
    unittest.main()
