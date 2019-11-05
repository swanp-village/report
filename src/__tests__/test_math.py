import unittest
from MRR import math


class Test(unittest.TestCase):
    def test_calc_eps(self):
        test_case = [
            (1, 0.001, 0.1),
            (0.1, 0.001, 0.01),
            (0.01, 0.001, 0.001)
        ]

        for x, y, expect in test_case:
            with self.subTest(x=x):
                self.assertEqual(math.calc_eps(x, y), expect)

    def test_is_zero(self):
        test_case = [
            (1, 0.01, False),
            (0.1, 0.01, False),
            (0.01, 0.01, True),
            (0.001, 0.01, True),
        ]

        for x, eps, expect in test_case:
            with self.subTest(x=x, eps=eps):
                self.assertEqual(math.is_zero(x, eps), expect)

    def test_mod(self):
        test_case = [
            (3, 2, 1),
            (4, 2, 0),
            (4.5, 1.5, 0),
            (3.9, 2.6, 1.3),
            (2.5, 1.2, 0.1)
        ]

        for x, y, expect in test_case:
            with self.subTest(x=x, y=y):
                self.assertAlmostEqual(math.mod(x, y), expect, places=7)

    def test_gcd(self):
        test_case = [
            (2, 3, 1),
            (2, 4, 2),
            (1.5, 4.5, 1.5),
            (3.9, 2.6, 1.3)
        ]

        for x, y, expect in test_case:
            eps = math.calc_eps(x, y)
            with self.subTest(x=x, y=y, eps=eps):
                self.assertAlmostEqual(math._gcd(x, y, eps), expect, places=7)

    def test_lcm(self):
        test_case = [
            ([2, 3], 6),
            ([2, 4], 4),
            ([1.5, 4.5], 4.5),
            ([3.9, 2.6], 7.8)
        ]

        for xs, expect in test_case:
            with self.subTest(xs=xs):
                self.assertAlmostEqual(math.lcm(xs), expect, places=7)

if __name__ == '__main__':
    unittest.main()
