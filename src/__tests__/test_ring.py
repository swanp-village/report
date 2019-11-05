import unittest
from MRR.ring import find_ring_length


class Test(unittest.TestCase):
    def test_find_ring_length(self):
        self.assertEqual(find_ring_length(1550e-9, 3.3938), 999)


if __name__ == '__main__':
    unittest.main()
