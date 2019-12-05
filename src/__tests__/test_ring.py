import unittest
from MRR.ring import calculate_practical_FSR, find_ring_length


class Test(unittest.TestCase):
    def test_find_ring_length(self):
        n = 3.3938
        resonant_wavelength = 1550e-9
        N, ring_length_list, FSR_list = find_ring_length(resonant_wavelength, n)
        actual = len(N)
        expect = 999
        self.assertEqual(actual, expect)

    def test_calculate_practical_FSR(self):
        n = 3.3938
        resonant_wavelength = 1550e-9
        N, ring_length_list, FSR_list = find_ring_length(resonant_wavelength, n, 10000)
        test_case = [
            ([300, 600], 5.166666666666667e-09),
            ([300, 800], 1.5979381443299004e-08),
            ([300, 900], 5.183946488294314e-09),
            ([350, 558], 2.38461538461539e-08),
            ([500, 700, 900], 2.38461538461539e-08),
        ]

        for index, expect in test_case:
            with self.subTest(
                N=N[index],
                ring_length_list=ring_length_list[index],
                FSR_list=FSR_list[index]
            ):
                actual = calculate_practical_FSR(FSR_list[index])
                self.assertEqual(actual, expect)


if __name__ == '__main__':
    unittest.main()
