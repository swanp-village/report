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
        N, ring_length_list, FSR_list = find_ring_length(resonant_wavelength, n)

        with self.subTest(
            N=[N[300], N[600]],
            ring_length_list=[ring_length_list[300], ring_length_list[600]],
            FSR_list=[FSR_list[300], FSR_list[600]]
        ):
            actual = calculate_practical_FSR([FSR_list[300], FSR_list[600]])
            expect = 5.166666666666667e-09
            self.assertEqual(actual, expect)

        with self.subTest(
            N=[N[300], N[800]],
            ring_length_list=[ring_length_list[300], ring_length_list[800]],
            FSR_list=[FSR_list[300], FSR_list[800]]
        ):
            actual = calculate_practical_FSR([FSR_list[300], FSR_list[800]])
            expect = 1.5979381443299004e-08
            self.assertEqual(actual, expect)

        with self.subTest(
            N=[N[300], N[900]],
            ring_length_list=[ring_length_list[300], ring_length_list[900]],
            FSR_list=[FSR_list[300], FSR_list[900]]
        ):
            actual = calculate_practical_FSR([FSR_list[300], FSR_list[900]])
            expect = 5.183946488294314e-09
            self.assertEqual(actual, expect)

        with self.subTest(
            N=[N[350], N[558]],
            ring_length_list=[ring_length_list[350], ring_length_list[558]],
            FSR_list=[FSR_list[350], FSR_list[558]]
        ):
            actual = calculate_practical_FSR([FSR_list[350], FSR_list[558]])
            expect = 2.38461538461539e-08
            self.assertEqual(actual, expect)

        with self.subTest(
            N=[N[500], N[700], N[900]],
            ring_length_list=[ring_length_list[500], ring_length_list[700], ring_length_list[900]],
            FSR_list=[FSR_list[500], FSR_list[700], FSR_list[900]]
        ):
            actual = calculate_practical_FSR([FSR_list[500], FSR_list[700], FSR_list[900]])
            expect = 2.38461538461539e-08
            self.assertEqual(actual, expect)


if __name__ == '__main__':
    unittest.main()
