import numpy as np
from util.math import lcm


def find_ring_length(resonant_wavelength, n, max_N=1000):
    N = np.arange(1, max_N)
    ring_length_list = N * resonant_wavelength / n
    FSR_list = resonant_wavelength / N

    return N, ring_length_list, FSR_list



def calculate_practical_FSR(FSR_list):
    return lcm(FSR_list)
