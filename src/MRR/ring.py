import numpy as np


def find_ring_length(resonant_wavelength, n, max_N=1000):
    N = np.arange(1, max_N)
    ring_length_list = N * resonant_wavelength / 3.3938
    FSR_list = resonant_wavelength / N

    return N, ring_length_list, FSR_list
