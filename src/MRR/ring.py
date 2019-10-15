import numpy as np


def find_ring_length(resonant_wavelength, max_N=1000):
    lst = np.arange(0, max_N + 1) * resonant_wavelength

    return lst
