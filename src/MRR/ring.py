import numpy as np
from typing import List
from .math import lcm



def find_ring_length(resonant_wavelength: float, n: float, max_N=1000):
    N = np.arange(1, max_N)
    ring_length_list = N * resonant_wavelength / n
    FSR_list = resonant_wavelength / N

    return N, ring_length_list, FSR_list


def init_K(number_of_rings: int):
    return np.full(number_of_rings + 1, 0.5)


def calculate_practical_FSR(FSR_list: List[float]):
    return lcm(FSR_list)
