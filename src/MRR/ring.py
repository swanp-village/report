import numpy as np
from typing import List
from .math import lcm


def calculate_x(center_wavelength: float, FSR: float):
    return np.hstack((
        np.arange(center_wavelength - FSR / 2, center_wavelength, 1e-12),
        np.arange(center_wavelength, center_wavelength + FSR / 2, 1e-12)
    ))


def find_ring_length(center_wavelength: float, n: float, max_N=1000):
    N = np.arange(1, max_N)
    ring_length_list = N * center_wavelength / n
    FSR_list = center_wavelength / N

    return N, ring_length_list, FSR_list


def init_K(number_of_rings: int):
    return np.full(number_of_rings + 1, 0.5)


def calculate_practical_FSR(FSR_list: List[float]):
    return lcm(FSR_list)
