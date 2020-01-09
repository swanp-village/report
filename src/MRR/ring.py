import numpy as np
from typing import List
from .math import lcm
from random import randrange, random


def calculate_x(center_wavelength: float, FSR: float):
    return np.hstack((
        np.arange(center_wavelength - FSR / 2, center_wavelength, 1e-12),
        np.arange(center_wavelength, center_wavelength + FSR / 2, 1e-12)
    ))


def find_ring_length(center_wavelength: float, n: float, max_N):
    N = np.arange(1, max_N)
    ring_length_list = N * center_wavelength / n
    FSR_list = center_wavelength / N

    return N, ring_length_list, FSR_list


def init_N(number_of_rings: int, max_N: int):
    return np.array([
        randrange(1, max_N - 1)
        for _ in range(number_of_rings)
    ])


def init_K(number_of_rings: int):
    return np.array([
        random()
        for _ in range(number_of_rings + 1)
    ])


def calculate_practical_FSR(FSR_list: List[float]):
    return lcm(FSR_list)
