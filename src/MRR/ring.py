import numpy as np
from typing import List
from .math import lcm
from math import isclose
from random import randrange, random, sample


def calculate_x(center_wavelength: float, FSR: float):
    return np.hstack((
        np.arange(center_wavelength - FSR / 2, center_wavelength, 1e-12),
        np.arange(center_wavelength, center_wavelength + FSR / 2, 1e-12)
    ))


def calculate_ring_length(N, center_wavelength, n):
    return N * center_wavelength / n


def calculate_FSR(N, center_wavelength):
    return center_wavelength / N


def find_ring_length(center_wavelength, n, max_N):
    N = np.arange(max_N)
    N[0] = 1
    ring_length_list = calculate_ring_length(N, center_wavelength, n)
    FSR_list = calculate_FSR(N, center_wavelength)

    return ring_length_list, FSR_list


def calculate_practical_FSR(FSR_list: List[float]):
    return lcm(FSR_list)


def init_N(
    number_of_rings: int,
    required_FSR: float,
    center_wavelength: int
):
    ratio = np.array(sample(range(2, 20), number_of_rings))
    base = randrange(100, 200)
    N = ratio * base
    i = 0
    while i < 10000:
        i = i + 1
        N = ratio * base
        FSR_list = calculate_FSR(N, center_wavelength)
        practical_FSR = calculate_practical_FSR(FSR_list)
        if practical_FSR > required_FSR:
            if i > 1000:
                break
            base = base + randrange(1, 3)
        else:
            if base > 3:
                base = base - randrange(1, 3)
            else:
                base = base + randrange(10, 20)

    N = ratio * base
    print(N)
    return N


def init_K(number_of_rings: int):
    return np.array([
        random()
        for _ in range(number_of_rings + 1)
    ])
