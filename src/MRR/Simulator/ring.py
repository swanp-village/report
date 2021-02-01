import numpy as np
from .mymath import lcm
from random import randrange, sample, uniform, choice
from itertools import combinations_with_replacement
from scipy.stats import norm
from math import ceil


class Ring:
    """
    Args:
        config (Dict[str, Any]): Configuration of the MRR.
            Keys:
                center_wavelength (float): The center wavelength.
                eta (float): The coupling loss coefficient.
                FSR (float): The required FSR.
                min_ring_length (float): The minimum round-trip length.
                number_of_rings (int): Number of rings. The ring order.
                n_g (float): The group index.
                n_eff (float): The equivalent refractive index.
    Attributes:
        center_wavelength (float): The center wavelength.
        eta (float): The coupling loss coefficient.
        FSR (float): The required FSR.
        min_ring_length (float): The minimum round-trip length.
        number_of_rings (int): Number of rings. The ring order.
        n_g (float): The group index.
        n_eff (float): The equivalent refractive index.
    """
    def __init__(self, config):
        self.center_wavelength = config['center_wavelength']
        self.eta = config['eta']
        self.FSR = config['FSR']
        self.min_ring_length = config['min_ring_length']
        self.number_of_rings = config['number_of_rings']
        self.n_g = config['n_g']
        self.n_eff = config['n_eff']

    def calculate_x(self, FSR):
        return np.hstack((
            np.arange(self.center_wavelength - FSR / 2, self.center_wavelength, 1e-12),
            np.arange(self.center_wavelength, self.center_wavelength + FSR / 2, 1e-12)
        ))

    def calculate_ring_length(self, N):
        return N * self.center_wavelength / self.n_eff

    def calculate_min_N(self):
        return self.min_ring_length * self.n_eff / self.center_wavelength

    def calculate_FSR(self, N):
        return self.center_wavelength * self.n_eff / (self.n_g * N)

    def calculate_N(self, L):
        return np.round(L * self.n_eff / self.center_wavelength)

    def find_ring_length(self, max_N):
        N = np.arange(max_N)
        N[0] = 1
        ring_length_list = self.calculate_ring_length(N)
        FSR_list = self.calculate_FSR(N)

        return ring_length_list, FSR_list

    def calculate_practical_FSR(self, N):
        return lcm(self.calculate_FSR(N))

    def init_ratio(self):
        n = self.number_of_rings
        p = [
            norm.cdf(-2),
            *[
                (norm.cdf(2 / (n - 2) * (x + 1)) - norm.cdf(2 / (n - 2) * x)) * 2
                for x in range(n - 2)
            ],
            norm.cdf(-2)
        ]
        a = np.arange(1, n + 1)
        number_of_different_perimeters = np.random.choice(a, p=p)
        perimeter_range = range(number_of_different_perimeters)
        combinations_of_perimeters = [
            x
            for x in combinations_with_replacement(perimeter_range, n)
            if set(x) == set(perimeter_range)
        ]
        c = choice(combinations_of_perimeters)
        base_ratio = sample(range(2, 30), number_of_different_perimeters)
        base_ratio = (np.lcm.reduce(base_ratio) / base_ratio).astype(int)
        ratio = [base_ratio[x] for x in c]
        ratio = sample(ratio, len(ratio))

        return np.array(ratio)

    def init_N(self):
        ratio = self.init_ratio()
        N = self.optimize_N(ratio)

        return N

    def optimize_N(self, ratio):
        rand_start = 1
        rand_end = 3
        min_N_0 = ceil(self.calculate_min_N() / min(ratio) + rand_end)
        N_0 = randrange(100, 200)

        for i in range(10000):
            N = ratio * N_0
            FSR_e = self.calculate_practical_FSR(N)
            if i > 7000 and 0.99 < FSR_e / self.FSR < 1.01:
                return N
            if N_0 < min_N_0:
                N = ratio * min_N_0
                return N
            if FSR_e > self.FSR:
                N_0 = N_0 + randrange(rand_start, rand_end)
            else:
                N_0 = N_0 - randrange(rand_start, rand_end)

        return N


    def init_K(self):
        return np.array([
            uniform(0, self.eta)
            for _ in range(self.number_of_rings + 1)
        ])
