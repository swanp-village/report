from itertools import combinations_with_replacement
from typing import Union

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from config.model import OptimizationConfig, SimulationConfig

from .mymath import lcm


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

    def __init__(self, config: Union[SimulationConfig, OptimizationConfig]):
        self.center_wavelength = np.float_(config.center_wavelength)
        self.eta = np.float_(config.eta)
        self.FSR = np.float_(config.FSR)
        self.min_ring_length = np.float_(config.min_ring_length)
        self.number_of_rings = config.number_of_rings
        self.n_g = np.float_(config.n_g)
        self.n_eff = np.float_(config.n_eff)
        self.rng = config.root_rng

    @property
    def min_N(self) -> np.float_:
        return self.min_ring_length * self.n_eff / self.center_wavelength

    def calculate_x(self, FSR: np.float_) -> npt.NDArray[np.float_]:
        return np.hstack(
            (
                np.arange(self.center_wavelength - FSR / 2, self.center_wavelength, 1e-12),
                np.arange(self.center_wavelength, self.center_wavelength + FSR / 2, 1e-12),
            )
        )

    def calculate_ring_length(self, N: npt.NDArray[np.int_]) -> npt.NDArray[np.float_]:
        return N * self.center_wavelength / self.n_eff

    def calculate_FSR(self, N: npt.NDArray[np.int_]) -> npt.NDArray[np.float_]:
        return self.center_wavelength * self.n_eff / (self.n_g * N)

    def calculate_N(self, L: npt.NDArray[np.float_]) -> npt.NDArray[np.int_]:
        return np.round(L * self.n_eff / self.center_wavelength)

    def calculate_practical_FSR(self, N: npt.NDArray[np.int_]) -> np.float_:
        return lcm(self.calculate_FSR(N))

    def calculate_practical_FSR_from_L(self, L: npt.NDArray[np.float_]) -> np.float_:
        return self.calculate_practical_FSR(self.calculate_N(L))

    def init_ratio(self) -> npt.NDArray[np.int_]:
        n = self.number_of_rings
        p = [
            norm.cdf(-2),
            *[(norm.cdf(2 / (n - 2) * (x + 1)) - norm.cdf(2 / (n - 2) * x)) * 2 for x in range(n - 2)],
            norm.cdf(-2),
        ]
        a = np.arange(1, n + 1)
        number_of_types = self.rng.choice(a, p=p)
        base = self.rng.choice(np.arange(2, 30), number_of_types, replace=False)
        reciprocal_of_ratio: npt.NDArray[np.int_] = self.rng.choice(base, n)
        while np.unique(reciprocal_of_ratio).size != number_of_types:
            reciprocal_of_ratio = self.rng.choice(base, n)
        ratio: npt.NDArray[np.int_] = (np.lcm.reduce(reciprocal_of_ratio) / reciprocal_of_ratio).astype(np.int_)

        return ratio

    def optimize_N(self) -> npt.NDArray[np.int_]:
        ratio = self.init_ratio()
        min_N_0 = np.ceil(self.min_N / ratio).min().astype(np.int_)
        N_0 = self.rng.integers(min_N_0, 100, dtype=np.int_)

        for _ in range(10000):
            a = np.power(np.arange(3), 4)
            neighborhood_N_0 = np.concatenate((-np.flip(a)[: a.size - 1], a)) + N_0
            neighborhood_N_0 = neighborhood_N_0[neighborhood_N_0 >= min_N_0]
            E = []
            for n_0 in np.nditer(neighborhood_N_0):
                N = ratio * n_0
                FSR = self.calculate_practical_FSR(N)
                E.append(np.square(FSR - self.FSR))

            best_E_index = np.argmin(E)
            best_N_0 = neighborhood_N_0[best_E_index]
            if best_N_0 == N_0:
                break
            else:
                N_0 = best_N_0
        N = ratio * N_0
        return N

    def init_K(self) -> npt.NDArray[np.float_]:
        return np.array([self.rng.uniform(0, self.eta) for _ in range(self.number_of_rings + 1)])
