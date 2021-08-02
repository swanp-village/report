import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from config.model import BaseConfig
from MRR.Simulator.mymath import lcm


class Ring:
    """
    Args:
        config (Dict[str, Any]): Configuration of the MRR.
            Keys:
                center_wavelength (float): The center wavelength.
                eta (float): The coupling loss coefficient.
                FSR (float): The required FSR.
                min_ring_length (float): The minimum round-trip length.
                n_g (float): The group index.
                n_eff (float): The equivalent refractive index.
    Attributes:
        center_wavelength (float): The center wavelength.
        eta (float): The coupling loss coefficient.
        FSR (float): The required FSR.
        min_ring_length (float): The minimum round-trip length.
        n_g (float): The group index.
        n_eff (float): The equivalent refractive index.
    """

    def __init__(self, config: BaseConfig):
        self.center_wavelength = np.float_(config.center_wavelength)
        self.eta = np.float_(config.eta)
        self.FSR = np.float_(config.FSR)
        self.min_ring_length = np.float_(config.min_ring_length)
        self.n_g = np.float_(config.n_g)
        self.n_eff = np.float_(config.n_eff)
        self.rng = np.random.default_rng()

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
        return np.float_(N) * self.center_wavelength / self.n_eff

    def calculate_FSR(self, N: npt.NDArray[np.int_]) -> npt.NDArray[np.float_]:
        return self.center_wavelength * self.n_eff / (self.n_g * N)

    def calculate_N(self, L: npt.NDArray[np.float_]) -> npt.NDArray[np.int_]:
        return np.round(L * self.n_eff / self.center_wavelength)

    def calculate_practical_FSR(self, N: npt.NDArray[np.int_]) -> np.float_:
        print(N)
        return lcm(self.calculate_FSR(N))

    def calculate_practical_FSR_from_L(self, L: npt.NDArray[np.float_]) -> np.float_:
        return self.calculate_practical_FSR(self.calculate_N(L))

    def calculate_min_N_0(self, ratio: npt.NDArray[np.int_]) -> np.int_:
        result = np.ceil(self.min_N / ratio).min()
        return np.int_(result)

    def init_ratio(self, number_of_rings: int) -> npt.NDArray[np.int_]:
        p = [
            norm.cdf(-2),
            *[
                (norm.cdf(2 / (number_of_rings - 2) * (x + 1)) - norm.cdf(2 / (number_of_rings - 2) * x)) * 2
                for x in range(number_of_rings - 2)
            ],
            norm.cdf(-2),
        ]
        a = np.arange(1, number_of_rings + 1)
        number_of_types = self.rng.choice(a, p=p)
        base = self.rng.choice(np.arange(2, 30), number_of_types, replace=False)
        reciprocal_of_ratio: npt.NDArray[np.int_] = self.rng.choice(base, number_of_rings)
        while np.unique(reciprocal_of_ratio).size != number_of_types:
            reciprocal_of_ratio = self.rng.choice(base, number_of_rings)
        ratio: npt.NDArray[np.int_] = (np.lcm.reduce(reciprocal_of_ratio) / reciprocal_of_ratio).astype(np.int_)
        return ratio

    def optimize_N(self, number_of_rings: int) -> npt.NDArray[np.int_]:
        ratio = self.init_ratio(number_of_rings)
        min_N_0 = self.calculate_min_N_0(ratio)
        N_0 = np.int_(self.rng.integers(min_N_0, 100))

        for _ in range(10000):
            a: npt.NDArray[np.int_] = np.power(np.arange(3, dtype=np.int_), 4)
            neighborhood_N_0: npt.NDArray[np.int_] = np.concatenate((-np.flip(a)[: a.size - 1], a)) + N_0
            neighborhood_N_0 = neighborhood_N_0[neighborhood_N_0 >= min_N_0]
            neighborhood_N: npt.NDArray[np.int_] = neighborhood_N_0.reshape(1, -1).T * ratio
            E = [np.square(self.calculate_practical_FSR(n) - self.FSR) for n in neighborhood_N]
            best_E_index = np.argmin(E)
            best_N_0: np.int_ = neighborhood_N_0[best_E_index]
            if best_N_0 == N_0:
                break
            else:
                N_0 = best_N_0
        N: npt.NDArray[np.int_] = ratio * N_0
        return N
