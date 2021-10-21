from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from config.model import BaseConfig, SimulationConfig
from MRR.Evaluator.evaluator import Evaluator
from MRR.gragh import Gragh
from MRR.logger import Logger
from MRR.mymath import lcm


@dataclass
class SimulatorResult:
    name: str
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    label: str
    evaluation_result: np.float_


class Simulator:
    def __init__(self, is_focus: bool = False) -> None:
        self.logger = Logger()
        self.xs: list[npt.NDArray[np.float64]] = []
        self.ys: list[npt.NDArray[np.float64]] = []
        self.number_of_rings: int = 0
        self.graph = Gragh(is_focus)
        self.graph.create()

    def simulate(
        self, config: SimulationConfig, skip_gragh: bool = False, skip_evaluation: bool = False
    ) -> SimulatorResult:
        mrr = TransferFunction(config.L, config.K, config)
        mrr.print_parameters()
        if config.format:
            print("K:", config.K.round(2).tolist())
            print("L:", (config.L * 1e6).round(1).tolist())
        else:
            print("K:", config.K.tolist())
            print("L:", config.L.tolist())
        ring = Ring(config)
        N = ring.calculate_N(config.L)
        FSR = ring.calculate_practical_FSR(N)

        if config.simulate_one_cycle:
            x = ring.calculate_x(FSR)
        else:
            x = config.lambda_limit

        y = mrr.simulate(x)

        if skip_evaluation:
            evaluation_result = np.float_(0)
        else:
            evaluator = Evaluator(x, y, config)
            evaluation_result = evaluator.evaluate_band()
            print("E:", evaluation_result)

        if not skip_gragh:
            self.logger.save_data_as_csv(x, y, config.name)
            self.graph.plot(x, y, config.label)
        return SimulatorResult(config.name, x, y, config.label, evaluation_result)

    def show(self) -> None:
        self.graph.show(self.logger.generate_image_path())


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
        self.rng = config.get_ring_rng()

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
        return np.float_(N) * self.center_wavelength / self.n_eff  # type: ignore

    def calculate_FSR(self, N: npt.NDArray[np.int_]) -> npt.NDArray[np.float_]:
        return self.center_wavelength * self.n_eff / (self.n_g * N)  # type: ignore

    def calculate_N(self, L: npt.NDArray[np.float_]) -> npt.NDArray[np.int_]:
        return np.round(L * self.n_eff / self.center_wavelength)  # type: ignore

    def calculate_practical_FSR(self, N: npt.NDArray[np.int_]) -> np.float_:
        return lcm(self.calculate_FSR(N))

    def calculate_practical_FSR_from_L(self, L: npt.NDArray[np.float_]) -> np.float_:
        return self.calculate_practical_FSR(self.calculate_N(L))

    def calculate_min_N_0(self, ratio: npt.NDArray[np.int_]) -> np.int_:
        result = np.ceil(self.min_N / ratio).max()
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
        while np.unique(reciprocal_of_ratio).size != number_of_types:  # type: ignore
            reciprocal_of_ratio = self.rng.choice(base, number_of_rings)
        ratio: npt.NDArray[np.int_] = (np.lcm.reduce(reciprocal_of_ratio) / reciprocal_of_ratio).astype(np.int_)
        return ratio

    def optimize_N(self, number_of_rings: int) -> npt.NDArray[np.int_]:
        ratio = self.init_ratio(number_of_rings)
        min_N_0 = self.calculate_min_N_0(ratio)
        N_0 = np.int_(self.rng.integers(min_N_0, 100))

        for _ in range(10000):
            a: npt.NDArray[np.int_] = np.power(np.arange(3, dtype=np.int_), 4)
            neighborhood_N_0: npt.NDArray[np.int_] = np.concatenate((-np.flip(a)[: a.size - 1], a)) + N_0  # type:ignore
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


class TransferFunction:
    """Simulator of the transfer function of the MRR filter.

    Args:
        L (List[float]): List of the round-trip length.
        K (List[float]): List of the coupling rate.
        config (Dict[str, Any]): Configuration of the MRR.
            Keys:
                eta (float): The coupling loss coefficient.
                n_eff (float): The effective refractive index.
                n_g (float): The group index.
                alpha (float): The propagation loss coefficient.
    Attributes:
        L (List[float]): List of the round-trip length.
        K (List[float]): List of the coupling rate.
        eta (float): The coupling loss coefficient.
        n_eff (float): The effective refractive index.
        n_g (float): The group index.
        a (List[float]): List of the propagation loss.
    """

    def __init__(self, L: npt.ArrayLike, K: npt.ArrayLike, config: BaseConfig) -> None:
        self.L: npt.NDArray[np.float64] = np.array(L)
        self.K: npt.NDArray[np.float64] = np.array(K)
        self.center_wavelength: float = config.center_wavelength
        self.eta: float = config.eta
        self.n_eff: float = config.n_eff
        self.n_g: float = config.n_g
        self.a: npt.NDArray[np.float64] = np.exp(-config.alpha * L)

    def _C(self, K_k: float) -> npt.NDArray[np.float64]:
        C: npt.NDArray[np.float64] = (
            1
            / (-1j * self.eta * np.sqrt(K_k))
            * np.array([[1, -self.eta * np.sqrt(self.eta - K_k)], [np.sqrt(self.eta - K_k) * self.eta, -self.eta ** 2]])
        )
        return C

    def _R(self, a_k: float, L_k: float, wavelength: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        N_k = np.round(L_k * self.n_eff / self.center_wavelength)
        shifted_center_wavelength = L_k * self.n_eff / N_k
        x = (
            1j
            * np.pi
            * L_k
            * self.n_g
            * (wavelength - shifted_center_wavelength)
            / shifted_center_wavelength
            / shifted_center_wavelength
        )
        return np.array([[np.exp(x) / np.sqrt(a_k), 0], [0, np.exp(-x) * np.sqrt(a_k)]], dtype="object")

    def _reverse(self, arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        revesed_arr: npt.NDArray[np.float64] = arr[::-1]
        return revesed_arr

    def _M(self, wavelength: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        product = np.identity(2)
        for _K, _a, _L in zip(self._reverse(self.K[1:]), self._reverse(self.a), self._reverse(self.L)):
            product = np.dot(product, self._C(_K))
            product = np.dot(product, self._R(_a, _L, wavelength))
        product = np.dot(product, self._C(self.K[0]))
        return product

    def _D(self, wavelength: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        D: npt.NDArray[np.float64] = 1 / self._M(wavelength)[0, 0]
        return D

    def print_parameters(self) -> None:
        print("eta:", self.eta)
        print("center_wavelength:", self.center_wavelength)
        print("n_eff:", self.n_eff)
        print("n_g:", self.n_g)
        print("a:", self.a)

    def simulate(self, wavelength: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        y: npt.NDArray[np.float64] = 20 * np.log10(np.abs(self._D(wavelength)))
        return y.reshape(y.size)
