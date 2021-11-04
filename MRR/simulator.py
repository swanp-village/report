from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from config.random import get_ring_rng
from MRR.Evaluator.evaluator import Evaluator
from MRR.gragh import Gragh
from MRR.logger import Logger
from MRR.mymath import lcm
from MRR.transfer_function import simulate_transfer_function


@dataclass
class SimulatorResult:
    name: str
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    label: str
    evaluation_result: np.float_


class Simulator:
    def __init__(self, is_focus: bool = False, init_graph: bool = True) -> None:
        self.logger = Logger()
        self.xs: list[npt.NDArray[np.float64]] = []
        self.ys: list[npt.NDArray[np.float64]] = []
        self.number_of_rings: int = 0
        if init_graph:
            self.graph = Gragh(is_focus)
            self.graph.create()

    def show(self) -> None:
        self.graph.show(self.logger.generate_image_path())


def simulate_MRR(
    simulator: Simulator,
    L: npt.NDArray[np.float64],
    K: npt.NDArray[np.float64],
    n_g: float,
    n_eff: float,
    eta: float,
    alpha: float,
    center_wavelength: float,
    FSR: float,
    min_ring_length: float,
    format: bool = False,
    simulate_one_cycle: bool = True,
    lambda_limit: npt.NDArray[np.float64] = np.array([]),
    name: str = "",
    label: str = "",
    skip_gragh: bool = False,
    skip_evaluation: bool = False,
    ignore_binary_evaluation: bool = False,
    seedsequence: np.random.SeedSequence = np.random.SeedSequence(),
    **config,
) -> SimulatorResult:
    print("eta:", eta)
    print("center_wavelength:", center_wavelength)
    print("n_eff:", n_eff)
    print("n_g:", n_g)
    if format:
        print("K:", K.round(2).tolist())
        print("L:", (L * 1e6).round(1).tolist())
    else:
        print("K:", K.tolist())
        print("L:", L.tolist())
    ring = Ring(center_wavelength, eta, FSR, min_ring_length, n_g, n_eff, seedsequence)
    N = ring.calculate_N(L)
    practical_FSR = ring.calculate_practical_FSR(N)

    if simulate_one_cycle:
        x = ring.calculate_x(practical_FSR)
    else:
        x = lambda_limit

    y = simulate_transfer_function(
        wavelength=x, L=L, K=K, alpha=alpha, eta=eta, n_eff=n_eff, n_g=n_g, center_wavelength=center_wavelength
    )

    if skip_evaluation:
        evaluation_result = np.float_(0)
    else:
        evaluator = Evaluator(x, y, config)
        evaluation_result = evaluator.evaluate_band(ignore_binary_evaluation)
        print("E:", evaluation_result)

    if not skip_gragh:
        simulator.logger.save_data_as_csv(x, y, name)
        simulator.graph.plot(x, y, label)
    return SimulatorResult(name, x, y, label, evaluation_result)


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

    def __init__(
        self,
        center_wavelength: float,
        eta: float,
        FSR: float,
        min_ring_length: float,
        n_g: float,
        n_eff: float,
        seedsequence: np.random.SeedSequence = np.random.SeedSequence(),
        **kwargs,
    ):
        self.center_wavelength = np.float_(center_wavelength)
        self.eta = np.float_(eta)
        self.FSR = np.float_(FSR)
        self.min_ring_length = np.float_(min_ring_length)
        self.n_g = np.float_(n_g)
        self.n_eff = np.float_(n_eff)
        self.rng = get_ring_rng(seedsequence)

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
        N_0 = np.int_(self.rng.integers(min_N_0, min_N_0 + 200))

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
