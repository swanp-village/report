from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from MRR.evaluator import evaluate_band
from MRR.graph import Graph
from MRR.logger import Logger
from MRR.mymath import lcm
from MRR.transfer_function import simulate_transfer_function




@dataclass
class SimulatorResult:
    name: str
    x: npt.NDArray[np.float_]
    y: npt.NDArray[np.float_]
    label: str
    evaluation_result: np.float_


class Accumulator:
    def __init__(self, logger: Logger = None, is_focus: bool = False, init_graph: bool = True) -> None:
        if logger is None:
            self.logger = Logger()
        else:
            self.logger = logger
        self.number_of_rings: int = 0
        if init_graph:
            self.graph = Graph(is_focus)
            self.graph.create()

    def show(self) -> None:
        self.graph.show(self.logger.generate_image_path())


def simulate_MRR(
    accumulator: Accumulator,
    L: npt.NDArray[np.float_],
    K: npt.NDArray[np.float_],
    n_eff: float,
    n_g: float,
    eta: float,
    alpha: float,
    center_wavelength: float,
    length_of_3db_band: float,
    max_crosstalk: float,
    H_i: float,
    H_p: float,
    H_s: float,
    r_max: float,
    weight: list[float],
    format: bool = False,
    simulate_one_cycle: bool = False,
    lambda_limit: npt.NDArray[np.float_] = np.array([]),
    name: str = "",
    label: str = "",
    skip_graph: bool = False,
    skip_evaluation: bool = False,
    ignore_binary_evaluation: bool = False,
) -> SimulatorResult:
    #N = calculate_N(center_wavelength=center_wavelength, n_eff=n_eff, L=L)
    N = [78,78,78,117,117,117]
    practical_FSR = calculate_practical_FSR(center_wavelength=center_wavelength, n_eff=n_eff, n_g=n_g, N=N)

    if simulate_one_cycle:
        x = calculate_x(center_wavelength=center_wavelength, FSR=practical_FSR)
    else:
        x = lambda_limit

    y = simulate_transfer_function(
        wavelength=x, L=L, K=K, alpha=alpha, eta=eta, n_eff=n_eff, n_g=n_g, center_wavelength=center_wavelength
    )
   
    if skip_evaluation:
        evaluation_result = np.float_(0)
    else:
        evaluation_result = evaluate_band(
            x=x,
            y=y,
            center_wavelength=center_wavelength,
            length_of_3db_band=length_of_3db_band,
            max_crosstalk=max_crosstalk,
            H_p=H_p,
            H_s=H_s,
            H_i=H_i,
            r_max=r_max,
            weight=weight,
            ignore_binary_evaluation=ignore_binary_evaluation,
        )
        
      
    accumulator.logger.print_parameters(K=K, L=L, N=N, FSR=practical_FSR, E=evaluation_result, format=format)

    if not skip_graph:
        accumulator.logger.save_data_as_csv(x, y, name)
        accumulator.graph.plot(x, y, label)
    return SimulatorResult(name, x, y, label, evaluation_result)


def calc_min_N(
    center_wavelength: float,
    min_ring_length: float,
    n_eff: float,
) -> np.float_:
    center_wavelength_ = np.float_(center_wavelength)
    min_ring_length_ = np.float_(min_ring_length)
    n_eff_ = np.float_(n_eff)
    return min_ring_length_ * n_eff_ / center_wavelength_


def calculate_x(center_wavelength: float, FSR: np.float_) -> npt.NDArray[np.float_]:
    center_wavelength_ = np.float_(center_wavelength)
    return np.hstack(
        (
            np.arange(center_wavelength_ - FSR / 2, center_wavelength_, 1e-12),
            np.arange(center_wavelength_, center_wavelength_ + FSR / 2, 1e-12),
        )
    )


def calculate_ring_length(center_wavelength: float, n_eff: float, N: npt.NDArray[np.int_]) -> npt.NDArray[np.float_]:
    center_wavelength_ = np.float_(center_wavelength)
    n_eff_ = np.float_(n_eff)
    N_ = np.float_(N)
    return N_ * center_wavelength_ / n_eff_  # type: ignore


def calculate_FSR(
    center_wavelength: float, n_eff: float, n_g: float, N: npt.NDArray[np.int_]
) -> npt.NDArray[np.float_]:
    center_wavelength_ = np.float_(center_wavelength)
    n_eff_ = np.float_(n_eff)
    n_g_ = np.float_(n_g)
    N_ = np.float_(N)
    return center_wavelength_ * n_eff_ / (n_g_ * N_)  # type: ignore


def calculate_N(center_wavelength: float, n_eff: float, L: npt.NDArray[np.float_]) -> npt.NDArray[np.int_]:
    center_wavelength_ = np.float_(center_wavelength)
    n_eff_ = np.float_(n_eff)
    return np.round(L * n_eff_ / center_wavelength_)  # type: ignore


def calculate_practical_FSR(center_wavelength: float, n_eff: float, n_g: float, N: npt.NDArray[np.int_]) -> np.float_:
    return lcm(calculate_FSR(center_wavelength=center_wavelength, n_eff=n_eff, n_g=n_g, N=N))


def calculate_practical_FSR_from_L(
    center_wavelength: float, n_eff: float, n_g: float, L: npt.NDArray[np.float_]
) -> np.float_:
    return calculate_practical_FSR(
        center_wavelength=center_wavelength,
        n_eff=n_eff,
        n_g=n_g,
        N=calculate_N(center_wavelength=center_wavelength, n_eff=n_eff, L=L),
    )


def calculate_min_N_0(
    center_wavelength: float, min_ring_length: float, n_eff: float, ratio: npt.NDArray[np.int_]
) -> np.int_:
    result = np.ceil(
        calc_min_N(center_wavelength=center_wavelength, min_ring_length=min_ring_length, n_eff=n_eff) / ratio
    ).max()
    return np.int_(result)


def init_ratio(number_of_rings: int, rng: np.random.Generator) -> npt.NDArray[np.int_]:
    p = [
        norm.cdf(-2),
        *[
            (norm.cdf(2 / (number_of_rings - 2) * (x + 1)) - norm.cdf(2 / (number_of_rings - 2) * x)) * 2
            for x in range(number_of_rings - 2)
        ],
        norm.cdf(-2),
    ]
    if number_of_rings == 2:
        p = [norm.cdf(-2), norm.sf(-2)]
    a = np.arange(1, number_of_rings + 1)
    number_of_types = rng.choice(a, p=p)
    base = rng.choice(np.arange(2, 20), number_of_types, replace=False)
    reciprocal_of_ratio: npt.NDArray[np.int_] = rng.choice(base, number_of_rings)
    while np.unique(reciprocal_of_ratio).size != number_of_types:  # type: ignore
        reciprocal_of_ratio = rng.choice(base, number_of_rings)
    ratio: npt.NDArray[np.int_] = (np.lcm.reduce(reciprocal_of_ratio) / reciprocal_of_ratio).astype(np.int_)
    return ratio


def optimize_N(
    center_wavelength: float,
    min_ring_length: float,
    n_eff: float,
    n_g: float,
    number_of_rings: int,
    FSR: float,
    rng: np.random.Generator,
) -> npt.NDArray[np.int_]:
    ratio = init_ratio(number_of_rings=number_of_rings, rng=rng)
    min_N_0 = calculate_min_N_0(
        center_wavelength=center_wavelength, min_ring_length=min_ring_length, n_eff=n_eff, ratio=ratio
    )
    N_0 = np.int_(rng.integers(min_N_0, min_N_0 + 10))

    for _ in range(10000):
        a: npt.NDArray[np.int_] = np.power(np.arange(3, dtype=np.int_), 4)
        neighborhood_N_0: npt.NDArray[np.int_] = np.concatenate((-np.flip(a)[: a.size - 1], a)) + N_0  # type:ignore
        neighborhood_N_0 = neighborhood_N_0[neighborhood_N_0 >= min_N_0]
        neighborhood_N: npt.NDArray[np.int_] = neighborhood_N_0.reshape(1, -1).T * ratio
        E = [
            np.square(calculate_practical_FSR(center_wavelength=center_wavelength, n_eff=n_eff, n_g=n_g, N=n) - FSR)
            for n in neighborhood_N
        ]
        best_E_index = np.argmin(E)
        best_N_0: np.int_ = neighborhood_N_0[best_E_index]
        if best_N_0 == N_0:
            break
        else:
            N_0 = best_N_0
    N: npt.NDArray[np.int_] = ratio * N_0
    return N
