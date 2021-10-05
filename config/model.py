from dataclasses import dataclass, field
from typing import Sequence, Union

import numpy as np
import numpy.typing as npt


@dataclass
class BaseConfig:
    entropy: Union[None, int, Sequence[int]] = None

    eta: float = 0.996
    alpha: float = 52.96
    n_eff: float = 2.2
    n_g: float = 4.4

    center_wavelength: float = 1550e-9
    FSR: float = 20e-9
    length_of_3db_band: float = 1e-9

    min_ring_length: float = 50e-6
    max_crosstalk: float = -30
    H_p: float = -20
    H_s: float = -60
    H_i: float = -10
    r_max: float = 5
    weight: list[float] = field(default_factory=list[float])

    def __post_init__(self) -> None:
        if len(self.weight) == 0:
            self.weight = [1.0, 3.5, 1.0, 5.0, 3.5, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    @property
    def seedsequence(self) -> np.random.SeedSequence:
        return np.random.SeedSequence(self.entropy)

    def __get_root_bit_generator(self) -> np.random.PCG64DXSM:
        return np.random.PCG64DXSM(self.seedsequence)

    def get_differential_evolution_rng(self) -> np.random.Generator:
        return np.random.Generator(self.__get_root_bit_generator().jumped(1))

    def get_multi_start_local_search_rng(self) -> np.random.Generator:
        return np.random.Generator(self.__get_root_bit_generator().jumped(2))

    def get_ring_rng(self) -> np.random.Generator:
        return np.random.Generator(self.__get_root_bit_generator().jumped(3))

    def get_analyzer_rng(self) -> np.random.Generator:
        return np.random.Generator(self.__get_root_bit_generator().jumped(4))


@dataclass
class OptimizationConfig(BaseConfig):
    number_of_rings: int = 8
    number_of_episodes_in_L: int = 100
    strategy: list[float] = field(default_factory=list[float])

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.strategy) == 0:
            self.strategy = [0.03, 0.07, 0.2, 0.7]


@dataclass
class SimulationConfig(BaseConfig):
    K: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    L: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    lambda_limit: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    name: str = ""
    label: str = ""
    format: bool = False
    simulate_one_cycle = False

    def __post_init__(self) -> None:
        super().__post_init__()
        self.K = np.array(self.K)
        self.L = np.array(self.L)
        self.lambda_limit = np.array(self.lambda_limit)

    @property
    def number_of_rings(self) -> int:
        return self.L.size
