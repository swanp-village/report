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

    @property
    def seedsequence(self) -> np.random.SeedSequence:
        return np.random.SeedSequence(self.entropy)

    def get_root_bit_generator(self) -> np.random.PCG64DXSM:
        return np.random.PCG64DXSM(self.seedsequence)

    def get_differential_evolution_rng(self) -> np.random.Generator:
        np.random.Generator(self.get_root_bit_generator().jumped(1))

    def get_multi_start_local_search_rng(self) -> np.random.Generator:
        np.random.Generator(self.get_root_bit_generator().jumped(2))

    def get_ring_rng(self) -> np.random.Generator:
        np.random.Generator(self.get_root_bit_generator().jumped(3))

@dataclass
class OptimizationConfig(BaseConfig):
    number_of_rings: int = 8
    number_of_episodes_in_L: int = 100
    strategy: list[float] = field(default_factory=[0.03, 0.07, 0.2, 0.7])


@dataclass
class SimulationConfig(BaseConfig):
    K: npt.NDArray[np.float64] = field(default_factory=np.ndarray)
    L: npt.NDArray[np.float64] = field(default_factory=np.ndarray)
    lambda_limit: npt.NDArray[np.float64] = field(default_factory=np.ndarray)
    name: str = ""
    label: str = ""

    @property
    def number_of_rings(self) -> int:
        return self.L.size

    def lambda_limit_is_defined(self) -> bool:
        return self.lambda_limit.size > 0
