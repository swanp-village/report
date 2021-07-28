from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class BaseConfig:
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


@dataclass
class OptimizationConfig(BaseConfig):
    number_of_rings: int = 8
    number_of_episodes_in_L: int = 100
    strategy: list[float] = [0.03, 0.07, 0.2, 0.7]


@dataclass
class SimulationConfig(BaseConfig):
    K: npt.NDArray[np.float64] = np.array([])
    L: npt.NDArray[np.float64] = np.array([])
    lambda_limit: npt.NDArray[np.float64] = np.array([])
    name: str = ""

    @property
    def number_of_rings(self) -> int:
        return self.L.size
