from dataclasses import dataclass

import numpy as np


@dataclass
class BaseConfig:
    eta: float = 0.996
    alpha: float = 52.96
    n_eff: float = 2.2
    n_g: float = 4.4

    max_crosstalk: float = -30
    H_p: float = -20
    H_s: float = -60
    H_i: float = -10
    r_max: float = 5


@dataclass
class OptimizeConfig(BaseConfig):
    number_of_rings: int = 8
    center_wavelength: float = 1550e-9
    FSR: float = 20e-9
    length_of_3db_band: float = 1e-9
    min_ring_length: float = 50e-6
    number_of_episodes_in_L: int = 100


@dataclass
class SimulationConfig(BaseConfig):
    K: np.ndarray = np.array([])
    L: np.ndarray = np.array([])
    lambda_limit: np.ndarray = np.array([])
