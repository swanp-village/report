import argparse
import csv
from dataclasses import dataclass
from importlib import import_module
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from config.model import SimulationConfig
from config.random import get_analyzer_rng
from MRR.simulator import Accumulator, simulate_MRR

# 10_DE4 8_DE29 6_DE7 4_DE18
entropy = 5
n = 10000
sigma_L = 0.001 / 3
sigma_K = 0.1 / 3


def analyze(
    L: npt.NDArray[np.float_],
    K: npt.NDArray[np.float_],
    n_eff: float,
    n_g: float,
    eta: float,
    alpha: float,
    center_wavelength: float,
    length_of_3db_band: float,
    FSR: float,
    max_crosstalk: float,
    H_i: float,
    H_p: float,
    H_s: float,
    r_max: float,
    weight: list[float],
    min_ring_length: float,
    format: bool = False,
    lambda_limit: npt.NDArray[np.float_] = np.array([]),
    name: str = "",
    label: str = "",
    seedsequence: np.random.SeedSequence = np.random.SeedSequence(),
) -> None:
    accumulator = Accumulator(init_graph=False)
    base_result = simulate_MRR(
        accumulator=accumulator,
        L=L,
        K=K,
        n_g=n_g,
        n_eff=n_eff,
        eta=eta,
        alpha=alpha,
        center_wavelength=center_wavelength,
        length_of_3db_band=length_of_3db_band,
        max_crosstalk=max_crosstalk,
        H_p=H_p,
        H_s=H_s,
        H_i=H_i,
        r_max=r_max,
        weight=weight,
        format=format,
        simulate_one_cycle=True,
        lambda_limit=lambda_limit,
        name=name,
        label=label,
        skip_graph=True,
        skip_evaluation=False,
        ignore_binary_evaluation=True,
    )
    base_lambda_limit = base_result.x
    # mrr.logger.save_simulation_config(config)

    with Pool() as pool:
        result_with_L_and_K = pool.map(
            simulate_with_error,
            (
                SimulateWithErrorParams(
                    accumulator=accumulator,
                    L=L,
                    K=K,
                    n_g=n_g,
                    n_eff=n_eff,
                    eta=eta,
                    alpha=alpha,
                    center_wavelength=center_wavelength,
                    length_of_3db_band=length_of_3db_band,
                    FSR=FSR,
                    max_crosstalk=max_crosstalk,
                    H_p=H_p,
                    H_s=H_s,
                    H_i=H_i,
                    r_max=r_max,
                    weight=weight,
                    min_ring_length=min_ring_length,
                    lambda_limit=base_lambda_limit,
                    rng=get_analyzer_rng(seedsequence, i),
                    format=format,
                    simulate_one_cycle=False,
                    name=name,
                    label=label,
                    skip_graph=True,
                    skip_evaluation=False,
                    ignore_binary_evaluation=True,
                )
                for i in range(n)
            ),
        )

    with open(accumulator.logger.target / "analyzer_sigma.txt", "w") as fp:
        tsv_writer = csv.writer(fp, delimiter="\t")
        tsv_writer.writerows(zip([sigma_K, sigma_L]))

    with open(accumulator.logger.target / "analyzer_result_with_L_and_K.txt", "w") as fp:
        tsv_writer = csv.writer(fp, delimiter="\t")
        tsv_writer.writerows(zip(result_with_L_and_K))

    result = [x[0] for x in result_with_L_and_K]

    with open(accumulator.logger.target / "analyzer_result.txt", "w") as fp:
        tsv_writer = csv.writer(fp, delimiter="\t")
        tsv_writer.writerows(zip(result))

    normalized_result = np.array(result)
    normalized_result[normalized_result < 0] = np.float_(0.0)
    plt.hist(normalized_result, range=(0, 15), bins=15 * 4)
    plt.ylabel("frequency", size=20)
    plt.xlabel("evaluation function value", size=20)
    plt.savefig(accumulator.logger.target / "analyzer_result.png")
    plt.show()


@dataclass
class SimulateWithErrorParams:
    accumulator: Accumulator
    L: npt.NDArray[np.float_]
    K: npt.NDArray[np.float_]
    n_g: float
    n_eff: float
    eta: float
    alpha: float
    center_wavelength: float
    length_of_3db_band: float
    FSR: float
    max_crosstalk: float
    H_p: float
    H_s: float
    H_i: float
    r_max: float
    weight: list[float]
    min_ring_length: float
    lambda_limit: npt.NDArray[np.float_]
    rng: np.random.Generator
    format: bool = False
    simulate_one_cycle: bool = True
    name: str = ""
    label: str = ""
    skip_graph: bool = True
    skip_evaluation: bool = False
    ignore_binary_evaluation: bool = True


def simulate_with_error(params: SimulateWithErrorParams) -> tuple[np.float_, list[np.float_], list[np.float_]]:
    L_error_rate = params.rng.normal(1, sigma_L)
    params.L = params.L * L_error_rate
    K_with_error: npt.NDArray[np.float_] = params.K * np.float_(-1)
    while (np.logical_or(0 >= K_with_error, K_with_error >= params.eta)).all():
        K_with_error = params.K * params.rng.normal(1, sigma_K)
    params.K = K_with_error
    result = simulate_MRR(
        accumulator=params.accumulator,
        L=params.L,
        K=params.K,
        n_g=params.n_g,
        n_eff=params.n_eff,
        eta=params.eta,
        alpha=params.alpha,
        center_wavelength=params.center_wavelength,
        length_of_3db_band=params.length_of_3db_band,
        max_crosstalk=params.max_crosstalk,
        H_p=params.H_p,
        H_s=params.H_s,
        H_i=params.H_i,
        r_max=params.r_max,
        weight=params.weight,
        format=params.format,
        simulate_one_cycle=True,
        lambda_limit=params.lambda_limit,
        name=params.name,
        label=params.label,
        skip_graph=True,
        skip_evaluation=False,
        ignore_binary_evaluation=True,
    )
    return (result.evaluation_result, params.K.tolist(), params.L.tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()
    config_name: str = args.config
    if config_name[-3:] == ".py":
        config_name = config_name[:-3]
    try:
        imported_module = import_module(f"config.simulate.{config_name}")
        imported_config = getattr(imported_module, "config")
        simulation_config = SimulationConfig(**imported_config, entropy=entropy)
        analyze(
            L=simulation_config.L,
            K=simulation_config.K,
            n_g=simulation_config.n_g,
            n_eff=simulation_config.n_eff,
            eta=simulation_config.eta,
            alpha=simulation_config.alpha,
            center_wavelength=simulation_config.center_wavelength,
            length_of_3db_band=simulation_config.length_of_3db_band,
            FSR=simulation_config.FSR,
            max_crosstalk=simulation_config.max_crosstalk,
            H_p=simulation_config.H_p,
            H_s=simulation_config.H_s,
            H_i=simulation_config.H_i,
            r_max=simulation_config.r_max,
            weight=simulation_config.weight,
            min_ring_length=simulation_config.min_ring_length,
            format=simulation_config.format,
            lambda_limit=simulation_config.lambda_limit,
            name=simulation_config.name,
            label=simulation_config.label,
            seedsequence=simulation_config.seedsequence,
        )
    except ModuleNotFoundError as e:
        print(e)
