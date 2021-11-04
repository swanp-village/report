import argparse
import csv
from dataclasses import asdict, dataclass
from importlib import import_module
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from config.model import SimulationConfig
from config.random import get_analyzer_rng
from MRR.simulator import Simulator, simulate_MRR

# 10_DE4 8_DE29 6_DE7 4_DE18
entropy = 5
n = 3
sigma_L = 0.01 / 3
sigma_K = 0.1 / 3


def analyze(
    L: npt.NDArray[np.float64],
    K: npt.NDArray[np.float64],
    n_g: float,
    n_eff: float,
    eta: float,
    alpha: float,
    center_wavelength: float,
    length_of_3db_band: float,
    FSR: float,
    max_crosstalk: float,
    H_p: float,
    H_s: float,
    H_i: float,
    r_max: float,
    weight: list[float],
    min_ring_length: float,
    format: bool = False,
    lambda_limit: npt.NDArray[np.float64] = np.array([]),
    name: str = "",
    label: str = "",
    seedsequence: np.random.SeedSequence = np.random.SeedSequence(),
    **kwargs,
) -> None:
    mrr = Simulator(init_graph=False)
    base_result = simulate_MRR(
        simulator=mrr,
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
        format=format,
        simulate_one_cycle=True,
        lambda_limit=lambda_limit,
        name=name,
        label=label,
        skip_gragh=True,
        skip_evaluation=False,
        ignore_binary_evaluation=True,
        seedsequence=seedsequence,
    )
    base_lambda_limit = base_result.x
    # mrr.logger.save_simulation_config(config)

    with Pool() as pool:
        result_with_L_and_K = pool.map(
            simulate_with_error,
            (
                SimulateWithErrorParams(
                    simulator=mrr,
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
                    skip_gragh=True,
                    skip_evaluation=False,
                    ignore_binary_evaluation=True,
                )
                for i in range(n)
            ),
        )

    with open(mrr.logger.target / "analyzer_sigma.txt", "w") as fp:
        tsv_writer = csv.writer(fp, delimiter="\t")
        tsv_writer.writerows(zip([sigma_K, sigma_L]))

    with open(mrr.logger.target / "analyzer_result_with_L_and_K.txt", "w") as fp:
        tsv_writer = csv.writer(fp, delimiter="\t")
        tsv_writer.writerows(zip(result_with_L_and_K))

    result = [x[0] for x in result_with_L_and_K]

    with open(mrr.logger.target / "analyzer_result.txt", "w") as fp:
        tsv_writer = csv.writer(fp, delimiter="\t")
        tsv_writer.writerows(zip(result))

    normalized_result = np.array(result)
    normalized_result[normalized_result < 0] = np.float_(0.0)
    plt.hist(normalized_result, range=(0, 15), bins=15 * 4)
    plt.ylabel("frequency", size=20)
    plt.xlabel("evaluation function value", size=20)
    plt.savefig(mrr.logger.target / "analyzer_result.png")
    plt.show()


@dataclass
class SimulateWithErrorParams:
    simulator: Simulator
    L: npt.NDArray[np.float64]
    K: npt.NDArray[np.float64]
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
    lambda_limit: npt.NDArray[np.float64]
    rng: np.random.Generator
    format: bool = False
    simulate_one_cycle: bool = True
    name: str = ""
    label: str = ""
    skip_gragh: bool = True
    skip_evaluation: bool = False
    ignore_binary_evaluation: bool = True


def simulate_with_error(params: SimulateWithErrorParams) -> tuple[np.float_, list[np.float_], list[np.float_]]:
    dif_l = params.rng.normal(params.L, params.L * np.float_(sigma_L), params.L.shape)
    params.L = dif_l
    dif_k = np.array([add_design_error(params.rng, k, params.eta) for k in params.K])
    params.K = dif_k
    result = simulate_MRR(**asdict(params))
    return (result.evaluation_result, params.K.tolist(), params.L.tolist())


def add_design_error(rng: np.random.Generator, k: float, eta: float) -> float:
    result = -1.0
    while True:
        if 0 < result < eta:
            return result
        result = rng.normal(k, k * sigma_K)


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
        analyze(**asdict(simulation_config), seedsequence=simulation_config.seedsequence)
    except ModuleNotFoundError as e:
        print(e)
