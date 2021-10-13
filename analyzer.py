import argparse
import csv
from importlib import import_module
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np

from config.model import SimulationConfig
from MRR.simulator import Simulator


def analyze(config: SimulationConfig) -> None:
    mrr = Simulator()
    config.simulate_one_cycle = True
    base_result = mrr.simulate(config, True)
    config.lambda_limit = base_result.x
    config.simulate_one_cycle = False
    mrr.logger.save_simulation_config(config)

    with Pool() as pool:
        result_with_L_and_K = pool.map(
            simulate_with_error, ((config.get_analyzer_rng(i), mrr, config) for i in range(10))
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


SimulateWithErrorParams = tuple[np.random.Generator, Simulator, SimulationConfig]
sigma_L = 0.01 / 3
sigma_K = 0.1 / 3


def simulate_with_error(params: SimulateWithErrorParams) -> tuple[np.float_, list[np.float_], list[np.float_]]:
    rng, mrr, config = params
    dif_l = rng.normal(config.L, config.L * np.float_(sigma_L), config.L.shape)
    config.L = dif_l
    dif_k = np.array([add_design_error(rng, k, config.eta) for k in config.K])
    config.K = dif_k
    return (mrr.simulate(config, True).evaluation_result, config.K.tolist(), config.L.tolist())


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
    try:
        imported_module = import_module(f"config.simulate.{config_name}")
        imported_config = getattr(imported_module, "config")
        simulation_config = SimulationConfig(**imported_config)
        simulation_config.entropy = 5
        analyze(simulation_config)
    except ModuleNotFoundError as e:
        print(e)
