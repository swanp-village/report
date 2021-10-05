import argparse
import csv
from importlib import import_module

import matplotlib.pyplot as plt
import numpy as np

from config.model import SimulationConfig
from MRR.simulator import Simulator


def analyze(config: SimulationConfig) -> None:
    mrr = Simulator()
    mrr.logger.save_simulation_config(config)
    rng = np.random.default_rng()
    result = [simulate_with_error(rng, mrr, config) for _ in range(10000)]

    with open(mrr.logger.target / "analyzer_result.txt", "w") as fp:
        tsv_writer = csv.writer(fp, delimiter="\t")
        tsv_writer.writerows(zip(result))

    normalized_result = np.array(result)
    normalized_result[normalized_result < 0] = np.float_(0.0)
    plt.hist(normalized_result, range=(0, 15), bins=15 * 4)
    plt.savefig(mrr.logger.target / "analyzer_result.jpg")
    plt.show()


def simulate_with_error(rng: np.random.Generator, mrr: Simulator, config: SimulationConfig) -> np.float_:
    dif_l = rng.normal(config.L, config.L * np.float_(0.01 / 3), config.L.shape)
    config.L = dif_l
    dif_k = np.array([add_design_error(rng, k, config.eta) for k in config.K])
    config.K = dif_k
    return mrr.simulate(config, True).evaluation_result


def add_design_error(rng: np.random.Generator, k: float, eta: float) -> float:
    result = -1.0
    while True:
        if 0 < result < eta:
            return result
        result = rng.normal(k, 0.1 / 3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()
    config_name: str = args.config
    try:
        imported_module = import_module(f"config.simulate.{config_name}")
        imported_config = getattr(imported_module, "config")
        simulation_config = SimulationConfig(**imported_config)
        simulation_config.simulate_one_cycle = True
        analyze(simulation_config)
    except ModuleNotFoundError as e:
        print(e)
