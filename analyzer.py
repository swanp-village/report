# coding: utf-8

import csv

import matplotlib.pyplot as plt
import numpy as np

from config.model import SimulationConfig
from MRR.simulator import Simulator

base_config = {
    "eta": 0.996,  # 結合損
    "alpha": 52.96,  # 伝搬損失係数
    "K": [
        0.6843076241067676,
        0.13216441410521854,
        0.06880118310062745,
        0.04020737891166124,
        0.0462118513719252,
        0.12904667330266356,
        0.4606181759014707,
        0.27652315122370236,
        0.03910018568630988,
        0.07683629850124549,
        0.5880439816223341,
    ],  # 結合率
    "L": [
        8.243181818181816e-05,
        8.243181818181816e-05,
        8.243181818181816e-05,
        5.495454545454545e-05,
        8.243181818181816e-05,
        5.495454545454545e-05,
        5.495454545454545e-05,
        5.495454545454545e-05,
        5.495454545454545e-05,
        8.243181818181816e-05,
    ],  # リング周長
    "n_eff": 2.2,  # 実行屈折率
    "n_g": 4.4,  # 群屈折率
    "center_wavelength": 1550e-9,
    "lambda_limit": np.arange(1520e-9, 1560e-9, 1e-12),
    "label": "{}{\\(2\\textrm{--}10^\\textrm{th}(p=[0.03,0.07,0.2,0.7])\\)}",
}


mrr = Simulator()
rng = np.random.default_rng()


def add_design_error(rng, k: float, eta: float) -> float:
    result = -1
    while True:
        if 0 < result < eta:
            return result
        # if 0 < k < 0.1:
        #     result = rng.normal(k, 0.01)
        # elif 0.1 <= k <= 0.9:
        #     result = rng.normal(k, 0.2)
        # else:
        #     result = rng.normal(k, 0.01)
        result = rng.normal(k, 0.1)


result = []
for _ in range(1000):
    config = SimulationConfig(**base_config)
    config.simulate_one_cycle = True
    # dif_l = rng.normal(config.L, config.L * 0.1, config.L.shape)
    # config.L = dif_l
    dif_k = np.array([add_design_error(rng, k, config.eta) for k in config.K])
    config.K = dif_k
    result.append(mrr.simulate(config, True).evaluation_result)

with open("result/analyzer_result_k.txt", "w") as fp:
    tsv_writer = csv.writer(fp, delimiter="\t")
    tsv_writer.writerows(zip(result))

plt.hist(result)
plt.savefig("result/analyzer_result_k.jpg")
plt.show()
