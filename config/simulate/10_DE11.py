import numpy as np

config = {
    "eta": 0.996,  # 結合損
    "alpha": 52.96,  # 伝搬損失係数
    "K": [
        0.8436244108235607,
        0.3206806557245825,
        0.14508987107202917,
        0.1087230374402195,
        0.11304571938048884,
        0.10599745145286502,
        0.08266817489121495,
        0.053047500926419695,
        0.06738491616667856,
        0.11428483337342964,
        0.5566334743976372,
    ],  # 結合率
    "L": [
        0.0001099090909090909,
        0.0001099090909090909,
        0.0001099090909090909,
        0.0001099090909090909,
        0.0001099090909090909,
        0.0001099090909090909,
        8.243181818181816e-05,
        8.243181818181816e-05,
        8.243181818181816e-05,
        8.243181818181816e-05,
    ],  # リング周長
    "n_eff": 2.2,  # 実行屈折率
    "n_g": 4.4,  # 群屈折率
    "center_wavelength": 1550e-9,
    "lambda_limit": np.arange(1520e-9, 1560e-9, 1e-12),
    "label": r"10th, E=14.20",
}