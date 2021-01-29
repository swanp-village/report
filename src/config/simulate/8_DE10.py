import numpy as np

config = {
    'eta': 0.996,  # 結合損
    'alpha': 52.96,  # 伝搬損失係数
    'K': np.array([
        0.64346182,
        0.120943,
        0.07188976,
        0.12561968,
        0.14042602,
        0.23304473,
        0.05719831,
        0.07988251,
        0.7461665
    ]),  # 結合率
    'L': np.array([
        2.83163416e-05,
        2.83163416e-05,
        2.83163416e-05,
        1.13265366e-04,
        2.83163416e-05,
        5.66326831e-05,
        2.83163416e-05,
        5.66326831e-05
    ]),  # リング周長
    'n_eff': 3.3938,  # 実行屈折率
    'n_g': 4.2,  # 群屈折率
    'center_wavelength': 1550e-9,
    # 'lambda': np.arange(1545e-9, 1555e-9, 1e-12)
}