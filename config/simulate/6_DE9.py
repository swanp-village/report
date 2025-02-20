import numpy as np

config = {
    "eta": 0.996,
    "alpha": 52.96,
    "K": [
        0.3465658521292391,
        0.038051217830144035,
        0.024435394368557628,
        0.040438324839299156,
        0.07354604508345125,
        0.08663868387253248,
        0.47304327205712965
    ],
    "L": [
        5.495454545454545e-05,
        5.495454545454545e-05,
        5.495454545454545e-05,
        8.243181818181816e-05,
        8.243181818181816e-05,
        8.243181818181816e-05
    ],
    "n_eff": 2.2,
    "n_g": 4.4,
    "center_wavelength": 1.55e-06,
    "label": "DE",
    "lambda_limit": np.arange(1520e-9, 1560e-9, 1e-12)
}
