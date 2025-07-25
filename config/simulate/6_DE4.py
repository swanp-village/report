import numpy as np

config = {
    "eta": 0.996,
    "alpha": 52.96,
    "K": [
        0.2836975434843274,
        0.06501198607448408,
        0.03866038607157398,
        0.058459822281189386,
        0.06744411386938662,
        0.07891303068717524,
        0.4568623528091366
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
    "label":"CMAES_no_error",
    "lambda_limit": np.arange(1540e-9, 1560e-9, 1e-12)

}
