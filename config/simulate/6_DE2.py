import numpy as np

config = {
    "eta": 0.996,
    "alpha": 52.96,
    "K": [
        0.2689955267042948,
        0.07101652652925749,
        0.04694993358990173,
        0.063038159683297674,
        0.07271902752423253,
        0.09406977408930683,
        0.4860063452560342
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
    "label":"DE_+_error",
    "lambda_limit": np.arange(1540e-9, 1560e-9, 1e-12)
}
