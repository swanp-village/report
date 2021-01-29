from config.base import config as base

config = {
    'eta': 0.996,
    'alpha': 52.96,
    'n_eff': 2.2,
    'n_g': 4.4,

    'number_of_rings': 1,
    'center_wavelength': 1550e-9,
    'FSR': 10e-9,
    'length_of_3db_band': 1e-9,

    'max_crosstalk': base['max_crosstalk'],
    'number_of_evaluate_function': 6,
    'H_p': base['H_p'],
    'H_s': base['H_s'],
    'H_i': base['H_i'],
    'r_max': base['r_max'],
}
