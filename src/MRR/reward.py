from scipy.signal import argrelmin, argrelmax
from typing import List
import copy
import numpy as np


def calculate_pass_band_range(x: List[float], y: List[float], max_loss: float) -> List[List[float]]:
    # x = copy.deepcopy(_x)
    # y = copy.deepcopy(_y)
    pass_band_range = np.array([])
    for i in np.arange(0, x.size - 1):
        if y[i] <= max_loss and y[i + 1] > max_loss:
            pass_band_range = np.append(pass_band_range, x[i])
        elif y[i] >= max_loss and y[i + 1] < max_loss and pass_band_range.size > 0:
            pass_band_range = np.append(pass_band_range, x[i])

    if pass_band_range.size % 2 == 1:
        pass_band_range = pass_band_range[:-1]

    pass_band_range = np.reshape(pass_band_range, [pass_band_range.size // 2, 2])

    return pass_band_range


def a(pass_band_range: List[List[float]], center_wavelength: float):
    for start, end in pass_band_range:
        if center_wavelength >= start and center_wavelength <= end:
            print('true:', start, end)
        else:
            print('false:', start, end)

