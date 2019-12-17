from typing import List
import numpy as np


def calculate_pass_band_range(x: List[float], y: List[float], max_loss: float) -> List[List[float]]:
    pass_band_range = np.array([], dtype=np.int)
    start = 0
    end = x.size - 1
    for i in np.arange(start, end):
        if y[i] <= max_loss and y[i + 1] > max_loss:  # increase
            pass_band_range = np.append(pass_band_range, i)
        elif y[i] >= max_loss and y[i + 1] < max_loss:  # decrease
            if pass_band_range.size > 0:
                pass_band_range = np.append(pass_band_range, i)
            else:
                pass_band_range = np.append(pass_band_range, [start, i])

    if pass_band_range.size % 2 == 1:
        pass_band_range = np.append(pass_band_range, end)

    pass_band_range = np.reshape(pass_band_range, [pass_band_range.size // 2, 2])

    return pass_band_range


def get_pass_band(x, y, pass_band_range, center_wavelength):
    pass_band = np.array([], dtype=np.int)
    cross_talk = np.array([], dtype=np.int)
    for start, end in pass_band_range:
        if center_wavelength >= x[start] and center_wavelength <= x[end]:
            pass_band = np.append(pass_band, [start, end])
        else:
            cross_talk = np.append(cross_talk, [start, end])

    pass_band = np.reshape(pass_band, [pass_band.size // 2, 2])
    cross_talk = np.reshape(cross_talk, [cross_talk.size // 2, 2])

    return pass_band, cross_talk


def evaluate_pass_band(x, y, center_wavelength, max_loss):
    pass_band_range = calculate_pass_band_range(x, y, max_loss)
    pass_band, cross_talk = get_pass_band(x, y, pass_band_range, center_wavelength)
    # a = - max_loss * (x[pass_band[0][1]] - x[pass_band[0][0]])

    if pass_band.shape[0] == 1 and cross_talk.shape[0] < 1:
        print('ok')
        return True
    else:
        print('failed')
        return False
