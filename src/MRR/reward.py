from typing import List
import numpy as np


def calculate_pass_band_range(x, y, max_loss):
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


def evaluate_pass_band(x, y, start, end, distance, loss):
    a = abs(
        loss * (
            x[end] - x[start]
        )
    )
    b = abs(
        sum(
            loss - y[start:end]
        ) * distance
    )

    return b / a


def evaluate_stop_band(x, y, start, end, distance, loss, bottom):
    c = abs(
        (bottom - loss) * (
            (x[start] - x[0]) + (x[-1] - x[end])
        )
    )

    y1 = np.where(
        y[0:start] > bottom,
        loss - y[0:start],
        loss - bottom
    )
    y1 = np.where(
        y1 > 0,
        y1,
        0
    )
    y2 = np.where(
        y[end:-1] > bottom,
        loss - y[end:-1],
        loss - bottom
    )
    y2 = np.where(
        y2 > 0,
        y2,
        0
    )
    d = abs(
        (
            sum(y1) + sum(y2)
        ) * distance
    )

    return d / c


def evaluate_band(x, y, center_wavelength, loss):
    pass_band_range = calculate_pass_band_range(x, y, loss)
    pass_band, cross_talk = get_pass_band(x, y, pass_band_range, center_wavelength)
    # if pass_band.shape[0] == 1 and cross_talk.shape[0] < 1:
    if pass_band.shape[0] == 1:
        start = pass_band[0][0]
        end = pass_band[0][1]
        distance = x[start] - x[start + 1]
        bottom = -15
        print('ok')

        return evaluate_pass_band(x, y, start, end, distance, loss) + evaluate_stop_band(x, y, start, end, distance, loss, bottom)
    else:
        print('failed')

        return 0


def evaluate_ring_combination(L, FSR_list, practical_FSR):
    print(L)
    print(FSR_list)
    print(practical_FSR)
    return True


def init_action(number_of_rings):
    I = np.eye(number_of_rings + 1)
    base = np.matrix([-0.1, -0.05, 0.05, 0.1])
    action = [np.zeros(number_of_rings + 1).tolist()]
    for i in range(number_of_rings + 1):
        a_i = base.T * I[i]
        action.extend(a_i.tolist())

    return action
