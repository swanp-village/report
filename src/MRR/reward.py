import numpy as np
from random import uniform
from time import time


def generate_action(number_of_rings):
    action = [np.zeros(number_of_rings + 1).tolist()]
    I = np.eye(number_of_rings + 1)
    v = np.matrix([-0.3, -0.1, -0.05, -0.01, 0.3, 0.1, 0.05, 0.01])
    for i in range(number_of_rings + 1):
        a_i = v.T * I[i]
        action.extend(a_i.tolist())
    action.extend([
        [
            uniform(-1, 1)
            for _ in range(number_of_rings + 1)
        ]
        for _ in range(5)
    ])

    return action


class Reward:
    def __init__(
        self,
        x,
        y,
        center_wavelength,
        number_of_rings,
        loss,
        bottom
    ):
        self.x = x
        self.y = y
        self.distance = x[1] - x[0]
        self.center_wavelength = center_wavelength
        self.number_of_rings = number_of_rings
        self.loss = loss
        self.bottom = bottom

    def calculate_pass_band_range(self):
        pass_band_range = []
        start = 0
        end = self.x.size - 1
        a = np.where(self.y <= self.loss, True, False)
        b = np.append(a[1:], a[-1])
        pass_band_range = np.where(np.logical_xor(a, b))[0]
        if pass_band_range.size == 0:
            return pass_band_range
        if not a[pass_band_range][0]:
            pass_band_range = np.append(start, pass_band_range)
        if a[pass_band_range][-1]:
            pass_band_range = np.append(pass_band_range, end)
        pass_band_range = np.reshape(pass_band_range, [pass_band_range.size // 2, 2])

        return pass_band_range

    def get_pass_band(self):
        pass_band = []
        cross_talk = []
        for start, end in self.calculate_pass_band_range():
            if self.center_wavelength >= self.x[start] and self.center_wavelength <= self.x[end]:
                pass_band.extend([start, end])
            else:
                cross_talk.extend([start, end])

        pass_band = np.reshape(pass_band, [len(pass_band) // 2, 2])
        cross_talk = np.reshape(cross_talk, [len(cross_talk) // 2, 2])

        return pass_band, cross_talk

    def evaluate_band(self):
        pass_band, cross_talk = self.get_pass_band()
        # if pass_band.shape[0] == 1 and cross_talk.shape[0] < 1:
        if pass_band.shape[0] == 1:
            start = pass_band[0][0]
            end = pass_band[0][1]
            number_of_cross_talk = cross_talk.shape[0]
            return (
                self.evaluate_pass_band(start, end) +
                self.evaluate_stop_band(start, end) +
                self.evaluate_cross_talk(number_of_cross_talk)
            )
        else:
            return 0

    def evaluate_pass_band(self, start, end):
        a = abs(
            self.loss * (
                self.x[end] - self.x[start]
            )
        )
        b = abs(
            sum(
                self.loss - self.y[start:end]
            ) * self.distance
        )

        return b / a

    def evaluate_stop_band(self, start, end):
        c = abs(
            (self.bottom - self.loss) * (
                (self.x[start] - self.x[0]) + (self.x[-1] - self.x[end])
            )
        )

        y1 = np.where(
            self.y[0:start] > self.bottom,
            self.loss - self.y[0:start],
            self.loss - self.bottom
        )
        y1 = np.where(
            y1 > 0,
            y1,
            0
        )
        y2 = np.where(
            self.y[end:-1] > self.bottom,
            self.loss - self.y[end:-1],
            self.loss - self.bottom
        )
        y2 = np.where(
            y2 > 0,
            y2,
            0
        )
        d = abs(
            (
                sum(y1) + sum(y2)
            ) * self.distance
        )

        return d / c

    def evaluate_cross_talk(self, number_of_cross_talk):
        return number_of_cross_talk * -0.1
