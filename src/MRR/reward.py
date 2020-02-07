import numpy as np
from random import uniform


def evaluate_ring_combination(required_FSR, L, FSR_list, practical_FSR):
    print(L)
    print(FSR_list)
    print(practical_FSR)
    return True


def init_action(number_of_rings):
    I = np.eye(number_of_rings + 1)
    base = np.matrix([-0.3, -0.1, -0.05, -0.01, 0.3, 0.1, 0.05, 0.01])
    action = [np.zeros(number_of_rings + 1).tolist()]
    for i in range(number_of_rings + 1):
        a_i = base.T * I[i]
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
        for i in range(start, end):
            if self.y[i] <= self.loss and self.y[i + 1] > self.loss:  # increase
                pass_band_range.append(i)
            elif self.y[i] >= self.loss and self.y[i + 1] < self.loss:  # decrease
                if len(pass_band_range) > 0:
                    pass_band_range.append(i)
                else:
                    pass_band_range.extend([start, i])

        if len(pass_band_range) % 2 == 1:
            pass_band_range.append(end)

        pass_band_range = np.reshape(pass_band_range, [len(pass_band_range) // 2, 2])

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
