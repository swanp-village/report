import numpy as np


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
        pass_band_range = np.array([], dtype=np.int)
        start = 0
        end = self.x.size - 1
        for i in np.arange(start, end):
            if self.y[i] <= self.loss and self.y[i + 1] > self.loss:  # increase
                pass_band_range = np.append(pass_band_range, i)
            elif self.y[i] >= self.loss and self.y[i + 1] < self.loss:  # decrease
                if pass_band_range.size > 0:
                    pass_band_range = np.append(pass_band_range, i)
                else:
                    pass_band_range = np.append(pass_band_range, [start, i])

        if pass_band_range.size % 2 == 1:
            pass_band_range = np.append(pass_band_range, end)

        pass_band_range = np.reshape(pass_band_range, [pass_band_range.size // 2, 2])

        return pass_band_range

    def get_pass_band(self):
        pass_band = np.array([], dtype=np.int)
        cross_talk = np.array([], dtype=np.int)
        for start, end in self.calculate_pass_band_range():
            if self.center_wavelength >= self.x[start] and self.center_wavelength <= self.x[end]:
                pass_band = np.append(pass_band, [start, end])
            else:
                cross_talk = np.append(cross_talk, [start, end])

        pass_band = np.reshape(pass_band, [pass_band.size // 2, 2])
        cross_talk = np.reshape(cross_talk, [cross_talk.size // 2, 2])

        return pass_band, cross_talk

    def evaluate_band(self):
        pass_band, cross_talk = self.get_pass_band()
        # if pass_band.shape[0] == 1 and cross_talk.shape[0] < 1:
        if pass_band.shape[0] == 1:
            start = pass_band[0][0]
            end = pass_band[0][1]
            print('ok')

            return self.evaluate_pass_band(start, end) + self.evaluate_stop_band(start, end)
        else:
            print('failed')

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
