from typing import Optional

import numpy as np
import numpy.typing as npt
from config.model import BaseConfig
from scipy.signal import argrelmax, argrelmin


class Evaluator:
    """Evaluator of the transfer function of the MRR filter.
    Args:
        x (List[float]): List of x.
        y (List[float]): List of y.
        config (Dict[str, Any]): Configuration of the MRR.
            Keys:
                center_wavelength (float): The center wavelength.
                H_p (float): The threshold of the max loss in pass band. H_p.
                H_s (float): The threshold of the min loss in stop band. H_s.
                length_of_3db_band (float): The required length of the 3dB band.
    Attributes:
        x (List[float]): List of x.
        y (List[float]): List of y.
        distance (float): Distance of x.
        center_wavelength (float): The center wavelength.
        H_p (float): The threshold of the max loss in pass band. H_p.
        H_s (float): The threshold of the min loss in stop band. H_s.
        length_of_3db_band (float): The required length of the 3dB band.
    """

    def __init__(
        self,
        x: npt.NDArray[np.float_],
        y: npt.NDArray[np.float_],
        config: BaseConfig,
        weight: Optional[list[float]] = None,
    ) -> None:
        self.x = x
        self.y = y
        self.distance: np.float_ = x[1] - x[0]
        self.center_wavelength = config.center_wavelength
        self.max_crosstalk = config.max_crosstalk
        self.H_p = config.H_p
        self.H_s = config.H_s
        self.H_i = config.H_i
        self.r_max = config.r_max
        self.length_of_3db_band = config.length_of_3db_band
        if weight is None:
            self.weight = np.array(config.weight, dtype=np.float_)
        else:
            self.weight = np.array(weight, dtype=np.float_)

    def calculate_pass_band_range(self) -> np.float_:
        pass_band_range = []
        start = 0
        end = self.x.size - 1
        a = np.where(self.y <= self.H_p, True, False)
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

    def get_pass_band(self) -> (npt.ArrayLike, npt.ArrayLike):
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

    def get_3db_band(self, start: int, end: int) -> npt.ArrayLike:
        border: np.float_ = self.y.max() - 3
        a = np.where(self.y[start:end] <= border, True, False)
        b = np.append(a[1:], a[-1])
        index = np.where(np.logical_xor(a, b))[0]

        return index

    def evaluate_band(self) -> np.float_:
        pass_band, _ = self.get_pass_band()
        if pass_band.shape[0] != 1:
            return np.float_(0)

        start = pass_band[0][0]
        end = pass_band[0][1]
        result = [
            self.evaluate_pass_band(start, end),
            self.evaluate_stop_band(start, end),
            self.evaluate_insertion_loss(),
            self.evaluate_3db_band(start, end),
            self.evaluate_ripple(start, end),
            self.evaluate_cross_talk(start, end),
            self.evaluate_shape_factor(start, end),
        ]
        n_eval = len(result)
        W_c = self.weight[:n_eval]
        W_b = self.weight[n_eval:]
        E_c = np.float_(0)
        E_b = np.float_(1)
        for i in range(n_eval):
            E_c += result[i][0] * W_c[i]
            if not result[i][1]:
                E_b *= W_b[i]
        E = E_c * E_b

        return E

    def evaluate_pass_band(self, start: int, end: int) -> tuple[np.float_, bool]:
        a = abs(self.H_p * (self.x[end] - self.x[start]))

        if a == 0:
            return (0, False)

        b = abs(np.sum(self.H_p - self.y[start:end]) * self.distance)
        E = b / a

        return (E, True)

    def evaluate_stop_band(self, start: int, end: int) -> tuple[np.float_, bool]:
        c = abs((self.H_s - self.H_p) * ((self.x[start] - self.x[0]) + (self.x[-1] - self.x[end])))

        if c == 0:
            return (0, False)

        y1 = np.where(self.y[0:start] > self.H_s, self.H_p - self.y[0:start], self.H_p - self.H_s)
        y1 = np.where(y1 > 0, y1, 0)
        y2 = np.where(self.y[end:-1] > self.H_s, self.H_p - self.y[end:-1], self.H_p - self.H_s)
        y2 = np.where(y2 > 0, y2, 0)
        d = abs((np.sum(y1) + np.sum(y2)) * self.distance)
        E = d / c

        return (E, True)

    def evaluate_cross_talk(self, pass_band_start: int, pass_band_end: int) -> tuple[np.float_, bool]:
        start = self.y[:pass_band_start]
        end = self.y[pass_band_end:]
        maxid_start = np.append(0, argrelmax(start))
        maxid_end = np.append(argrelmax(end), -1)
        start_peak = start[maxid_start]
        end_peak = end[maxid_end]
        a = np.any(start_peak > self.max_crosstalk)
        b = np.any(end_peak > self.max_crosstalk)
        if a or b:
            return (0, False)
        return (0, True)

    def evaluate_insertion_loss(self) -> tuple[np.float_, bool]:
        insertion_loss = self.y[self.x == self.center_wavelength]
        if insertion_loss[0] < self.H_i:
            return (0, False)
        E = 1 - insertion_loss[0] / self.H_i
        return (E, True)

    def evaluate_ripple(self, start: int, end: int) -> tuple[np.float_, bool]:
        pass_band = self.y[start:end]
        index = self.get_3db_band(start, end)
        if index.size <= 1:
            return (0, False)
        three_db_band = pass_band[index[0] : index[-1]]
        maxid = argrelmax(three_db_band, order=1)
        minid = argrelmin(three_db_band, order=1)
        peak_max = three_db_band[maxid]
        peak_min = three_db_band[minid]
        if len(peak_min) == 0:
            return (1, True)
        dif = peak_max.max() - peak_min.min()
        if dif > self.r_max:
            return (0, False)
        E = 1 - dif / self.r_max
        return (E, True)

    def evaluate_3db_band(self, start: int, end: int) -> tuple[np.float_, bool]:
        index = self.get_3db_band(start, end)
        if index.size <= 1:
            return (0, False)
        distance = self.distance * (index[-1] - index[0])
        if distance > self.length_of_3db_band:
            E = (2 * self.length_of_3db_band - distance) / self.length_of_3db_band
        else:
            E = distance / self.length_of_3db_band
        E = E ** 3
        return (E, True)

    def evaluate_shape_factor(self, start: int, end: int) -> tuple[np.float_, bool]:
        index = self.get_3db_band(start, end)
        if index.size <= 1:
            return (0, False)
        E = (index[-1] - index[0]) / (end - start)
        if E < 0.5:
            return (E, False)
        return (E, True)
