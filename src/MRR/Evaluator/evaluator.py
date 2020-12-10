import numpy as np


class Evaluator:
    """Evaluator of the transfer function of the MRR filter.
    Args:
        x (List[float]): List of x.
        y (List[float]): List of y.
        config (Dict[str, Any]): Configuration of the MRR.
            Keys:
                center_wavelength (float): The center wavelength.
                number_of_rings (int): Number of rings. The ring order.
                max_loss_in_pass_band (float): The threshold of the max loss in pass band. loss_p.
                required_loss_in_stop_band (float): The threshold of the min loss in stop band. loss_s.
                length_of_3db_band (float): The required length of the 3dB band.
    Attributes:
        x (List[float]): List of x.
        y (List[float]): List of y.
        distance (float): Distance of x.
        center_wavelength (float): The center wavelength.
        number_of_rings (int): Number of rings. The ring order.
        max_loss_in_pass_band (float): The threshold of the max loss in pass band. loss_p.
        required_loss_in_stop_band (float): The threshold of the min loss in stop band. loss_s.
        length_of_3db_band (float): The required length of the 3dB band.
    """
    def __init__(
        self,
        x,
        y,
        weight,
        config
    ):
        self.x: List[float] = x
        self.y: List[float] = y
        self.weight: List[float] = weight
        self.distance: float = x[1] - x[0]
        self.center_wavelength: float = config['center_wavelength']
        self.number_of_rings: int = config['number_of_rings']
        self.max_loss_in_pass_band: float = config['max_loss_in_pass_band']
        self.required_loss_in_stop_band: float = config['required_loss_in_stop_band']
        self.length_of_3db_band: float = config['length_of_3db_band']

    def calculate_pass_band_range(self):
        pass_band_range = []
        start = 0
        end = self.x.size - 1
        a = np.where(self.y <= self.max_loss_in_pass_band, True, False)
        b = np.append(a[1:], a[-1])
        pass_band_range = np.where(np.logical_xor(a, b))[0]
        if pass_band_range.size == 0:
            return pass_band_range
        if not a[pass_band_range][0]:
            pass_band_range = np.append(start, pass_band_range)
        if a[pass_band_range][-1]:
            pass_band_range = np.append(pass_band_range, end)
        pass_band_range = np.reshape(
            pass_band_range,
            [pass_band_range.size // 2, 2]
        )

        return pass_band_range

    def get_pass_band(self):
        pass_band = []
        cross_talk = []
        for start, end in self.calculate_pass_band_range():
            if (
                self.center_wavelength >= self.x[start] and
                self.center_wavelength <= self.x[end]
            ):
                pass_band.extend([start, end])
            else:
                cross_talk.extend([start, end])

        pass_band = np.reshape(pass_band, [len(pass_band) // 2, 2])
        cross_talk = np.reshape(cross_talk, [len(cross_talk) // 2, 2])

        return pass_band, cross_talk

    def get_3db_band(self, start, end):
        border = np.max(self.y) - 3
        a = np.where(self.y[start:end] <= border, True, False)
        b = np.append(a[1:], a[-1])
        index = np.where(np.logical_xor(a, b))[0]

        return index

    def evaluate_band(self):
        pass_band, cross_talk = self.get_pass_band()
        if pass_band.shape[0] == 1:
            start = pass_band[0][0]
            end = pass_band[0][1]
            number_of_cross_talk = cross_talk.shape[0]
            result = [
                self.evaluate_pass_band(start, end),
                self.evaluate_stop_band(start, end),
                self.evaluate_insertion_loss(),
                self.evaluate_3db_band(start, end),
                self.evaluate_ripple(start, end),
                self.evaluate_cross_talk(number_of_cross_talk)
            ]
            # print(result)
            return (
                (
                    result[0][0] * self.weight[0] +
                    result[1][0] * self.weight[1] +
                    result[2][0] * self.weight[2] +
                    result[3][0] * self.weight[3] +
                    result[4][0] * self.weight[4] +
                    result[5][0] * self.weight[5]
                ) *
                1 if result[0][1] else self.weight[6] *
                1 if result[1][1] else self.weight[7] *
                1 if result[2][1] else self.weight[8] *
                1 if result[3][1] else self.weight[9] *
                1 if result[4][1] else self.weight[10] *
                1 if result[5][1] else self.weight[11]
            )
        else:
            return 0

    def evaluate_insertion_loss(self):
        insertion_loss = self.y[self.x == self.center_wavelength]
        if insertion_loss.size == 0:
            return (0, True)
        if insertion_loss[0] < -20:
            return (0, False)
        return (1 + insertion_loss[0] / 20, True)

    def evaluate_ripple(self, start, end):
        pass_band = self.y[start:end]
        index = self.get_3db_band(start, end)
        if index.size <= 1:
            return (0, False)
        border = np.max(self.y) - 3

        y = pass_band[index[0]:index[-1]] + border
        var = np.var(y.T)
        if var == 0:
            return (1, True)
        result = 1 / (var + 1)

        return (result, True)

    def evaluate_3db_band(self, start, end):
        index = self.get_3db_band(start, end)
        if index.size <= 1:
            return (0, False)
        distance = self.distance * (index[-1] - index[0])
        if distance > self.length_of_3db_band:
            return ((2 * self.length_of_3db_band - distance) / self.length_of_3db_band, True)
        return (distance / self.length_of_3db_band, True)

    def evaluate_pass_band(self, start, end):
        a = abs(
            self.max_loss_in_pass_band * (
                self.x[end] - self.x[start]
            )
        )
        b = abs(
            sum(
                self.max_loss_in_pass_band - self.y[start:end]
            ) * self.distance
        )

        return (b / a, True)

    def evaluate_stop_band(self, start, end):
        c = abs(
            (self.required_loss_in_stop_band - self.max_loss_in_pass_band) * (
                (self.x[start] - self.x[0]) + (self.x[-1] - self.x[end])
            )
        )

        y1 = np.where(
            self.y[0:start] > self.required_loss_in_stop_band,
            self.max_loss_in_pass_band - self.y[0:start],
            self.max_loss_in_pass_band - self.required_loss_in_stop_band
        )
        y1 = np.where(
            y1 > 0,
            y1,
            0
        )
        y2 = np.where(
            self.y[end:-1] > self.required_loss_in_stop_band,
            self.max_loss_in_pass_band - self.y[end:-1],
            self.max_loss_in_pass_band - self.required_loss_in_stop_band
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

        return (d / c, True)

    def evaluate_cross_talk(self, number_of_cross_talk):
        if number_of_cross_talk > 0:
            return (0, False)
        return (0, True)


def build_Evaluator(config, weight=[1, 3, 3, 1, 4, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]):
    """Partial-apply config to Evaluator

    Args:
        config (Dict[str, Any]): Configuration of the Evaluator

    Returns:
        Evaluator_with_config: Evaluator that is partial-applied config to.
    """
    return lambda L, K: Evaluator(L, K, weight, config)
