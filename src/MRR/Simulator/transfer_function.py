import numpy as np
import numpy.typing as npt

from config.model import SimulationConfig


class TransferFunction:
    """Simulator of the transfer function of the MRR filter.

    Args:
        L (List[float]): List of the round-trip length.
        K (List[float]): List of the coupling rate.
        config (Dict[str, Any]): Configuration of the MRR.
            Keys:
                eta (float): The coupling loss coefficient.
                n_eff (float): The effective refractive index.
                n_g (float): The group index.
                alpha (float): The propagation loss coefficient.
    Attributes:
        L (List[float]): List of the round-trip length.
        K (List[float]): List of the coupling rate.
        eta (float): The coupling loss coefficient.
        n_eff (float): The effective refractive index.
        n_g (float): The group index.
        a (List[float]): List of the propagation loss.
    """

    def __init__(self, L: npt.ArrayLike, K: npt.ArrayLike, config: SimulationConfig) -> None:
        self.L: npt.NDArray[np.float64] = np.array(L)
        self.K: npt.NDArray[np.float64] = np.array(K)
        self.center_wavelength: float = config.center_wavelength
        self.eta: float = config.eta
        self.n_eff: float = config.n_eff
        self.n_g: float = config.n_g
        self.a: npt.NDArray[np.float64] = np.exp(-config.alpha * L)

    def _C(self, K_k: float) -> npt.NDArray[np.float64]:
        C: npt.NDArray[np.float64] = (
            1
            / (-1j * self.eta * np.sqrt(K_k))
            * np.array([[1, -self.eta * np.sqrt(self.eta - K_k)], [np.sqrt(self.eta - K_k) * self.eta, -self.eta ** 2]])
        )
        return C

    def _R(self, a_k: float, L_k: float, wavelength: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x = (
            1j
            * np.pi
            * L_k
            * self.n_g
            * (wavelength - self.center_wavelength)
            / self.center_wavelength
            / self.center_wavelength
        )
        return np.array([[np.exp(x) / np.sqrt(a_k), 0], [0, np.exp(-x) * np.sqrt(a_k)]], dtype="object")

    def _reverse(self, arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        revesed_arr: npt.NDArray[np.float64] = arr[::-1]
        return revesed_arr

    def _M(self, wavelength: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        product = np.identity(2)
        for _K, _a, _L in zip(self._reverse(self.K[1:]), self._reverse(self.a), self._reverse(self.L)):
            product = np.dot(product, self._C(_K))
            product = np.dot(product, self._R(_a, _L, wavelength))
        product = np.dot(product, self._C(self.K[0]))
        return product

    def _D(self, wavelength: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        D: npt.NDArray[np.float64] = 1 / self._M(wavelength)[0, 0]
        return D

    def print_parameters(self) -> None:
        print("eta:", self.eta)
        print("center_wavelength:", self.center_wavelength)
        print("n_eff:", self.n_eff)
        print("n_g:", self.n_g)
        print("a:", self.a)

    def simulate(self, wavelength: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        y: npt.NDArray[np.float64] = 20 * np.log10(np.abs(self._D(wavelength)))
        return y.reshape(y.size)


class build_TransferFunction_Factory:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config

    def create(self, L: npt.ArrayLike, K: npt.ArrayLike) -> TransferFunction:
        return TransferFunction(L, K, self.config)


def build_TransferFunction(config: SimulationConfig) -> TransferFunction:
    """Partial-apply config to TransferFunction

    Args:
        config (Dict[str, Any]): Configuration of the TransferFunction

    Returns:
        TransferFunction_with_config: TransferFunction that is partial-applied config to.
    """
    return build_TransferFunction_Factory(config).create
