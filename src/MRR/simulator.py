import numpy as np
from typing import Any, Dict, List


class MRR:
    """Simulator of the transfer function of the MRR filter.

    Args:
        L (List[float]): List of the round-trip length.
        K (List[float]): List of the coupling rate.
        config (Dict[str, Any]): Configuration of the MRR.
            Keys:
                eta (float): The coupling loss coefficient.
                n_eq (float): The equivalent refractive index.
                alpha (float): The propagation loss coefficient.
    Attributes:
        L (List[float]): List of the round-trip length.
        K (List[float]): List of the coupling rate.
        eta (float): The coupling loss coefficient.
        n_eq (float): The equivalent refractive index.
        a (List[float]): List of the propagation loss.
    """

    def __init__(
        self,
        L,
        K,
        config: Dict[str, Any]
    ) -> None:
        self.L: List[float] = L
        self.K: List[float] = K
        self.eta: float = config['eta']
        self.n_eq: float = config['n_eq']
        self.a: List[float] = np.exp(- config['alpha'] * config['L'])

    def _C(self, K_k: float) -> np.matrix:
        return 1 / (-1j * self.eta * np.sqrt(K_k)) * np.matrix([
            [1, - self.eta * np.sqrt(self.eta - K_k)],
            [np.sqrt(self.eta - K_k) * self.eta, - self.eta ** 2]
        ])

    def _R(self, a_k: float, L_k: float, l: np.array) -> np.matrix:
        return np.matrix([
            [np.exp(1j * np.pi * L_k * self.n_eq / l) / np.sqrt(a_k), 0],
            [0, np.exp(-1j * np.pi * L_k * self.n_eq / l) * np.sqrt(a_k)]
        ])

    def _reverse(self, arr: List) -> List:
        return arr[::-1]

    def _M(self, l: List[float]) -> np.matrix:
        product = np.identity(2)
        for _K, _a, _L in zip(
            self._reverse(self.K[1:]),
            self._reverse(self.a),
            self._reverse(self.L)
        ):
            product = product * self._C(_K) * self._R(_a, _L, l)
        product = product * self._C(self.K[0])
        return product

    def _D(self, l: List[float]) -> np.matrix:
        return 1 / self._M(l)[0, 0]

    def print_parameters(self) -> None:
        print('eta:', self.eta)
        print('n_eq:', self.n_eq)
        print('a:', self.a)
        print('K:', self.K)
        print('L:', self.L)

    def simulate(self, l: List[float]) -> np.array:
        y = 20 * np.log10(np.abs(self._D(l)))
        return y.reshape(y.size)


def factory_MRR(config):
    """Partial apply config for MRR

    Args:
        config (Dict[str, Any]): Configuration of the MRR

    Returns:
        partial_applied_MRR: MRR that is partial-applied config.
    """
    return lambda L, K: MRR(L, K, config)
