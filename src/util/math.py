from typing import List
import numpy as np
from functools import reduce


def lcm(xs: List[float]) -> float:
    return reduce(_lcm, xs)


def mod(x: float, y: float) -> float:
    return x - y * np.floor(x / y)


def equals(x: float, y: float, eps: float) -> bool:
    return np.abs(x - y) < eps


def calc_machine_eps(x: float) -> float:
    return 10 ** (np.floor(np.log10(x)))


def _gcd(x: float, y: float, eps: float) -> float:
    """
        x > y
    """
    if equals(y, 0, eps):
        return x
    else:
        return _gcd(y, mod(x, y), eps)


def _lcm(x: float, y: float) -> float:
    if x > y:
        return x * y / _gcd(y, x, calc_machine_eps(y))
    else:
        return y * x / _gcd(x, y, calc_machine_eps(x))
