from math import isclose
from typing import List
import numpy as np
from functools import reduce


def calc_eps(x: float, y: float) -> float:
    diff = 10 ** np.floor(abs(np.log10(x) - np.log10(y)))

    return min(x, y) * 0.1 * diff


def is_zero(x: float, eps: float) -> bool:
    return isclose(x, 0, abs_tol=eps)


def lcm(xs: List[float]) -> float:
    return reduce(_lcm, xs)


def mod(x: float, y: float) -> float:
    return x - y * np.floor(x / y)


def _gcd(x: float, y: float, eps: float) -> float:
    """
        x > y
    """
    if is_zero(y, eps):
        return x
    else:
        return _gcd(y, mod(x, y), eps)


def _lcm(x: float, y: float) -> float:
    eps = calc_eps(x, y)
    if x > y:
        return x * y / _gcd(y, x, eps)
    else:
        return y * x / _gcd(x, y, eps)
