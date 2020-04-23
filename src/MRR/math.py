from typing import List
import numpy as np
from functools import reduce


def is_zero(x: float, y: float) -> bool:
    """
        x > y > 0
    """
    result = x / y - np.floor(x / y)
    return result < 0.01


def lcm(xs: List[float]) -> float:
    return reduce(_lcm, xs)


def mod(x: float, y: float) -> float:
    return x - y * np.floor(x / y)


def _gcd(x: float, y: float) -> float:
    """
        x > y
    """
    if y == 0:
        return x
    elif is_zero(x, y):
        return y
    else:
        return _gcd(y, mod(x, y))


def _lcm(x: float, y: float) -> float:
    if x > y:
        return x * y / _gcd(x, y)
    else:
        return x * y / _gcd(y, x)
