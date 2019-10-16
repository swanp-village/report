import numpy as np
from functools import reduce


def lcm(xs):
    return reduce(_lcm, xs)


def mod(x, y):
    return x - y * np.floor(x / y)


def equals(x, y, eps):
    return np.abs(x - y) < eps


def calc_machine_eps(x):
    return 10 ** (np.floor(np.log10(x)) - 2)


def _gcd(x, y, eps):
    """
        x < y
    """
    if equals(y, 0, eps):
        return x
    else:
        return _gcd(y, mod(x, y), eps)


def _lcm(x, y):
    if x > y:
        return x * y / _gcd(y, x, calc_machine_eps(y))
    else:
        return y * x / _gcd(x, y, calc_machine_eps(x))
