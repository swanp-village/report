import numpy as np


def mod(x, y):
    return x - y * np.floor(x / y)


def gcd(x, y):
    if equals(y, 0):
        return x
    else:
        print(x, y)
        return gcd(y, mod(x, y))


def lcm(x, y):
    return x * y / gcd(x, y)


eps = 0.000001


def equals(x, y):
    return x - y < eps
