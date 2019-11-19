from scipy.signal import argrelmin, argrelmax
import numpy as np


def shape_factor(x, y):
    print('shape factor')
    maxid = argrelmax(y, order=10)
    print(x[maxid])
    print(y[maxid])


def a(x, y):
    n = int(x.size * 0.1)
    delay_list = np.arange(n, x.size - n)
    diff_average_list = np.zeros(delay_list.size)

    for i, delay in enumerate(delay_list):
        padding = np.zeros(delay)
        delayed_y = np.append(padding, y)
        _y = np.append(y, padding)
        diff = np.abs(_y - delayed_y)[delay:-delay]
        diff_average = np.average(diff)
        diff_average_list[i] = diff_average

    minid = argrelmin(diff_average_list, order=1)[0]
    print(min(diff_average_list))
    print(delay_list[minid], diff_average_list[minid])
    return delay_list[minid], diff_average_list[minid]
    # if minid.size == 1:
    #     return minid, diff_average_list[minid]
    # else:
    #     a(minid, diff_average_list[minid])
