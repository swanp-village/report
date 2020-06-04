import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import FuncFormatter
import numpy as np


@FuncFormatter
def formatter(x, pos):
    return "%.1f" % x


def plot(xs, ys, number_of_rings, img_path='img/out.pdf'):
    xs = np.array(xs)
    ys = np.array(ys)
    if xs.ndim == 1:
        xs = np.array([xs])
        ys = np.array([ys])
    rc('text', usetex=True)
    rc('font', size=16)
    if number_of_rings == 1:
        title = '{}st order MRR'.format(number_of_rings)
    elif number_of_rings == 2:
        title = '{}nd order MRR'.format(number_of_rings)
    elif number_of_rings == 3:
        title = '{}rd order MRR'.format(number_of_rings)
    else:
        title = '{}th order MRR'.format(number_of_rings)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(xs.shape[0]):
        ax.semilogx(xs[i] * 1e9, ys[i])
    ax.set(
        xlabel='Wavelength[nm]',
        ylabel='Drop Port Power [dB]',
        title=title
    )
    ax.axis([None, None, None, 5])
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_formatter(formatter)
    ax.set_ylim([-60, 0])
    ax.grid(which='both')
    fig.savefig(img_path)
    plt.show()
