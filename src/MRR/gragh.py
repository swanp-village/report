import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import FuncFormatter
import numpy as np


@FuncFormatter
def formatter(x, pos):
    return "%.1f" % x


def generate_title(number_of_rings):
    if number_of_rings == 1:
        return '{}st order MRR'.format(number_of_rings)
    elif number_of_rings == 2:
        return '{}nd order MRR'.format(number_of_rings)
    elif number_of_rings == 3:
        return '{}rd order MRR'.format(number_of_rings)
    else:
        return '{}th order MRR'.format(number_of_rings)

def validate(xs, ys, number_of_rings):
    if not isinstance(xs, list):
        print('xs is wrong')
    if not isinstance(ys, list):
        print('ys is wrong')
    if len(xs) == number_of_rings and len(ys) == number_of_rings:
        print('number_of_rings is wrong')


def plot(xs, ys, number_of_rings, img_path='img/out.pdf', skip_plot=False):
    validate(xs, ys, number_of_rings)
    rc('text', usetex=True)
    rc('font', size=16)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(len(xs)):
        ax.semilogx(xs[i] * 1e9, ys[i])
    ax.set_xlabel('Wavelength (nm)', fontsize=24)
    ax.set_ylabel('Drop Port Power (dB)', fontsize=24)
    # title = generate_title(number_of_rings)
    # ax.set(
    #     title=title
    # )
    ax.axis([None, None, None, 5])
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_formatter(formatter)
    # ax.set_xticks([1500, 1550, 1600], False)
    # ax.set_xticks([], True)
    ax.set_ylim([-30, 0])
    # ax.grid(which='both')
    fig.savefig(img_path)
    if not skip_plot:
        plt.show()


def plot_(xs, ys, number_of_rings, img_path='img/out.pdf'):
    xs = np.array(xs)
    ys = np.array(ys)
    if xs.ndim == 1:
        xs = np.array([xs])
        ys = np.array([ys])
    rc('text', usetex=True)
    rc('font', size=16)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(xs.shape[0]):
        ax.semilogx(xs[i] * 1e9, ys[i])
    ax.set_xlabel('Wavelength (nm)', fontsize=24)
    ax.set_ylabel('Drop Port Power (dB)', fontsize=24)
    ax.axis([None, None, None, 5])
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_formatter(formatter)
    ax.set_ylim([-70, 0])
    ax.set_xlim([1540, 1561])
    ax.set_xticks(np.arange(1540, 1561, 2))
    ax.set_xticks([], True)
    fig.savefig(img_path)
    if not skip_plot:
            plt.show()
