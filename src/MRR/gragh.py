import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import FuncFormatter


@FuncFormatter
def formatter(x, pos):
    return "%d" % x


def plot(x, y, number_of_rings, img_path='img/out.pdf'):
    rc('text', usetex=True)
    if number_of_rings == 1:
        title = '{}st order MRR'.format(number_of_rings)
    elif number_of_rings == 2:
        title = '{}nd order MRR'.format(number_of_rings)
    elif number_of_rings == 3:
        title = '{}rd order MRR'.format(number_of_rings)
    else:
        title = '{}th order MRR'.format(number_of_rings)
    fig, ax = plt.subplots()
    ax.semilogx(x * 1e9, y)
    ax.set(
        xlabel='Wavelength[nm]',
        ylabel='Drop Port Power [dB]',
        title=title
    )
    ax.axis([None, None, None, 5])
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_formatter(formatter)
    ax.grid(which='both')
    fig.savefig(img_path)
    plt.show()
