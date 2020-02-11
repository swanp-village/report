import matplotlib.pyplot as plt


def plot(x, y, number_of_rings, img_path='img/out.png'):
    print(number_of_rings == 2)
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
    ax.set(xlabel='Wavelength[nm]', ylabel='Drop Port Power [dB]', title=title)
    ax.axis([None, None, None, 5])
    ax.grid()
    fig.savefig(img_path)
    plt.show()
