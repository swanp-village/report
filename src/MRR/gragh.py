import matplotlib.pyplot as plt

def plot(x, y, title, img_path='img/out.png'):
    fig, ax = plt.subplots()
    ax.semilogx(x * 1e9, y)
    ax.set(xlabel='Wavelength[nm]', ylabel='Drop Port Power [dB]', title=title)
    ax.axis([None, None, None, 5])
    ax.grid()
    fig.savefig(img_path)
    plt.show()
