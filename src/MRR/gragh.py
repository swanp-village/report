import matplotlib.pyplot as plt

def plot(x, y, title):
    plt.semilogx(x * 1e9, y)
    plt.xlabel('Wavelength[nm]')
    plt.ylabel('Drop Port Power [dB]')
    plt.title(title)
    plt.axis([None, None, None, 5])
    plt.grid()
    plt.show()
