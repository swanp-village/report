import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


class MyFigure(Figure):
    """A figure with a text watermark."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(self)
        # self.set_xlabel(xlabel="Wavelength (nm)", fontsize=24)
        # self.set_ylabel("Drop Port Power (dB)", fontsize=24)


x = np.linspace(-3, 3, 201)
y = np.tanh(x) + 0.1 * np.cos(5 * x)

plt.figure(FigureClass=MyFigure, figsize=(8, 6), xlabel=())
plt.plot(x, y)
plt.show()
