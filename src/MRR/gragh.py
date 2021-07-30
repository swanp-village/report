import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import rc
from matplotlib.ticker import (
    AutoLocator,
    FormatStrFormatter,
    MultipleLocator,
    NullFormatter,
)

rc("text", usetex=True)
rc("font", size=16)


class Gragh:
    def __init__(self, is_focus: bool = False):
        self.is_focus = is_focus
        self.fig, self.ax = plt.subplots(figsize=(8, 6))

    def plot(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        label: str,
    ) -> None:
        self.ax.semilogx(x * 1e9, y, label=label)

    def show(
        self,
        img_path: str = "img/out.pdf",
    ) -> None:
        self.ax.set_xlabel("Wavelength (nm)", fontsize=24)
        self.ax.set_ylabel("Drop Port Power (dB)", fontsize=24)
        self.ax.axis([None, None, None, 5])

        if self.is_focus:
            self.ax.set_xlim([1549, 1551])
            self.ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            self.ax.xaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
            self.ax.xaxis.set_minor_locator(MultipleLocator(0.5))
            self.ax.xaxis.set_minor_formatter(NullFormatter())
            self.ax.set_ylim([-12, 0])
            self.ax.yaxis.set_major_locator(MultipleLocator(2))
        else:
            self.ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
            self.ax.xaxis.set_minor_formatter(FormatStrFormatter("%d"))
            self.ax.xaxis.set_major_locator(AutoLocator())
            self.ax.set_ylim([-60, 0])
        plt.legend(loc="lower right")
        self.fig.savefig(img_path)
        plt.show()
