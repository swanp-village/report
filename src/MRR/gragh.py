import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import rc
from matplotlib.ticker import AutoLocator, FuncFormatter, MultipleLocator, NullFormatter


@FuncFormatter  # type: ignore
def formatter(x: float, pos: int) -> str:
    return "%d" % x


rc("text", usetex=True)
rc("font", size=16)


class Gragh:
    def __init__(self, is_focus: bool = False):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel("Wavelength (nm)", fontsize=24)
        ax.set_ylabel("Drop Port Power (dB)", fontsize=24)
        # title = self.__generate_title(number_of_rings)
        # ax.set(
        #     title=title
        # )
        ax.axis([None, None, None, 5])
        ax.set_ylim([-60, 0])
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_major_locator(AutoLocator())
        ax.xaxis.set_minor_formatter(formatter)
        if is_focus:
            ax.set_xlim([1549, 1551])
            ax.xaxis.set_major_locator(MultipleLocator())
            ax.xaxis.set_minor_locator(MultipleLocator(0.5))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_ylim([-12, 0])
            ax.yaxis.set_major_locator(MultipleLocator(2))
        self.ax = ax
        self.fig = fig

    def __generate_title(self, number_of_rings: int) -> str:
        if number_of_rings == 1:
            return f"{number_of_rings}st order MRR"
        elif number_of_rings == 2:
            return f"{number_of_rings}nd order MRR"
        elif number_of_rings == 3:
            return f"{number_of_rings}rd order MRR"
        else:
            return f"{number_of_rings}th order MRR"

    def plot(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> None:
        self.ax.semilogx(x * 1e9, y)

    def show(
        self,
        img_path: str = "img/out.pdf",
    ) -> None:
        self.fig.savefig(img_path)
        plt.show()
