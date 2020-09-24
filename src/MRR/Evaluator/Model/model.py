import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from os.path import join
from .config import config
# from config import config


@FuncFormatter
def formatter(x, pos):
    return "%.1f" % x


class Model:
    def __init__(
        self,
        name,
    ):
        self.FSR = config['FSR']
        self.center_wavelength = config['center_wavelength']
        self.length_of_3db_band = config['length_of_3db_band']
        self.x = np.hstack((
            np.arange(self.center_wavelength - self.FSR / 2, self.center_wavelength, 1e-12),
            np.arange(self.center_wavelength, self.center_wavelength + self.FSR / 2, 1e-12)
        ))
        self.y = np.repeat(0, self.x.size)
        self.img_path = join('MRR/Evaluator/Model/figure', '{}.pdf'.format(name))
        self.validate()

    def export_gragh(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Wavelength (nm)', fontsize=24)
        ax.set_ylabel('Drop Port Power (dB)', fontsize=24)
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xticks([1500, 1550, 1600], False)
        ax.set_ylim([-70, 0])

        ax.semilogx(self.x * 1e9, self.y)
        fig.savefig(self.img_path)
        plt.show()

    def validate(self):
        if self.x.size != self.FSR / 1e-12:
            print('The size of x is wrong.')
        if self.x.size != self.y.size:
            print('The size of x does not equal to the size of y.')
        if self.x[self.x.size // 2] != self.center_wavelength:
            print('The center value of x is not the center wavelength.')

    def set_y(self, y):
        self.y = y
        self.validate()

