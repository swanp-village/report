from ..model import Model
import numpy as np

data = Model('data8')
d1 = np.arange(-60, 0, 0.5)
d2 = np.arange(-7, 0, 0.5)
y = np.hstack((
    np.repeat(-60, (data.FSR - data.length_of_3db_band) / 2 / 1e-12 - d1.size),
    d1,
    np.repeat(0, data.length_of_3db_band / 5 / 1e-12 - d2.size),
    d2[::-1],
    np.repeat(-7, data.length_of_3db_band / 5 / 1e-12 - d2.size),
    d2,
    np.repeat(0, data.length_of_3db_band / 5 / 1e-12),
    d2[::-1],
    np.repeat(-7, data.length_of_3db_band / 5 / 1e-12 - d2.size),
    d2,
    np.repeat(0, data.length_of_3db_band / 5 / 1e-12 - d2.size),
    d1[::-1],
    np.repeat(-60, (data.FSR - data.length_of_3db_band) / 2 / 1e-12 - d1.size)
))
data.set_y(y)
