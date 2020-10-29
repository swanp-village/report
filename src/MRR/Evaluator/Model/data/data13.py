from ..model import Model
import numpy as np

data = Model('data13')
d1 = np.arange(-60, 0, 0.5)
d2 = np.arange(-60, -40, 0.5)
y = np.hstack((
    np.repeat(-60, (data.FSR - data.length_of_3db_band) / 4 / 1e-12 - d2.size),
    d2,
    d2[::-1],
    np.repeat(-60, (data.FSR - data.length_of_3db_band) / 4 / 1e-12 - d2.size - d1.size),
    d1,
    np.repeat(0, data.length_of_3db_band / 1e-12),
    d1[::-1],
    np.repeat(-60, (data.FSR - data.length_of_3db_band) / 4 / 1e-12 - d2.size - d1.size),
    d2,
    d2[::-1],
    np.repeat(-60, (data.FSR - data.length_of_3db_band) / 4 / 1e-12 - d2.size)
))
data.set_y(y)
