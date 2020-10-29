from ..model import Model
import numpy as np

data = Model('data2')
d = np.arange(-40, 0, 0.5)
y = np.hstack((
    np.repeat(-40, (data.FSR - data.length_of_3db_band) / 2 / 1e-12 - d.size),
    d,
    np.repeat(0, data.length_of_3db_band / 1e-12),
    d[::-1],
    np.repeat(-40, (data.FSR - data.length_of_3db_band) / 2 / 1e-12 - d.size)
))
data.set_y(y)
