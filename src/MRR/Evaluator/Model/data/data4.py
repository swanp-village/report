from ..model import Model
import numpy as np

data = Model('data4')
d = np.arange(-60, 0, 0.5)
y = np.hstack((
    np.repeat(-60, (data.FSR - data.length_of_3db_band) / 2 / 1e-12 - d.size),
    d,
    np.repeat(0, data.length_of_3db_band / 1e-12),
    d[::-1],
    np.repeat(-60, (data.FSR - data.length_of_3db_band) / 2 / 1e-12 - d.size)
))
data.set_y(y)
data.set_rank(1)
