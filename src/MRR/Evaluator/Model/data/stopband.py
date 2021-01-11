from ..model import Model
import numpy as np

def generate_model(stopband, rank):
    m = Model('stopband{}'.format(stopband))
    d = np.arange(stopband, 0, 0.5)
    y = np.hstack((
        np.repeat(stopband, (m.FSR - m.length_of_3db_band) / 2 / 1e-12 - d.size),
        d,
        np.repeat(0, m.length_of_3db_band / 1e-12),
        d[::-1],
        np.repeat(stopband, (m.FSR - m.length_of_3db_band) / 2 / 1e-12 - d.size)
    ))
    m.set_y(y)
    m.set_rank(rank)

    return m

data = [
    *[
        generate_model(i, 3)
        for i in np.arange(-30, -40, -1)
    ],
    *[
        generate_model(i, 2)
        for i in np.arange(-40, -50, -1)
    ],
    *[
        generate_model(i, 1)
        for i in np.arange(-50, -60, -1)
    ]
]
