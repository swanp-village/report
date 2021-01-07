from ..model import Model
import numpy as np

def generate_model(ripple, rank):
    m = Model('ripple{}'.format(ripple))
    d1 = np.arange(-60, 0, 0.5)
    d2 = np.arange(ripple, 0, 0.5)
    y = np.hstack((
        np.repeat(-60, (m.FSR - m.length_of_3db_band) / 2 / 1e-12 - d1.size),
        d1,
        np.repeat(0, m.length_of_3db_band / 5 / 1e-12 - d2.size),
        d2[::-1],
        np.repeat(ripple, m.length_of_3db_band / 5 / 1e-12 - d2.size),
        d2,
        np.repeat(0, m.length_of_3db_band / 5 / 1e-12),
        d2[::-1],
        np.repeat(ripple, m.length_of_3db_band / 5 / 1e-12 - d2.size),
        d2,
        np.repeat(0, m.length_of_3db_band / 5 / 1e-12 - d2.size),
        d1[::-1],
        np.repeat(-60, (m.FSR - m.length_of_3db_band) / 2 / 1e-12 - d1.size)
    ))
    m.set_y(y)
    m.set_rank(rank)

    return m

data = [
    *[
        generate_model(i / 10, 1)
        for i in range(0, -10, -1)
    ],
    *[
        generate_model(i / 10, 2)
        for i in range(-10, -20, -1)
    ],
    *[
        generate_model(i / 10, 3)
        for i in range(-20, -50, -1)
    ]
]
