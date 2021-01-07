from ..model import Model
import numpy as np

def generate_model(three_db_band, rank):
    m = Model('three_db_band{}'.format(three_db_band))
    d = np.arange(-60, 0, 0.5)
    length_of_3db_band = m.length_of_3db_band * three_db_band
    y = np.hstack((
        np.repeat(-60, (m.FSR - length_of_3db_band) / 2 / 1e-12 - d.size),
        d,
        np.repeat(0, length_of_3db_band / 1e-12),
        d[::-1],
        np.repeat(-60, (m.FSR - length_of_3db_band) / 2 / 1e-12 - d.size)
    ))
    m.set_y(y)
    m.set_rank(rank)

    return m

data = [
    *[
        generate_model(i / 100, 3)
        for i in range(80, 95, 1)
    ],
    *[
        generate_model(i / 100, 2)
        for i in range(95, 99, 1)
    ],
    *[
        generate_model(i / 100, 1)
        for i in range(99, 102, 1)
    ],
    *[
        generate_model(i / 100, 2)
        for i in range(102, 106, 1)
    ],
    *[
        generate_model(i / 100, 3)
        for i in range(106, 120, 1)
    ]
]
