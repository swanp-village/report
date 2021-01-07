from ..model import Model
import numpy as np

def generate_model(insertion_loss, rank):
    m = Model('insertion_loss{}'.format(insertion_loss))
    d = np.arange(-60, insertion_loss, 0.5)
    y = np.hstack((
        np.repeat(-60, (m.FSR - m.length_of_3db_band) / 2 / 1e-12 - d.size),
        d,
        np.repeat(insertion_loss, m.length_of_3db_band / 1e-12),
        d[::-1],
        np.repeat(-60, (m.FSR - m.length_of_3db_band) / 2 / 1e-12 - d.size)
    ))
    m.set_y(y)
    m.set_rank(1)

    return m

data = [
    *[
        generate_model(i, 1)
        for i in range(-1, -4, -1)
    ],
    *[
        generate_model(i, 2)
        for i in range(-4, -20, -1)
    ],
    *[
        generate_model(i, 3)
        for i in range(-20, -30, -1)
    ]
]
