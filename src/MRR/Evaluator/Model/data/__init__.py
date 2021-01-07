from .stopband import data as stopband
from .ripple import data as ripple
from .crosstalk import data as crosstalk
from .three_db_band import data as three_db_band
from .insertion_loss import data as insertion_loss


data = [
    *crosstalk,
    *insertion_loss,
    *ripple,
    *stopband,
    *three_db_band,
]
