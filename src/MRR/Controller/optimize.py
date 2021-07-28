from random import seed

from MRR.Model.DE import Model


def optimize(config, skip_plot):
    # seed(1)
    model = Model(config, skip_plot)
    model.train()
