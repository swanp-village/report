from MRR.Model.DE import Model


def optimize(config, skip_plot):
    model = Model(config, skip_plot)
    model.optimize()
