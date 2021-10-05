import argparse

from config.base import config
from MRR.model.DE import Model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-plot", action="store_true")
    args = vars(parser.parse_args())
    skip_plot = args["skip_plot"]
    model = Model(config, skip_plot)
    model.optimize()
