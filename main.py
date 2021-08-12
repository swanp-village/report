import argparse
from importlib import import_module

from config.base import config
from config.model import SimulationConfig
from MRR.Evaluator.Model.train import show_data, train_evaluator
from MRR.model.DE import Model
from MRR.simulator import Simulator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="config file path", nargs="*")
    parser.add_argument("--train-evaluator", action="store_true")
    parser.add_argument("--show-data", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--format", action="store_true")
    parser.add_argument("-f", "--focus", action="store_true")
    args = vars(parser.parse_args())
    skip_plot = args["skip_plot"]
    is_focus = args["focus"]
    format = args["format"]
    if args["config"]:
        simulator = Simulator(is_focus)
        try:
            for name in args["config"]:
                imported_config = import_module(f"config.simulate.{name}").config
                simulation_config = SimulationConfig(**imported_config)
                simulation_config.name = name
                simulation_config.format = format
                simulator.simulate(simulation_config)
            if not skip_plot:
                simulator.show()
        except ModuleNotFoundError as e:
            print(e)
            parser.print_help()
    elif args["train_evaluator"]:
        train_evaluator()
    elif args["show_data"]:
        show_data(skip_plot)
    else:
        model = Model(config, skip_plot)
        model.optimize()
