import argparse
import csv
from importlib import import_module
from typing import Any

from config.base import config
from config.model import SimulationConfig
from MRR.Evaluator.Model.train import show_data, train_evaluator
from MRR.model.DE import Model
from MRR.simulator import Simulator, SimulatorResult

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
        results: list[SimulatorResult] = []
        simulator = Simulator(is_focus)
        for name in args["config"]:
            try:
                imported_module = import_module(f"config.simulate.{name}")
                imported_config = getattr(imported_module, "config")
                simulation_config = SimulationConfig(**imported_config)
                simulation_config.name = name
                simulation_config.format = format
                result = simulator.simulate(simulation_config)
                results.append(result)
            except ModuleNotFoundError as e:
                print(e)

        # Plot with PGFPlots
        basedir = simulator.logger.target
        steps = [(1 if len(result.x) < 500 else len(result.x) // 500) for result in results]
        for result, step in zip(results, steps):
            with open(f"{basedir}/{result.name}_pgfplots.tsv", "w") as tsvfile:
                x = result.x[::step]
                y = result.y[::step]
                tsv_writer = csv.writer(tsvfile, delimiter="\t")
                tsv_writer.writerows(zip(x.tolist(), y.tolist()))
        import subprocess

        from jinja2 import Environment, PackageLoader

        env = Environment(loader=PackageLoader("MRR"))
        template = env.get_template("pgfplots.tex.j2")
        legends = "{" + ",".join([result.label for result in results]) + "}"
        tsvnames = ["{" + result.name + "_pgfplots.tsv}" for result in results]
        with open(basedir / "pgfplots.tex", "w") as fp:
            fp.write(template.render(tsvnames=tsvnames, legends=legends))
        subprocess.run(["lualatex", "pgfplots"], cwd=basedir)
        # End of plot with PGFPlots

        if not skip_plot:
            simulator.show()
    elif args["train_evaluator"]:
        train_evaluator()
    elif args["show_data"]:
        show_data(skip_plot)
    else:
        model = Model(config, skip_plot)
        model.optimize()
