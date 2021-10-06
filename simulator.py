import argparse
import csv
import os.path
import subprocess
from glob import glob
from importlib import import_module
from pathlib import Path

from jinja2 import Environment, PackageLoader

from config.model import SimulationConfig
from MRR.simulator import Simulator, SimulatorResult


def plot_with_pgfplots(basedir: Path, results: list[SimulatorResult], is_focus: bool) -> None:
    max_points = 2500
    steps = [(1 if result.x.size < max_points else result.x.size // max_points) for result in results]
    for result, step in zip(results, steps):
        with open(f"{basedir}/{result.name}_pgfplots.tsv", "w") as tsvfile:
            x = result.x[::step]
            y = result.y[::step]
            tsv_writer = csv.writer(tsvfile, delimiter="\t")
            tsv_writer.writerows(zip(x, y))

    env = Environment(loader=PackageLoader("MRR"))
    template = env.get_template("pgfplots.tex.j2")
    legends = "{" + ",".join([result.label for result in results]) + "}"
    tsvnames = ["{" + result.name + "_pgfplots.tsv}" for result in results]
    with open(basedir / "pgfplots.tex", "w") as fp:
        fp.write(template.render(tsvnames=tsvnames, legends=legends, is_focus=is_focus))
    subprocess.run(["lualatex", "pgfplots"], cwd=basedir, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("NAME", help="from config.simulate import NAME", nargs="*")
    parser.add_argument("-l", "--list", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--format", action="store_true")
    parser.add_argument("-f", "--focus", action="store_true")
    parser.add_argument("-s", "--simulate-one-cycle", action="store_true")
    args = vars(parser.parse_args())
    ls = args["list"]
    skip_plot = args["skip_plot"]
    is_focus = args["focus"]
    format = args["format"]
    simulate_one_cycle = args["simulate_one_cycle"]

    results: list[SimulatorResult] = []
    simulator = Simulator(is_focus)
    if ls:
        print("\t".join([os.path.splitext(os.path.basename(p))[0] for p in sorted(glob("config/simulate/*.py"))]))
    else:
        for name in args["NAME"]:
            try:
                imported_module = import_module(f"config.simulate.{name}")
                imported_config = getattr(imported_module, "config")
                simulation_config = SimulationConfig(**imported_config)
                simulation_config.name = name
                simulation_config.format = format
                simulation_config.simulate_one_cycle = simulate_one_cycle
                result = simulator.simulate(simulation_config)
                results.append(result)
                print("E:", result.evaluation_result)
            except ModuleNotFoundError as e:
                print(e)

        plot_with_pgfplots(simulator.logger.target, results, simulator.graph.is_focus)

        if not skip_plot:
            simulator.show()
