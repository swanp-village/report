import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import numpy.typing as npt

from config.model import OptimizationConfig, SimulationConfig


class Logger:
    def __init__(self) -> None:
        format = "%Y-%m-%d-%H-%M-%S"
        self.result_path = Path.cwd() / "result"
        self.result_path.mkdir(exist_ok=True)
        self.target = self.result_path / datetime.now().strftime(format)
        self.target.mkdir()

    def save_optimization_config(self, config: OptimizationConfig) -> None:
        self.config = config
        src = json.dumps(asdict(config), indent=4)
        path = self.target / "config.json"
        path.write_text(src)

    def save_simulation_config(self, config: SimulationConfig) -> None:
        self.config = config
        src = json.dumps(
            {**asdict(config), "K": config.K.tolist(), "L": config.L.tolist(), "lambda_limit": []},
            indent=4,
        )
        path = self.target / "config.json"
        path.write_text(src)

    def generate_image_path(self, name: str = "out") -> Path:
        return self.target / f"{name}.pdf"

    def save_data_as_csv(self, x: npt.NDArray[np.float_], y: npt.NDArray[np.float_], name: str = "out") -> None:
        path = self.target / f"{name}.tsv"
        with open(path, "w") as tsvfile:
            tsv_writer = csv.writer(tsvfile, delimiter="\t")
            tsv_writer.writerows(zip(x.tolist(), y.tolist()))

    def save_evaluation_value(
        self, E_list: list[float], method_list: list[int], name: str = "evaluation_value"
    ) -> None:
        path = self.target / f"{name}.tsv"
        with open(path, "w") as tsvfile:
            tsv_writer = csv.writer(tsvfile, delimiter="\t")
            tsv_writer.writerows(zip(E_list, method_list))

    def save_result(self, L: npt.NDArray[np.float_], K: npt.NDArray[np.float_]) -> None:
        result = {
            "eta": self.config.eta,
            "alpha": self.config.alpha,
            "K": K,
            "L": L,
            "n_eff": self.config.n_eff,
            "n_g": self.config.n_g,
            "center_wavelength": self.config.center_wavelength,
        }
        src = json.dumps(result, indent=4)
        path = self.target / "result.json"
        path.write_text(src)

    def typeset_pgfplots_graph(self, tsv_name: Optional[str] = "out", label: str = "") -> None:
        import subprocess
        from importlib.resources import read_text

        from MRR import templates

        tsv_path = f"{tsv_name}.tsv"
        template = read_text(templates, "pgfplots.tex").replace("data.txt", tsv_path).replace("Legend Text", label)
        with open(self.target.joinpath("pgfplots.tex"), "w") as fp:
            fp.write(template)
        subprocess.run(["lualatex", "pgfplots"], cwd=self.target)

    def print_parameters(
        self, K: npt.NDArray[np.float_], L: npt.NDArray[np.float_], N: npt.NDArray[np.int_], FSR: float, E: float
    ) -> None:
        print("K  : {}".format(K.tolist()))
        print("L  : {}".format(L.tolist()))
        print("N  : {}".format(N.tolist()))
        print("FSR: {}".format(FSR))
        print("E  : {}".format(E))
