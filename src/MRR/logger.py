import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import numpy.typing as npt

from config.model import OptimizationConfig, SimulationConfig


class Logger:
    def __init__(self) -> None:
        format = "%Y-%m-%d-%H-%M-%S"
        self.result_path = Path.cwd().parent.joinpath("result")
        self.result_path.mkdir(exist_ok=True)
        self.target = self.result_path.joinpath(datetime.now().strftime(format))
        self.target.mkdir()

    def save_config(self, config: Union[OptimizationConfig, SimulationConfig]) -> None:
        self.config = config
        src = json.dumps(config, indent=4)
        self.target.joinpath("config.json").write_text(src)

    def generate_image_path(self, name: str = "out") -> Path:
        return self.target.joinpath(f"{name}.pdf")

    def save_data_as_csv(self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], name: str = "out") -> None:
        path = self.target.joinpath(f"{name}.tsv")
        with open(path, "w") as tsvfile:
            tsv_writer = csv.writer(tsvfile, delimiter="\t")
            tsv_writer.writerows(zip(x.tolist(), y.tolist()))

    def save_evaluation_value(
        self, E_list: npt.NDArray[np.float64], method_list: list[int], name: str = "evaluation_value"
    ) -> None:
        path = self.target.joinpath(f"{name}.tsv")
        with open(path, "w") as tsvfile:
            tsv_writer = csv.writer(tsvfile, delimiter="\t")
            tsv_writer.writerows(zip(E_list, method_list))

    def save_result(self, L: npt.NDArray[np.float64], K: npt.NDArray[np.float64]) -> None:
        result = {
            "eta": self.config["eta"],
            "alpha": self.config["alpha"],
            "K": K,
            "L": L,
            "n_eff": self.config["n_eff"],
            "n_g": self.config["n_g"],
            "center_wavelength": self.config["center_wavelength"],
        }
        src = json.dumps(result, indent=4)
        self.target.joinpath("result.json").write_text(src)
