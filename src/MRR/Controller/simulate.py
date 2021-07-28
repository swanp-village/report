import numpy as np
import numpy.typing as npt

from MRR.Evaluator import build_Evaluator
from MRR.gragh import plot
from MRR.logger import Logger
from MRR.Simulator import Ring, TransferFunction
from src.config.model import SimulationConfig


class Simulator:
    def __init__(self, is_focus: bool) -> None:
        self.logger = Logger()
        self.xs: list[npt.NDArray[np.float64]] = []
        self.ys: list[npt.NDArray[np.float64]] = []
        self.is_focus: bool = is_focus
        self.number_of_rings: int = 0

    def simulate(self, config: SimulationConfig) -> None:
        Evaluator = build_Evaluator(config)
        mrr = TransferFunction(config.L, config.K, config)
        mrr.print_parameters()
        ring = Ring(config)
        N = ring.calculate_N(config.L)
        FSR = ring.calculate_practical_FSR(N)

        if config.lambda_limit:
            x = config.lambda_limit
            y = mrr.simulate(x)
        else:
            x = ring.calculate_x(FSR)
            y = mrr.simulate(x)
            evaluator = Evaluator(x, y)
            result = evaluator.evaluate_band()
            print(result)

        self.logger.save_data_as_csv(x, y, config.name)
        self.xs.append(x)
        self.ys.append(y)

    def plot(self, name: str) -> None:
        plot(self.xs, self.ys, self.number_of_rings, self.logger.generate_image_path(name), self.is_focus)
