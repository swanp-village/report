import numpy as np
import numpy.typing as npt

from config.model import SimulationConfig
from MRR.Evaluator import build_Evaluator
from MRR.gragh import Gragh
from MRR.logger import Logger
from MRR.Simulator import Ring, TransferFunction


class Simulator:
    def __init__(self, is_focus: bool = False) -> None:
        self.logger = Logger()
        self.xs: list[npt.NDArray[np.float64]] = []
        self.ys: list[npt.NDArray[np.float64]] = []
        self.number_of_rings: int = 0
        self.graph = Gragh(is_focus)

    def simulate(self, config: SimulationConfig) -> None:
        Evaluator = build_Evaluator(config)
        mrr = TransferFunction(config.L, config.K, config)
        mrr.print_parameters()
        ring = Ring(config)
        N = ring.calculate_N(config.L)
        FSR = ring.calculate_practical_FSR(N)

        if config.lambda_limit_is_defined():
            x = config.lambda_limit
            y = mrr.simulate(x)
        else:
            x = ring.calculate_x(FSR)
            y = mrr.simulate(x)
            evaluator = Evaluator(x, y)
            result = evaluator.evaluate_band()
            print(result)

        self.logger.save_data_as_csv(x, y, config.name)
        self.graph.plot(x, y)

    def show(self) -> None:
        self.graph.show(self.logger.generate_image_path())
