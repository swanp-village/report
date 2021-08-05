from abc import ABC, abstractmethod

from config.model import OptimizationConfig
from MRR.logger import Logger
from MRR.simulator import Ring


class BaseModel(ABC):
    def __init__(self, config: OptimizationConfig, skip_plot: bool = False) -> None:
        self.config = config
        self.logger = Logger()
        self.logger.save_config(config)
        self.ring = Ring(config)
        self.skip_plot = skip_plot

    @abstractmethod
    def optimize(self) -> None:
        pass
