import numpy as np
import numpy.typing as npt
from scipy.optimize import differential_evolution

from config.model import OptimizationConfig
from MRR.Evaluator.evaluator import Evaluator
from MRR.gragh import Gragh
from MRR.model.base import BaseModel
from MRR.transfer_function import TransferFunction


class Model(BaseModel):
    """
    Args:
        config (Dict[str, Any]): Configuration of the MRR.
            Keys:
                number_of_generations (int): Number of generations.
                center_wavelength (float): The center wavelength.
                number_of_rings (int): Number of rings. The ring order.
                H_p (float): The threshold of the max loss in pass band. loss_p.
                H_s (float): The threshold of the min loss in stop band. loss_s.
                length_of_3db_band (float): The required length of the 3dB band.
                FSR (float): The required FSR.
                alpha (float): The propagation loss coefficient.
                eta (float): The coupling loss coefficient.
                n_eff (float): The equivalent refractive index.
                n_g (float): The group index.
                min_ring_length (float): The minimum round-trip length.
    Attributes:
        eta (float): The coupling loss coefficient.
        number_of_generations (int): Number of generations.
        number_of_rings (int): Number of rings. The ring order.
        required_FSR (float): The required FSR.
        logger (Logger): Logger.
        ring (Ring): The minimum round-trip length.
        Evaluator (Evaluator): Evaluator.
        MRR (MRR): MRR.
    """

    def __init__(self, config: OptimizationConfig, skip_plot: bool = False) -> None:
        super().__init__(config, skip_plot)
        self.eta = config.eta
        self.number_of_generations = config.number_of_episodes_in_L
        self.number_of_rings = config.number_of_rings
        self.required_FSR = config.FSR
        self.rng = config.get_differential_evolution_rng()
        self.strategy = config.strategy

    def optimize_L(self) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float_], float]:
        for i in range(100):
            N = self.ring.optimize_N(self.number_of_rings)
            L = self.ring.calculate_ring_length(N)
            FSR = self.ring.calculate_practical_FSR(N)
            if FSR > self.required_FSR * 0.99 and FSR < self.required_FSR * 1.01 and np.all(L < 0.1):
                break
        if i == 99:
            raise Exception("required_FSR is too strict")

        return N, L, FSR

    def optimize_K(self, L: npt.NDArray[np.float_], FSR: float) -> tuple[npt.NDArray[np.float_], float]:
        bounds = [(1e-12, self.eta) for _ in range(self.number_of_rings + 1)]

        result = differential_evolution(
            optimize_K_func,
            bounds,
            args=(self, L, FSR),
            strategy="currenttobest1bin",
            workers=-1,
            updating="deferred",
            popsize=20,
            maxiter=1,
            seed=self.rng,
        )
        E: float = -result.fun
        K: npt.NDArray[np.float_] = result.x

        return K, E

    def optimize(self) -> None:
        N_list: list[npt.NDArray[np.int_]] = [np.array([]) for _ in range(self.number_of_generations)]
        L_list: list[npt.NDArray[np.float_]] = [np.array([]) for _ in range(self.number_of_generations)]
        K_list: list[npt.NDArray[np.float_]] = [np.array([]) for _ in range(self.number_of_generations)]
        FSR_list: list[float] = [0 for _ in range(self.number_of_generations)]
        E_list: list[float] = [0 for _ in range(self.number_of_generations)]
        method_list: list[int] = [0 for _ in range(self.number_of_generations)]
        best_E_list: list[float] = [0 for _ in range(self.number_of_generations)]
        for m in range(self.number_of_generations):
            N: npt.NDArray[np.int_]
            L: npt.NDArray[np.float_]
            FSR: float

            kind: npt.NDArray[np.int_]
            counts: npt.NDArray[np.int_]

            if m < 10:
                method: int = 4
            else:
                # _, counts = np.unique(best_E_list[:m], return_counts=True)  # type: ignore
                # max_counts: np.int_ = np.max(counts)  # type: ignore
                # if max_counts > 30:
                #     break
                method = self.rng.choice([1, 2, 3, 4], p=self.strategy)

            if method == 1:
                max_index = np.argmax(E_list)
                max_N = N_list[max_index]

                kind, counts = self.rng.permutation(np.unique(max_N, return_counts=True), axis=1)  # type: ignore
                N = np.repeat(kind, counts)
                L = self.ring.calculate_ring_length(N)
                FSR = self.ring.calculate_practical_FSR(N)
            elif method == 2:
                max_index = np.argmax(E_list)
                max_N = N_list[max_index]
                N = self.rng.permutation(max_N)
                L = self.ring.calculate_ring_length(N)
                FSR = self.ring.calculate_practical_FSR(N)
            elif method == 3:
                max_index = np.argmax(E_list)
                max_N = N_list[max_index]
                kind = np.unique(max_N)  # type: ignore
                N = self.rng.choice(kind, self.number_of_rings)
                while not set(kind) == set(N):
                    N = self.rng.choice(kind, self.number_of_rings)
                L = self.ring.calculate_ring_length(N)
                FSR = self.ring.calculate_practical_FSR(N)
            else:
                N, L, FSR = self.optimize_L()
            K, E = self.optimize_K(L, FSR)

            N_list[m] = N
            L_list[m] = L
            FSR_list[m] = FSR
            K_list[m] = K
            E_list[m] = E
            best_index = np.argmax(E_list)
            best_N = N_list[best_index]
            best_L = L_list[best_index]
            best_K = K_list[best_index]
            best_FSR = FSR_list[best_index]
            best_E = E_list[best_index]
            print(m + 1)
            self.logger.print_parameters(K, L, N, FSR, E)
            print("==best==")
            self.logger.print_parameters(best_K, best_L, best_N, best_FSR, best_E)
            print("================")

            method_list[m] = method
            best_E_list[m] = best_E

        max_index = np.argmax(E_list)
        N = N_list[max_index]
        L = L_list[max_index]
        K = K_list[max_index]
        FSR = FSR_list[max_index]
        E = E_list[max_index]
        mrr = TransferFunction(L, K, self.config)
        x = self.ring.calculate_x(FSR)
        y = mrr.simulate(x)
        print("result")
        mrr.print_parameters()
        self.logger.print_parameters(K, L, N, FSR, E)
        self.logger.save_result(L.tolist(), K.tolist())
        self.logger.save_evaluation_value(best_E_list, method_list)
        print("end")
        if E > 0 and not self.skip_plot:
            graph = Gragh()
            graph.create()
            graph.plot(x, y)
            graph.show(self.logger.generate_image_path())


def optimize_K_func(K: npt.NDArray[np.float_], model: Model, L: npt.NDArray[np.float_], FSR: np.float_) -> np.float_:
    mrr = TransferFunction(L, K, model.config)
    x = model.ring.calculate_x(FSR)
    y = mrr.simulate(x)
    evaluator = Evaluator(x, y, model.config)

    return -evaluator.evaluate_band()
