import numpy as np
from MRR.Simulator import build_TransferFunction, Ring
from MRR.gragh import plot
from MRR.Evaluator import build_Evaluator
from MRR.logger import Logger
from scipy.optimize import differential_evolution

def optimize_func(x, model):
    ratio = np.ceil(x[:model.number_of_rings])
    K = x[model.number_of_rings:]
    N = model.ring.optimize_N(ratio)
    FSR = model.ring.calculate_practical_FSR(N)
    L = model.ring.calculate_ring_length(N)
    if FSR > model.required_FSR * 0.99 and FSR < model.required_FSR * 1.01 and np.all(L < 0.1):
        return 0

    mrr = model.TransferFunction(
        L,
        K
    )
    x = model.ring.calculate_x(FSR)
    y = mrr.simulate(x)
    evaluator = model.Evaluator(
        x,
        y
    )

    return - evaluator.evaluate_band()

class Model:
    """
    Args:
        config (Dict[str, Any]): Configuration of the MRR.
            Keys:
                number_of_episodes_in_L (int): Number of the episodes in L.
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
        number_of_episodes_in_L (int): Number of the episodes in L.
        number_of_rings (int): Number of rings. The ring order.
        required_FSR (float): The required FSR.
        logger (Logger): Logger.
        ring (Ring): The minimum round-trip length.
        Evaluator (Evaluator): Evaluator.
        MRR (MRR): MRR.
    """
    def __init__(
        self,
        config,
        skip_plot
    ):
        self.eta = config['eta']
        self.number_of_episodes_in_L = config['number_of_episodes_in_L']
        self.number_of_rings = config['number_of_rings']
        self.required_FSR = config['FSR']
        self.logger = Logger()
        self.logger.save_config(config)
        self.ring = Ring(config)
        self.Evaluator = build_Evaluator(config)
        self.TransferFunction = build_TransferFunction(config)
        self.skip_plot = skip_plot

    def optimize(self):
        bounds = [
            *[
                (1e-12, 30)
                for _ in range(self.number_of_rings)
            ],
            *[
                (1e-12, self.eta)
                for _ in range(2* self.number_of_rings + 1)
            ]
        ]

        result = differential_evolution(
            optimize_func,
            bounds,
            args=([self]),
            strategy='currenttobest1bin',
            workers=-1,
            updating='deferred',
            popsize=20,
            maxiter=1000
        )
        E = -result.fun
        ratio = result.x[:self.number_of_rings]
        N = self.ring.optimize_N(ratio)
        L = self.ring.calculate_ring_length(ratio)
        FSR = self.ring.calculate_FSR(ratio)
        K = result.x[self.number_of_rings:]

        return L, K, FSR, E

    def train(self):
        L, K, FSR, E = self.optimize()

        print('L  : {}'.format(L.tolist()))
        print('K  : {}'.format(K.tolist()))
        print('FSR: {}'.format(FSR))
        print('E  : {}'.format(E))
        print('================')

        mrr = self.TransferFunction(
            L,
            K
        )
        x = self.ring.calculate_x(FSR)
        y = mrr.simulate(x)
        print('result')
        mrr.print_parameters()
        print('FSR: {}'.format(FSR))
        print('E: {}'.format(E))
        self.logger.save_result(L.tolist(), K.tolist())
        print('end')
        if E > 0 and not self.skip_plot:
            plot([x], [y], self.number_of_rings, self.logger.generate_image_path())
