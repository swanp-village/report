import numpy as np
from MRR.Simulator import build_TransferFunction, Ring
from MRR.gragh import plot
from MRR.Evaluator import build_Evaluator
from MRR.logger import Logger
from scipy.optimize import differential_evolution

def optimize_K_func(K, model, L, FSR):
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
        self.rng = np.random.default_rng()


    def optimize_L(self):
        for i in range(100):
            N = self.ring.init_N()
            L = self.ring.calculate_ring_length(N)
            FSR = self.ring.calculate_practical_FSR(N)
            if FSR > self.required_FSR * 0.99 and FSR < self.required_FSR * 1.01 and np.all(L < 0.1):
                break
        if i == 99:
            raise Exception('required_FSR is too strict')

        return N, L, FSR

    def optimize_K(self, L, FSR):
        bounds = [
            (1e-12, self.eta)
            for _ in range(self.number_of_rings + 1)
        ]

        result = differential_evolution(
            optimize_K_func,
            bounds,
            args=(self, L, FSR),
            strategy='currenttobest1bin',
            workers=-1,
            updating='deferred',
            popsize=20,
            maxiter=1000
        )
        E = -result.fun
        K = result.x

        return K, E

    def train(self):
        N_list = [[] for _ in range(self.number_of_episodes_in_L)]
        L_list = [[] for _ in range(self.number_of_episodes_in_L)]
        K_list = [[] for _ in range(self.number_of_episodes_in_L)]
        FSR_list = [0 for _ in range(self.number_of_episodes_in_L)]
        E_list = [0 for _ in range(self.number_of_episodes_in_L)]
        for m in range(self.number_of_episodes_in_L):
            if m > 20 and self.rng.choice([True, False], p=[0.2, 0.8]):
                max_index = np.argmax(E_list)
                N = self.rng.permutation(N_list[max_index])
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
            print(m + 1)
            print('L  : {}'.format(L.tolist()))
            print('K  : {}'.format(K.tolist()))
            print('FSR: {}'.format(FSR))
            print('E  : {}'.format(E))
            print('================')

        max_index = np.argmax(E_list)
        L = L_list[max_index]
        K = K_list[max_index]
        FSR = FSR_list[max_index]
        mrr = self.TransferFunction(
            L,
            K
        )
        x = self.ring.calculate_x(FSR)
        y = mrr.simulate(x)
        evaluator = self.Evaluator(
            x,
            y
        )
        result = evaluator.evaluate_band()
        print('result')
        mrr.print_parameters()
        print('FSR: {}'.format(FSR))
        print('E: {}'.format(result))
        self.logger.save_result(L.tolist(), K.tolist())
        print('end')
        if result > 0 and not self.skip_plot:
            plot([x], [y], self.number_of_rings, self.logger.generate_image_path())
