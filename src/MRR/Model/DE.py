import numpy as np
from random import uniform
from MRR.Simulator import build_TransferFunction, Ring
from MRR.gragh import plot
from MRR.Evaluator import build_Evaluator
from MRR.logger import Logger
from scipy.optimize import differential_evolution
from multiprocessing import Pool


class Model:
    """
    Args:
        config (Dict[str, Any]): Configuration of the MRR.
            Keys:
                number_of_episodes_in_L (int): Number of the episodes in L.
                center_wavelength (float): The center wavelength.
                number_of_rings (int): Number of rings. The ring order.
                max_loss_in_pass_band (float): The threshold of the max loss in pass band. loss_p.
                required_loss_in_stop_band (float): The threshold of the min loss in stop band. loss_s.
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


    def optimize_L(self):
        for i in range(100):
            N = self.ring.init_N()
            L = self.ring.calculate_ring_length(N)
            FSR = self.ring.calculate_practical_FSR(N)
            if FSR > self.required_FSR and FSR < self.required_FSR * 1.05 and np.all(L < 0.1):
                break
        if i == 99:
            raise Exception('required_FSR is too big')

        return L, FSR

    def optimize_K(self, m_L, L, FSR):
        bounds = [
            (0.0000000001, self.eta)
            for _ in range(self.number_of_rings + 1)
        ]

        def func(K):
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

            return - evaluator.evaluate_band()

        result = differential_evolution(func, bounds, strategy='currenttobest1bin')
        E = -result.fun
        K = result.x

        return K, E

    def train(self):
        L_list = [[] for _ in range(self.number_of_episodes_in_L)]
        K_list = [[] for _ in range(self.number_of_episodes_in_L)]
        FSR_list = [0 for _ in range(self.number_of_episodes_in_L)]
        E_list = [0 for _ in range(self.number_of_episodes_in_L)]
        for m_L in range(self.number_of_episodes_in_L):
            L, FSR = self.optimize_L()
            K, E = self.optimize_K(m_L, L, FSR)

            L_list[m_L] = L
            FSR_list[m_L] = FSR
            K_list[m_L] = K
            E_list[m_L] = E
            print(m_L + 1)
            print('L  : {}'.format(L))
            print('K  : {}'.format(K))
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
        print('E: {}'.format(result))
        self.logger.save_result(L.tolist(), K.tolist())
        print('end')
        if result > 0:
            plot([x], [y], self.number_of_rings, self.logger.generate_image_path(), self.skip_plot)
