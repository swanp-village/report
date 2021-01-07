import numpy as np
from random import uniform
from MRR.Simulator import build_TransferFunction, Ring
from MRR.gragh import plot
from MRR.Evaluator import build_Evaluator
from MRR.logger import Logger
from scipy.optimize import differential_evolution


class Model:
    """
    Args:
        config (Dict[str, Any]): Configuration of the MRR.
            Keys:
                number_of_episodes_in_L (int): Number of the episodes in L.
                number_of_episodes_in_K (int): Number of the episodes in K.
                number_of_steps (int): Number of steps.
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
        number_of_episodes_in_K (int): Number of the episodes in K.
        number_of_steps (int): Number of steps.
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
        self.number_of_episodes_in_K = config['number_of_episodes_in_K']
        self.number_of_steps = config['number_of_steps']
        self.number_of_rings = config['number_of_rings']
        self.required_FSR = config['FSR']
        self.logger = Logger()
        self.logger.save_config(config)
        self.ring = Ring(config)
        self.Evaluator = build_Evaluator(config)
        self.TransferFunction = build_TransferFunction(config)
        self.skip_plot = skip_plot
        self.L_list = []
        self.K_list = []
        self.Q_list = []

    def optimize_L(self):
        for i in range(100):
            N = self.ring.init_N()
            L = self.ring.calculate_ring_length(N)
            FSR = self.ring.calculate_practical_FSR(N)
            if FSR > self.required_FSR:
                break
        if i == 99:
            raise Exception('required_FSR is too big')

        self.L = L
        self.FSR = FSR

    def optimize_K(self, m_L):
        bounds = [
            (0.0000000001, self.eta)
            for _ in range(self.number_of_rings + 1)
        ]

        def func(K):
            mrr = self.TransferFunction(
                self.L,
                K
            )
            x = self.ring.calculate_x(self.FSR)
            y = mrr.simulate(x)
            evaluator = self.Evaluator(
                x,
                y
            )

            return - evaluator.evaluate_band()

        result = differential_evolution(func, bounds, strategy='currenttobest1bin')
        print(self.L, result.x, -result.fun)

        self.K_list[m_L].append(result.x.tolist())
        self.Q_list[m_L].append(-result.fun)

    def train(self):
        for m_L in range(self.number_of_episodes_in_L):
            print(m_L + 1)
            self.optimize_L()

            self.K_list.append([])
            self.Q_list.append([])

            self.optimize_K(m_L)
            self.L_list.append(self.L)
            max_index = np.argmax(self.Q_list[m_L])
            self.Q_list[m_L] = self.Q_list[m_L][max_index]
            self.K_list[m_L] = self.K_list[m_L][max_index]

        L = self.L_list[np.argmax(self.Q_list)]
        K = self.K_list[np.argmax(self.Q_list)]
        mrr = self.TransferFunction(
            L,
            K
        )
        x = self.ring.calculate_x(self.FSR)
        y = mrr.simulate(x)
        evaluator = self.Evaluator(
            x,
            y
        )
        result = evaluator.evaluate_band()
        print(result)
        print('end')
        if result > 0:
            mrr.print_parameters()
            plot([x], [y], L.size, self.logger.generate_image_path(), self.skip_plot)
