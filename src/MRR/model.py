import numpy as np
from MRR.simulator import build_MRR
from MRR.gragh import plot
from MRR.evaluator import build_Evaluator, generate_action
from MRR.ring import Ring
from multiprocessing import Pool
from MRR.logger import Logger


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
                n_eq (float): The equivalent refractive index.
                n_eff (float): The effective refractive index.
                min_ring_length (float): The minimum round-trip length.
    Attributes:
        logger (Logger): Logger.
        number_of_episodes_in_L (int): Number of the episodes in L.
        number_of_episodes_in_K (int): Number of the episodes in K.
        number_of_steps (int): Number of steps.
        number_of_rings (int): Number of rings. The ring order.
        required_FSR (float): The required FSR.
        ring (Ring): The minimum round-trip length.
    """
    def __init__(
        self,
        config
    ):
        self.logger = Logger()
        self.MRR = build_MRR(config)
        self.Evaluator = build_Evaluator(config)
        self.logger.save_config(config)
        self.number_of_episodes_in_L = config['number_of_episodes_in_L']
        self.number_of_episodes_in_K = config['number_of_episodes_in_K']
        self.number_of_steps = config['number_of_steps']
        self.number_of_rings = config['number_of_rings']
        self.required_FSR = config['FSR']
        self.ring = Ring(config)
        self.L_list = []
        self.K_list = []
        self.Q_list = []

    def train(self):
        for m_L in range(self.number_of_episodes_in_L):
            for i in range(100):
                N = self.ring.init_N(
                    self.required_FSR
                )
                L = self.ring.calculate_ring_length(N)
                FSR = self.ring.calculate_practical_FSR(N)
                if FSR > self.required_FSR:
                    break
            if i == 99:
                print('required_FSR is too big')
                return

            self.L = L
            self.FSR = FSR

            _K_list = []
            _Q_list = []

            for m_K in range(self.number_of_episodes_in_K):
                print('episode {}-{}'.format(m_L + 1, m_K + 1))
                self.K = self.ring.init_K(self.number_of_rings)
                action = generate_action(self.number_of_rings)

                for t in range(self.number_of_steps):
                    print('step {}'.format(t + 1))
                    print(self.K)
                    with Pool(10) as p:
                        Q = p.map(self.calc_Q, action)
                    print(np.max(Q))
                    if np.max(Q) <= 0 or np.argmax(Q) == 0:
                        break
                    else:
                        self.K = self.K + action[np.argmax(Q)]
                _K_list.append(self.K.tolist())
                _Q_list.append(np.max(Q))
            self.L_list.append(L)
            self.Q_list.append(np.max(_Q_list))
            self.K_list.append(_K_list[np.argmax(_Q_list)])
        mrr = self.MRR(
            self.L_list[np.argmax(self.Q_list)],
            self.K_list[np.argmax(self.Q_list)]
        )
        x = self.ring.calculate_x(FSR)
        y = mrr.simulate(x)
        evaluator = self.Evaluator(
            x,
            y
        )
        result = evaluator.evaluate_band()
        if result > 0:
            print(result)
            mrr.print_parameters()
            plot(x, y, L.size, self.logger.generate_image_path())

    def calc_Q(self, a):
        if np.all(np.where((self.K + a > 0) & (self.K + a < 1), True, False)):
            mrr = self.MRR(
                self.L,
                self.K + a
            )
            x = self.ring.calculate_x(self.FSR)
            y = mrr.simulate(x)
            evaluator = self.Evaluator(
                x,
                y
            )
            result = evaluator.evaluate_band()
        else:
            result = 0
        return result
