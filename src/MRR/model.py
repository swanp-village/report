import numpy as np
from MRR.simulator import MRR
from MRR.gragh import plot
from MRR.reward import Reward, generate_action
from MRR.ring import (
    calculate_x,
    calculate_practical_FSR,
    calculate_ring_length,
    init_K,
    init_N,
    calculate_FSR,
    calculate_min_N
)
from multiprocessing import Pool


class Model:
    def __init__(
        self,
        config
    ):
        self.number_of_episodes_in_L = config['number_of_episodes_in_L']
        self.number_of_episodes_in_K = config['number_of_episodes_in_K']
        self.number_of_steps = config['number_of_steps']
        self.center_wavelength = config['center_wavelength']
        self.number_of_rings = config['number_of_rings']
        self.min_ring_length = config['min_ring_length']
        self.max_loss_in_pass_band = config['max_loss_in_pass_band']
        self.required_loss_in_stop_band = config['required_loss_in_stop_band']
        self.length_of_3db_band = config['length_of_3db_band']
        self.required_FSR = config['FSR']
        self.alpha = config['alpha']
        self.n = config['n']
        self.eta = config['eta']
        self.min_N = calculate_min_N(
            self.min_ring_length,
            self.center_wavelength,
            self.n
        )
        self.L_list = []
        self.K_list = []
        self.Q_list = []

    def train(self):
        for m_L in range(self.number_of_episodes_in_L):
            for i in range(100):
                N = init_N(
                    self.number_of_rings,
                    self.required_FSR,
                    self.center_wavelength,
                    self.min_N
                )
                L = calculate_ring_length(N, self.center_wavelength, self.n)
                FSR_list = calculate_FSR(N, self.center_wavelength)
                FSR = calculate_practical_FSR(FSR_list)
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
                self.K = init_K(self.number_of_rings)
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
        mrr = MRR(
            self.eta,
            self.n,
            self.alpha,
            self.K_list[np.argmax(self.Q_list)],
            self.L_list[np.argmax(self.Q_list)]
        )
        x = calculate_x(self.center_wavelength, FSR)
        y = mrr.simulate(x)
        reward = Reward(
            x,
            y,
            self.center_wavelength,
            self.number_of_rings,
            self.max_loss_in_pass_band,
            self.required_loss_in_stop_band
            self.length_of_3db_band
        )
        result = reward.evaluate_band()
        if result > 0:
            print(result)
            mrr.print_parameters()
            plot(x, y, L.size)

    def calc_Q(self, a):
        if np.all(np.where((self.K + a > 0) & (self.K + a < 1), True, False)):
            mrr = MRR(
                self.eta,
                self.n,
                self.alpha,
                self.K + a,
                self.L
            )
            x = calculate_x(self.center_wavelength, self.FSR)
            y = mrr.simulate(x)
            reward = Reward(
                x,
                y,
                self.center_wavelength,
                self.number_of_rings,
                self.max_loss_in_pass_band,
                self.required_loss_in_stop_band
                self.required_loss_in_stop_band,
            )
            result = reward.evaluate_band()
        else:
            result = 0
        return result
