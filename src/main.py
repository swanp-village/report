from MRR.simulator import MRR
from MRR.gragh import plot
from importlib import import_module
import argparse
import numpy as np
from MRR.reward import Reward, init_action
from MRR.ring import calculate_x, calculate_practical_FSR, init_K, init_N, calculate_ring_length, calculate_FSR, calculate_min_N
from MRR.model import Model
from random import seed


def main(config):
    seed(1)
    model = Model(config)
    model.train()


def train(config):
    number_of_episodes_in_L = config['number_of_episodes_in_L']
    number_of_episodes_in_K = config['number_of_episodes_in_K']
    number_of_steps = config['number_of_steps']
    center_wavelength = config['center_wavelength']
    number_of_rings = config['number_of_rings']
    min_ring_length = config['min_ring_length']
    max_loss_in_pass_band = config['max_loss_in_pass_band']
    required_loss_in_stop_band = config['required_loss_in_stop_band']
    required_FSR = config['FSR']
    n = config['n']
    min_N = calculate_min_N(min_ring_length, center_wavelength, n)

    L_list = []
    K_list = []
    Q_list = []

    for m_L in range(number_of_episodes_in_L):
        for i in range(100):
            N = init_N(number_of_rings, required_FSR, center_wavelength, min_N)
            L = calculate_ring_length(N, center_wavelength, n)
            FSR_list = calculate_FSR(N, center_wavelength)
            FSR = calculate_practical_FSR(FSR_list)
            if FSR > required_FSR:
                break
        if i == 99:
            print('required_FSR is too big')
            return

        _K_list = []
        _Q_list = []

        for m_K in range(number_of_episodes_in_K):
            print('episode {}-{}'.format(m_L + 1, m_K + 1))
            K = init_K(number_of_rings)
            action = init_action(number_of_rings)

            for t in range(number_of_steps):
                print('step {}'.format(t + 1))
                print(K)
                Q = []
                for a in action:
                    if np.all(np.where((K + a > 0) & (K + a < 1), True, False)):
                        mrr = MRR(
                            config['eta'],
                            n,
                            config['alpha'],
                            K + a,
                            L
                        )
                        x = calculate_x(center_wavelength, FSR)
                        y = mrr.simulate(x)

                        reward = Reward(
                            x,
                            y,
                            center_wavelength,
                            number_of_rings,
                            max_loss_in_pass_band,
                            required_loss_in_stop_band
                        )
                        result = reward.evaluate_band()
                        print(result)
                    else:
                        result = 0
                    Q.append(result)
                if np.max(Q) <= 0 or np.argmax(Q) == 0:
                    break
                else:
                    K = K + action[np.argmax(Q)]
            _K_list.append(K.tolist())
            _Q_list.append(Q[np.argmax(Q)])
        L_list.append(L)
        Q_list.append(_Q_list[np.argmax(_Q_list)])
        K_list.append(_K_list[np.argmax(_Q_list)])
    mrr = MRR(
        config['eta'],
        n,
        config['alpha'],
        K_list[np.argmax(Q_list)],
        L_list[np.argmax(Q_list)]
    )
    x = calculate_x(center_wavelength, FSR)
    y = mrr.simulate(x)
    reward = Reward(
        x,
        y,
        center_wavelength,
        number_of_rings,
        max_loss_in_pass_band,
        required_loss_in_stop_band
    )
    result = reward.evaluate_band()
    if result > 0:
        print(result)
        mrr.print_parameters()
        plot(x, y, L.size)


def simulate(config):
    mrr = MRR(
        config['eta'],
        config['n'],
        config['alpha'],
        config['K'],
        config['L']
    )
    mrr.print_parameters()
    x = config['lambda']
    y = mrr.simulate(x)
    plot(x, y, config['L'].size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='config file path')
    args = vars(parser.parse_args())
    if args['config']:
        try:
            config = import_module('config.simulate.{}'.format(args['config'])).config
        except:
            parser.print_help()
        else:
            simulate(config)
    else:
        try:
            config = import_module('config.base').config
        except:
            parser.print_help()
        else:
            main(config)
