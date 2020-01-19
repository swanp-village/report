from MRR.simulator import MRR
from MRR.gragh import plot
from importlib import import_module
import argparse
import numpy as np
from MRR.reward import evaluate_pass_band, evaluate_ring_combination, init_action
from MRR.ring import calculate_x, calculate_practical_FSR, find_ring_length, init_K, init_N, calculate_ring_length, calculate_FSR


def main(config):
    center_wavelength = config['center_wavelength']
    number_of_rings = config['number_of_rings']
    max_loss_in_pass_band = config['max_loss_in_pass_band']
    max_N = config['max_N']
    n = config['n']
    ring_length_list, FSR_list = find_ring_length(center_wavelength, n, max_N)
    index = [39, 831]
    L = ring_length_list[index]
    FSR = calculate_practical_FSR(FSR_list[index])
    K = init_K(number_of_rings)
    mrr = MRR(
        config['eta'],
        n,
        config['alpha'],
        K,
        L
    )
    x = calculate_x(center_wavelength, FSR)
    y = mrr.simulate(x)
    result = evaluate_pass_band(x, y, center_wavelength, max_loss_in_pass_band)
    # if result:
    mrr.print_parameters()
    title = '{} order MRR'.format(L.size)
    plot(x, y, title)


def train(config):
    number_of_episodes = config['number_of_episodes']
    number_of_steps = config['number_of_steps']
    center_wavelength = config['center_wavelength']
    number_of_rings = config['number_of_rings']
    max_loss_in_pass_band = config['max_loss_in_pass_band']
    required_FSR = config['FSR']
    n = config['n']

    N = init_N(number_of_rings, required_FSR, center_wavelength)
    L = calculate_ring_length(N, center_wavelength, n)
    FSR_list = calculate_FSR(N, center_wavelength)
    FSR = calculate_practical_FSR(FSR_list)
    if not evaluate_ring_combination(L, FSR_list, FSR):
        return

    for m in range(number_of_episodes):
        print('episode {}'.format(m + 1))
        K = init_K(number_of_rings)
        action = init_action(number_of_rings)

        for t in range(number_of_steps):
            print('step {}'.format(t + 1))
            print(K)
            q = []
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
                    result = evaluate_pass_band(x, y, center_wavelength, max_loss_in_pass_band)
                    if result > 0:
                        print(result)
                        # mrr.print_parameters()
                        # title = '{} order MRR'.format(L.size)
                        # plot(x, y, title)
                else:
                    result = 0
                q.append(result)
            if np.max(q) == 0 or np.argmax(q) == 0:
                break
            else:
                K = K + action[np.argmax(q)]
        mrr = MRR(
            config['eta'],
            n,
            config['alpha'],
            K,
            L
        )
        x = calculate_x(center_wavelength, FSR)
        y = mrr.simulate(x)
        mrr.print_parameters()
        title = '{} order MRR'.format(L.size)
        plot(x, y, title)


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
    result = evaluate_pass_band(x, y, 1550e-9, -5)
    title = '{} order MRR'.format(config['L'].size)
    plot(x, y, title)


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
            train(config)
