from MRR.simulator import MRR
from MRR.gragh import plot
from importlib import import_module
import argparse
from MRR.reward import calculate_pass_band
from MRR.ring import calculate_practical_FSR, find_ring_length, init_K
import numpy as np


def main(config):
    center_wavelength = config['center_wavelength']
    number_of_rings = config['number_of_rings']
    max_loss_in_pass_band = config['max_loss_in_pass_band']
    n = config['n']
    _, ring_length_list, FSR_list = find_ring_length(center_wavelength, n)
    index = [513, 747, 953]
    L = np.array(ring_length_list[index])
    FSR = calculate_practical_FSR(FSR_list[index])
    print(FSR)
    K = init_K(number_of_rings)
    mrr = MRR(
        config['eta'],
        n,
        config['alpha'],
        K,
        L
    )
    mrr.print_parameters()
    half_FSR = np.ceil(FSR * 4 / 1e-12) * 1e-12
    x = np.arange(center_wavelength - half_FSR, center_wavelength + half_FSR, 1e-12)
    y = mrr.simulate(x)
    print(calculate_pass_band(x, y, max_loss_in_pass_band))
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
            main(config)
