from MRR.simulator import MRR
from MRR.gragh import plot
from importlib import import_module
import argparse
from MRR.ring import calculate_practical_FSR, find_ring_length, init_K
import numpy as np


def main(config):
    center_wavelength = config['center_wavelength']
    number_of_rings = config['number_of_rings']
    n = config['n']
    _, ring_length_list, FSR_list = find_ring_length(center_wavelength, n)
    L = np.array([ring_length_list[300], ring_length_list[500], ring_length_list[700]])
    FSR = calculate_practical_FSR([FSR_list[300], FSR_list[500], FSR_list[700]])
    K = init_K(number_of_rings)
    mrr = MRR(
        config['eta'],
        n,
        config['alpha'],
        K,
        L
    )
    mrr.print_parameters()
    x = np.arange(center_wavelength - FSR / 2, center_wavelength + FSR / 2, 1e-12)
    y = mrr.simulate(x)
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
