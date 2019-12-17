from MRR.simulator import MRR
from MRR.gragh import plot
from importlib import import_module
import argparse
from MRR.reward import evaluate_pass_band
from random import randrange
from MRR.ring import calculate_x, calculate_practical_FSR, find_ring_length, init_K


def main(config):
    center_wavelength = config['center_wavelength']
    number_of_rings = config['number_of_rings']
    max_loss_in_pass_band = config['max_loss_in_pass_band']
    n = config['n']
    _, ring_length_list, FSR_list = find_ring_length(center_wavelength, n)
    index = [200, 202]
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
    if result:
        mrr.print_parameters()
        title = '{} order MRR'.format(L.size)
        plot(x, y, title)

def train(config):
    center_wavelength = config['center_wavelength']
    number_of_rings = config['number_of_rings']
    max_loss_in_pass_band = config['max_loss_in_pass_band']
    n = config['n']
    _, ring_length_list, FSR_list = find_ring_length(center_wavelength, n)
    for _ in range(1000):
        index = [randrange(0, 999), randrange(0, 999)]
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
        if result:
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
            main(config)
