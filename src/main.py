from MRR.simulator import MRR
from MRR.gragh import plot
from importlib import import_module
import argparse
from MRR.model import Model
from random import seed
from MRR.ring import calculate_N, calculate_practical_FSR, calculate_FSR, calculate_x
import csv


def main(config):
    seed(1)
    model = Model(config)
    model.train()


def simulate(config):
    mrr = MRR(
        config['eta'],
        config['n'],
        config['alpha'],
        config['K'],
        config['L']
    )
    mrr.print_parameters()
    if 'lambda' in config:
        x = config['lambda']
    else:
        N = calculate_N(config['L'], config['center_wavelength'], config['n'])
        FSR = calculate_practical_FSR(
            calculate_FSR(N, config['center_wavelength'])
        )
        x = calculate_x(config['center_wavelength'], FSR)
    y = mrr.simulate(x)
    plot(x, y, config['L'].size)
    with open('img/out.tsv', 'w') as tsvfile:
        tsv_writer = csv.writer(tsvfile, delimiter='\t')
        tsv_writer.writerows(zip(x.tolist(), y.tolist()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='config file path')
    args = vars(parser.parse_args())
    if args['config']:
        try:
            config = import_module(
                'config.simulate.{}'.format(args['config'])
            ).config
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
