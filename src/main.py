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
