import matplotlib.pyplot as plt
import numpy as np
from MRR import MRR
from importlib import import_module
import argparse


def main(config):
    mrr = MRR(
        config['eta'],
        config['n'],
        config['alpha'],
        config['K'],
        config['L']
    )
    y = 20 * np.log10(np.abs(mrr.simulate(config['lambda'])))
    plt.semilogx(config['lambda'] * 1e9, y)
    plt.xlabel('Wavelength[nm]')
    plt.ylabel('Drop Port Power [dB]')
    plt.title('{} order MRR'.format(config['L'].size))
    plt.axis([None, None, None, 5])
    plt.grid()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='config file path')
    args = vars(parser.parse_args())
    try:
        config = import_module('config.{}'.format(args['config'])).config
    except:
        parser.print_help()
    else:
        main(config)
