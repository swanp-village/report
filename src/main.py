from MRR.simulator import MRR
from MRR.gragh import plot
from importlib import import_module
import argparse
from MRR.model import Model
from random import seed
from MRR.reward import Reward
from MRR.ring import (
    calculate_N,
    calculate_practical_FSR,
    calculate_FSR,
    calculate_x
)
from MRR.logger import Logger


def main(config):
    seed(1)
    model = Model(config)
    model.train()


def simulate(config):
    logger = Logger()
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
    max_loss_in_pass_band = config.get('max_loss_in_pass_band', -10)
    required_loss_in_stop_band = config.get('required_loss_in_stop_band', -20)
    length_of_3db_band = config.get('length_of_3db_band', 1e-9)
    center_wavelength = config.get('center_wavelength', 1550e-9)
    reward = Reward(
        x,
        y,
        center_wavelength,
        len(config['L']),
        max_loss_in_pass_band,
        required_loss_in_stop_band,
        length_of_3db_band
    )
    result = reward.evaluate_band()
    print(result)
    plot(x, y, config['L'].size, logger.generate_image_path())
    logger.save_data_as_csv(x, y)

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
