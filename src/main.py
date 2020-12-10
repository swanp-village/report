from MRR.simulator import MRR
from MRR.gragh import plot
from importlib import import_module
import argparse
from MRR.Model.DE import Model
from random import seed
from MRR.Evaluator.Model.train import train_evaluator, show_data
from MRR.Evaluator.evaluator import build_Evaluator
from MRR.ring import Ring
from MRR.logger import Logger


def main(config, skip_plot):
    # seed(1)
    model = Model(config, skip_plot)
    model.train()


def simulate(config_list, skip_plot):
    logger = Logger()
    xs = []
    ys = []

    for config in config_list:
        number_of_rings = len(config['L'])
        config.setdefault('number_of_rings', number_of_rings)
        config.setdefault('FSR', 10e-9)
        config.setdefault('min_ring_length', 10e-9)
        config.setdefault('max_loss_in_pass_band', -10)
        config.setdefault('required_loss_in_stop_band', -20)
        config.setdefault('length_of_3db_band', 1e-9)
        Evaluator = build_Evaluator(config)

        mrr = MRR(
            config['L'],
            config['K'],
            config
        )
        mrr.print_parameters()
        ring = Ring(config)
        N = ring.calculate_N(config['L'])
        FSR = ring.calculate_practical_FSR(N)
        print(FSR)

        if 'lambda' in config:
            x = config['lambda']
        else:
            x = ring.calculate_x(FSR)

        y = mrr.simulate(x)

        evaluator = Evaluator(
            x,
            y
        )
        result = evaluator.evaluate_band()
        print(result)
        logger.save_data_as_csv(x, y, config['name'])
        xs.append(x)
        ys.append(y)
    plot(xs, ys, config['L'].size, logger.generate_image_path(config['name']), skip_plot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='config file path', nargs='*')
    parser.add_argument('--train-evaluator', action='store_true')
    parser.add_argument('--show-data', action='store_true')
    parser.add_argument('--skip-plot', action='store_true')
    args = vars(parser.parse_args())
    skip_plot = args['skip_plot']
    if args['config']:
        try:
            config_list = []
            for c in args['config']:
                config = import_module('config.simulate.{}'.format(c)).config
                config['name'] = c
                config_list.append(config)
        except:
            parser.print_help()
        else:
            simulate(config_list, skip_plot)
    elif args['train_evaluator']:
        train_evaluator()
    elif args['show_data']:
        show_data(skip_plot)
    else:
        try:
            config = import_module('config.base').config
        except:
            parser.print_help()
        else:
            main(config, skip_plot)
