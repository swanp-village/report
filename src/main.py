import argparse
from importlib import import_module
from MRR.Evaluator.Model.train import train_evaluator, show_data
from MRR.Controller import simulate, optimize
from config.base import config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='config file path', nargs='*')
    parser.add_argument('--train-evaluator', action='store_true')
    parser.add_argument('--show-data', action='store_true')
    parser.add_argument('--skip-plot', action='store_true')
    parser.add_argument('-f', '--focus', action='store_true')
    args = vars(parser.parse_args())
    skip_plot = args['skip_plot']
    is_focus = args['focus']
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
            simulate(config_list, skip_plot, is_focus)
    elif args['train_evaluator']:
        train_evaluator()
    elif args['show_data']:
        show_data(skip_plot)
    else:
        optimize(config, skip_plot)
