from MRR.simulator import MRR
from MRR.gragh import plot
from importlib import import_module
import argparse
from MRR.reward import shape_factor, a
from MRR.ring import find_ring_length
from util.math import lcm, gcd


def main(config):
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
    minid, _ = a(x, y)
    print(x[minid])
    print(y[minid])
    N, ring_length_list, FSR_list = find_ring_length(1550e-9, config['n'])
    # shape_factor(x, y)
    print(gcd(0.5, 0.2))
    # plot(x, y, title)


    # maxid = argrelmax(y, order=10)[0]
    # print(y[maxid])
    # plt.plot(x[maxid] * 1e9, y[maxid], "ro")
    # sorted_maxid_index = y[maxid].argsort()[::-1]
    # loss1_id = maxid[sorted_maxid_index[0]]
    # plt.plot(x[loss1_id] * 1e9, y[loss1_id], 'bo')
    # loss2_id = maxid[sorted_maxid_index[1]]
    # plt.plot(x[loss2_id] * 1e9, y[loss2_id], 'go')


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
