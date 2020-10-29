from .data import data
from MRR.Evaluator.evaluator import build_Evaluator
import numpy as np
from .config import config
from itertools import product

weight_list = product(
    *[
        np.arange(0, 5, 0.5)
        for _ in range(config['number_of_weights'])
    ]
)

def train():
    for weight in weight_list:
        print('==========================')
        print(weight)
        Evaluator = build_Evaluator(config, weight)

        evaluate_result = [
            Evaluator(
                data_i.x,
                data_i.y
            ).evaluate_band()
            for data_i in data
        ]
        print(evaluate_result)


def plot():
    for data_i in data:
        evaluator = Evaluator(
            data_i.x,
            data_i.y
        )
        print(evaluator.evaluate_band())
        data_i.export_gragh()
