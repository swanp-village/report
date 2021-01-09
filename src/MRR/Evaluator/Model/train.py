from .data import data
from MRR.Evaluator.evaluator import build_Evaluator
import numpy as np
from .config import config
from itertools import product
from scipy.cluster.vq import kmeans, whiten
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, cross_val_score
from multiprocessing import Pool

weight_of_binary_evaluation = np.array([0.5])
weight_list = product(
    *[
        np.arange(1, 6, 0.5),
        np.arange(1, 6, 0.5),
        np.arange(1, 6, 0.5),
        np.arange(1, 6, 0.5),
        np.arange(1, 6, 0.5),
        np.array([1]),
        weight_of_binary_evaluation,
        weight_of_binary_evaluation,
        weight_of_binary_evaluation,
        weight_of_binary_evaluation,
        weight_of_binary_evaluation,
        weight_of_binary_evaluation
    ]
)
rank_list = np.array([
    d.rank
    for d in data
])


def train_evaluator():
    max_score = 0
    optimized_weight = []
    for weight in weight_list:
        print('==========================')
        print(weight)

        Evaluator = build_Evaluator(config)
        evaluate_result = np.array([
            Evaluator(
                data_i.x,
                data_i.y
            ).evaluate_band()
            for data_i in data
        ])
        print(evaluate_result)
        evaluate_result = np.array(evaluate_result).reshape(-1, 1)
        X = evaluate_result
        y = rank_list
        clf = SVC()
        scores = cross_val_score(clf, X, y, cv=4, scoring='f1_micro', n_jobs=-1)
        average_score = scores.mean()
        print(scores.mean(), scores)

        if average_score > max_score:
            max_score = average_score
            optimized_weight = weight

    print(max_score, optimized_weight)


def show_data(skip_plot):
    Evaluator = build_Evaluator(config)
    for data_i in data:
        evaluator = Evaluator(
            data_i.x,
            data_i.y
        )
        print(evaluator.evaluate_band())
        data_i.export_gragh(skip_plot)
