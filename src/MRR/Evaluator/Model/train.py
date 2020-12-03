from .data import data
from MRR.Evaluator.evaluator import build_Evaluator
import numpy as np
from .config import config
from itertools import product
from scipy.cluster.vq import kmeans, whiten
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut

weight_list = product(
    *[
        np.arange(1, 6, 1)
        for _ in range(config['number_of_weights'])
    ]
)
rank_list = np.array([
    d.rank
    for d in data
])

def train():
    loo = LeaveOneOut()
    max_score = 0
    optimized_weight = []
    for weight in weight_list:
        print('==========================')
        print(weight)
        Evaluator = build_Evaluator(config, weight)

        evaluate_result = np.array([
            Evaluator(
                data_i.x,
                data_i.y
            ).evaluate_band()
            for data_i in data
        ])
        print(evaluate_result)
        score_list = []
        for train_index, test_index in loo.split(evaluate_result):
            evaluate_result = np.array(evaluate_result).reshape(-1, 1)
            clf = SVC()
            clf.fit(evaluate_result[train_index], rank_list[train_index])
            score = clf.score(evaluate_result[test_index], rank_list[test_index])
            score_list.append(score)
        average_score = np.average(score_list)
        print(average_score)
        print(score_list)

        if average_score > max_score:
            max_score = average_score
            optimized_weight = weight

    print(max_score, optimized_weight)


def _train():
    Evaluator = build_Evaluator(config)
    for data_i in data:
        evaluator = Evaluator(
            data_i.x,
            data_i.y
        )
        print(evaluator.evaluate_band())
        data_i.export_gragh()
