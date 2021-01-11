from .data import all_data, get_splited_data
from MRR.Evaluator.evaluator import build_Evaluator
import numpy as np
from .config import config
from itertools import product
from scipy.cluster.vq import kmeans, whiten
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
import json
from pathlib import Path
from multiprocessing import Pool

weight_of_binary_evaluation = np.array([0.5])
weight_list = list(product(
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
))
train, test = get_splited_data(0.3)
train_rank_list = np.array([
    d.rank
    for d in train
])
test_rank_list = np.array([
    d.rank
    for d in test
])
def train_data(weight):
    Evaluator = build_Evaluator(config, weight)
    train_evaluate_result = np.array([
        Evaluator(
            d.x,
            d.y
        ).evaluate_band()
        for d in train
    ])
    test_evaluate_result = np.array([
        Evaluator(
            d.x,
            d.y
        ).evaluate_band()
        for d in test
    ])
    train_evaluate_result = np.array(train_evaluate_result).reshape(-1, 1)
    test_evaluate_result = np.array(test_evaluate_result).reshape(-1, 1)
    X_train = train_evaluate_result
    y_train = train_rank_list
    X_test = test_evaluate_result
    y_test = test_rank_list
    clf = SVC()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(score, weight)

    return score


def train_evaluator():
    max_score = 0

    with Pool() as p:
        scores = p.map(train_data, weight_list)

    scores = np.array(scores)
    weights = np.array(weight_list)
    max_score = np.max(scores)
    optimized_weights = weights[scores == max_score]
    print(max_score)
    save_result(max_score, optimized_weights.tolist())


def show_data(skip_plot):
    Evaluator = build_Evaluator(config)
    for data_i in all_data:
        evaluator = Evaluator(
            data_i.x,
            data_i.y
        )
        print(evaluator.evaluate_band())
        data_i.export_gragh(skip_plot)


def save_result(score, weight):
    p = Path('result.json')
    result = {
        'score': score,
        'weights': weight
    }
    src = json.dumps(result, indent=4)
    p.write_text(src)
