import os

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import confusion_matrix, accuracy_score
from statsmodels.stats.contingency_tables import mcnemar

"""
This file is used for significance tests.

"""

full_name = {'energy': 'energy', 'non': 'non-energy', 'agriculture': 'agriculture', 'beverages': 'beverages',
             'food': 'food', 'raw': 'raw-materials', 'metals': 'metals-minerals', 'precious': 'precious-metals'}


def pesarmann(actual, pred):
    size = actual.shape[0]
    print(accuracy_score(actual, pred))

    pyz = np.sum(np.sign(actual) == np.sign(pred)) / size
    py = np.sum(actual > 0) / size
    qy = py * (1 - py) / size
    pz = np.sum(pred > 0) / size
    qz = pz * (1 - pz) / size
    p = py * pz + (1 - py) * (1 - pz)
    v = p * (1 - p) / size
    w = ((2 * py - 1) ** 2) * qz + ((2 * pz - 1) ** 2) * qy + 4 * qy * qz
    pt = (pyz - p) / (np.sqrt(v - w))
    pval = 1 - norm.cdf(pt, 0, 1)
    return pyz, pt, pval


def mcnem(model1_predictions, model2_predictions, actual):
    model1_predictions.index = actual.index
    model2_predictions.index = actual.index

    model1_correct = model1_predictions == actual
    model2_correct = model2_predictions == actual
    model1_correct = model1_correct.stack().reset_index(drop=True)
    model2_correct = model2_correct.stack().reset_index(drop=True)
    actual = actual.stack().reset_index(drop=True)

    table = confusion_matrix(model1_correct, model2_correct)

    result = mcnemar(table)

    print(f'statistic={result.statistic}, p-value={result.pvalue}')
    return result.statistic


def confusion():
    for file in os.listdir('data/output/measures/overall/aggregated-preds'):
        preds = pd.read_csv(f'data/output/measures/overall/aggregated-preds/{file}', index_col=0)
        if 'OLS' in file:
            preds = preds.applymap(lambda x: 1 if x > 0 else 0)
        preds = preds.stack().reset_index(drop=True)
        print(file)
        print(confusion_matrix(returns, preds))
