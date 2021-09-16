import pandas as pd
from variables import *
import numpy as np


def measure_perf(name, scores, matrix):
    labels = genres
    means = [0.]*4
    for i in range(len(labels)):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for j in range(len(labels)):
            for k in range(len(labels)):
                if j == i and k == i:
                    TP += matrix[j][k]
                elif j != i and k != i:
                    TN += matrix[j][k]
                elif j != i and k == i:
                    FP += matrix[j][k]
                elif j == i and k != i:
                    FN += matrix[j][k]
        try:
            scores['accuracy'].loc[name, 'all'] = np.sum(
                np.diagonal(matrix)) / np.matrix(matrix).sum()
        except:
            scores['accuracy'].loc[name, 'all'] = 0

        try:
            scores['TP_rate'].loc[genres[i], name] = TP / (TP + FN)
            means[0] += TP / (TP + FN)
        except:
            scores['TP_rate'].loc[genres[i], name] = 0
        try:
            scores['FP_rate'].loc[genres[i], name] = FP / (FP + TN)
            means[1] += FP / (FP + TN)
        except:
            scores['FP_rate'].loc[genres[i], name] = 0
        try:
            scores['Precision'].loc[genres[i], name] = TP / (TP + FP)
            means[2] += TP / (TP + FP)
        except:
            scores['Precision'].loc[genres[i], name] = 0
        try:
            scores['F_measure'].loc[genres[i], name] = 2 * \
                TP / (2 * TP + FP + FN)
            means[3] += 2 * TP / (2 * TP + FP + FN)
        except:
            scores['F_measure'].loc[genres[i], name] = 0

    scores['TP_rate'].loc["mean", name] = means[0] / len(genres)
    scores['FP_rate'].loc["mean", name] = means[1] / len(genres)
    scores['Precision'].loc["mean", name] = means[2] / len(genres)
    scores['F_measure'].loc["mean", name] = means[3] / len(genres)
