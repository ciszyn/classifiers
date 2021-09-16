from tools.preprocess import pre_process
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from joblib import load
from variables import *
import os
import seaborn as sn
from measure import measure_perf


def test_classifiers():
    y_train, y_val, y_test, X_train, X_val, X_test = pre_process(
        tracks, features_all, list(features.columns.levels[0]))

    scores = {}
    for measure in measures:
        scores[measure] = pd.DataFrame(columns=classifiers_base, index=genres)
    scores['accuracy'] = pd.DataFrame(columns=['all'], index=classifiers_base)

    for name in classifiers_base:
        clf = load('./models/'+set_size+'/'+name+'.joblib')
        disp = plot_confusion_matrix(clf, X_test, y_test,
                                     display_labels=genres,
                                     cmap=plt.cm.Blues, xticks_rotation=45, normalize=None)
        matrix = disp.confusion_matrix

        measure_perf(name, scores, matrix)

        plt.clf()
        df_cm = pd.DataFrame(matrix, index=genres,
                             columns=genres)
        plt.figure(figsize=(10, 7))
        plt.xticks(rotation=45)
        ax = sn.heatmap(df_cm, annot=True, cmap=plt.cm.Blues)
        ax.figure.tight_layout()

        disp.ax_.set_title(name)
        try:
            plt.savefig('./matrices/' + set_size + '/' + name + '.png')
        except:
            os.makedirs("./matrices" + '/' + set_size)
            plt.savefig('./matrices/' + set_size + '/' + name + '.png')
        plt.close()

    for measure in scores:
        try:
            scores[measure].to_csv(
                './results/' + set_size + '/' + measure + '.csv')
        except:
            os.makedirs("./results/" + set_size)
            scores[measure].to_csv(
                './results/' + set_size + '/' + measure + '.csv')


if __name__ == "__main__":
    test_classifiers()
