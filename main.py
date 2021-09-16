import base_classif
import create_spec_full
import create_spec
import evaluate_CNN
import evaluate_CNN_full
import performance
import train_CNN_full
import train_CNN
import train_PRCNN
from variables import *
from plot_history import plot
import pandas as pd

# base_classif.create_classifiers(classifiers, feature_sets)
# performance.test_classifiers()

# create_spec.create_spectrograms()
# create_spec_full.create_spectrograms_full()

# train_CNN.train()
# train_PRCNN.train()
# train_CNN_full.train()

#evaluate_CNN.evaluate('cnn')
#evaluate_CNN.evaluate('prcnn')
# evaluate_CNN.evaluate('crnn2')
# evaluate_CNN.evaluate('crnn3')
#evaluate_CNN_full.evaluate()

# plot('cnn')
# plot('prcnn')
# plot('crnn')
# plot('crnn2')
# plot('cnn_full')

measures = ["TP_rate", "FP_rate", "Precision", "F_measure"]
scores = {}
for measure in measures:
    scores[measure] = pd.read_csv('./results/' + set_size + '/' + measure + '.csv', index_col=0)

for measure in measures:
    for classifier in list(scores[measure].columns):
        mean = 0.
        for genre in genres:
            mean += scores[measure].loc[genre, classifier]
        mean /= len(genres)
        scores[measure].loc["mean", classifier] = mean
        print(mean)

for measure in measures:
        scores[measure].to_csv('./results/' + set_size + '/' + measure + '.csv')
    
