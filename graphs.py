import matplotlib.pyplot as plt
import pandas as pd
from variables import *

measure = 'accuracy'

scores = pd.read_csv("./results/" + set_size + '/' + measure + '.csv', index_col=0)

plt.bar(list(scores.index), list(scores['all']))
plt.xticks(rotation=90)
plt.savefig("./figures/" + set_size + '/' + measure + '.png', bbox_inches = "tight")

measures = ['TP_rate', 'FP_rate', 'Precision', 'F_measure']

for measure in measures:
    scores = pd.read_csv("./results/" + set_size + '/' + measure + '.csv', index_col=0)
    for genre in genres:
        plt.clf()
        plt.bar(list(scores.columns), list(scores.loc[genre]))
        plt.xticks(rotation=90)
        plt.savefig("./figures/" + set_size + '/' + measure + '/' + genre +'.png', bbox_inches = "tight")
