from joblib import load
import pandas as pd
from math import sqrt
from scipy.stats import t
import statistics
from plot_histograms import plot
from votingConfig import *
from variables import set_size

def votingSummary():
    resultDf = pd.DataFrame()
    for optimizationMethod in optimizationMethods:
        for votingMethod in votingMethods:
            try:
                df = pd.read_csv('./votingVars/' + votingMethod + '_' + optimizationMethod +'.csv')
                resultDf.loc[optimizationMethod + '_' + votingMethod, 'mean'] = df.loc[0, 'mean']
                resultDf.loc[optimizationMethod + '_' + votingMethod, 'variance'] = df.loc[0, 'variance']
                resultDf.loc[optimizationMethod + '_' + votingMethod, 'p_value'] = df.loc[0, 'p_value']
                resultDf.loc[optimizationMethod + '_' + votingMethod, 'max_accuracy'] = df.loc[0, 'max_accuracy']
                resultDf.loc[optimizationMethod + '_' + votingMethod, 'max_increment'] = df.loc[0, 'max_increment']
            except:
                continue
    resultDf.to_csv('./results/' + set_size + '/voting.csv')

def extractVoting():
    for optimizationMethod in optimizationMethods:
        for votingMethod in votingMethods:
            try:
                dict = load('./votingVars/' + votingMethod + '_' + optimizationMethod +'.joblib')
                df = pd.DataFrame(columns=["commity", "val_acc_1", "val_acc_2", "test_acc_1", "test_acc_2"])
                for key in dict:
                    df.loc[len(df)] = [key, dict[key]["val_acc_1"], dict[key]["val_acc_2"], dict[key]["test_acc_1"], dict[key]["test_acc_2"]]
                df["differences"] = df["test_acc_2"] - df["test_acc_1"]
                count = df.at[0, "count"] = df['differences'].size
                mean = df.at[0, "mean"] = statistics.mean(df["differences"].to_numpy())
                variance = df.at[0, "variance"] = statistics.stdev(df["differences"].to_numpy())
                t_value = df.at[0, "t_value"] = mean / variance * sqrt(count)
                df.at[0, "p_value"] = 2 * (1 - t.cdf(t_value, count-1)) * 100
                df.at[0, "max_accuracy"] = max(df['test_acc_2'])
                df.at[0, "max_increment"] = max(df['differences'])
                df.to_csv('./votingVars/' + votingMethod + '_' + optimizationMethod +'.csv')
                plot('./votingVars/' + votingMethod + '_' + optimizationMethod +'.csv')
            except:
                continue
    
if __name__ == "__main__":
    extractVoting()
    votingSummary()