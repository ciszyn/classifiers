from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from joblib import dump
from sklearn.pipeline import Pipeline
from variables import *
import os
from tools.preprocess import pre_process


def create_classifiers(classifiers, feature_sets):
    y_train, y_val, y_test, X_train, X_val, X_test = pre_process(
        tracks, features_all, list(features.columns.levels[0]))
    for clf_name, clf in classifiers.items():
        pipe = Pipeline(
            [('scaler', StandardScaler(copy=False)), (clf_name, clf)])
        pipe.fit(X_train, y_train)
        try:
            dump(pipe, "./models/"+set_size+'/'+clf_name+'.joblib')
        except:
            os.makedirs("./models/"+set_size)
            dump(pipe, "./models/"+set_size+'/'+clf_name+'.joblib')


if __name__ == "__main__":
    create_classifiers(classifiers_base, feature_sets)
