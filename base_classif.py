from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import StandardScaler
from joblib import dump
from sklearn.pipeline import Pipeline
from variables import *
import os


def pre_process(tracks, features, columns):
    enc = LabelEncoder()
    labels = tracks['track', 'genre_top']
    print(labels[train])
    print(features.loc[train, columns])
    y_train = enc.fit_transform(labels[train])
    y_val = enc.transform(labels[val])
    y_test = enc.transform(labels[test])
    X_train = features.loc[train, columns].to_numpy()
    X_val = features.loc[val, columns].to_numpy()
    X_test = features.loc[test, columns].to_numpy()

    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    return y_train, y_val, y_test, X_train, X_val, X_test


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
