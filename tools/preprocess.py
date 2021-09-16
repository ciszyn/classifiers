from sklearn.preprocessing import LabelEncoder, StandardScaler
from variables import *
from sklearn.utils import shuffle


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
