from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from joblib import dump, load
import os
from sklearn.metrics import accuracy_score
import numpy as np

def svm_cnn():
    enc = LabelEncoder()
    y_training = pd.read_csv("./features/y_training.csv", index_col=0)
    y_training = np.ravel(y_training)
    y_training = enc.fit_transform(y_training)
    X_training = pd.read_csv("./features/x_training.csv", index_col=0)
    X_training, y_training = shuffle(X_training, y_training, random_state=42)

    clf = SVC(kernel='rbf')
    clf.fit(X_training, y_training)
    dump(clf, "./models/small/svm_cnn.joblib")

def check_score():
    enc = LabelEncoder()
    X_test = pd.read_csv("./features/x_test.csv", index_col=0)
    y_test = pd.read_csv("./features/y_test.csv", index_col=0)
    y_test = np.ravel(y_test)
    y_test = enc.fit_transform(y_test)
    clf = load("./models/small/svm_cnn.joblib")
    predictions = clf.predict(X_test)
    y_test = np.ravel(y_test)
    score = accuracy_score(y_test, predictions)
    print(score)

if __name__ == '__main__':
    svm_cnn()
    check_score()