import os
import sklearn as skl
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from numpy import mean, std
import util

n_jobs = -1

AUDIO_DIR = os.environ.get('/media/misiek/Dane/fma_small')

tracks = util.load('/media/misiek/Dane/fma_metadata/tracks.csv')
genres = util.load('/media/misiek/Dane/fma_metadata/genres.csv')
features = util.load('/media/misiek/Dane/fma_metadata/features.csv')
echonest = util.load('/media/misiek/Dane/fma_metadata/echonest.csv')

small = tracks['set', 'subset'] <= 'small'
medium = tracks['set', 'subset'] <= 'medium'
large = tracks['set', 'subset'] <= 'large'

size = small
F_SET = 'mfcc'
CV = 5

columns = ['mfcc', 'chrome_cens', 'tonnetz', 'spectral_contrast', 'spectral_centroid',
        'spectral_bandwidth', 'spectral_rolloff', 'rmse', 'zcr']

train = tracks['set', 'split'] == 'training'
val = tracks['set', 'split'] == 'validation'
test = tracks['set', 'split'] == 'test'

y = tracks.loc[size, ('track', 'genre_top')]
X = features.loc[size, F_SET]

PARAMS = {'degree': [1, 2, 3]}
cv_outer = KFold(n_splits=10, shuffle=True, random_state=42)
outer_results = list()
print(X.size)
for train_ix, test_ix in cv_outer.split(X):
    X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
    y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

    cv_inner = KFold(n_splits=3, shuffle=True, random_state=42)
    model = RandomForestClassifier(random_state=42)

    space = dict()
    space['n_estimators'] = [10, 100, 500]
    space['max_features'] = [2, 4, 6]

    search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True)

    result = search.fit(X_train, y_train)
    best_model = result.best_estimator_

    yhat = best_model.predict(X_test)
    acc = accuracy_score(y_test, yhat)
    outer_results.append(acc)
    print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))

print('Accuracy: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))
