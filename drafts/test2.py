import os
import sklearn as skl
from sklearn.pipeline import make_pipeline, Pipeline
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

size = medium
F_SET = 'mfcc'
CV = 10

columns = ['mfcc', 'chrome_cens', 'tonnetz', 'spectral_contrast', 'spectral_centroid',
        'spectral_bandwidth', 'spectral_rolloff', 'rmse', 'zcr']

train = tracks['set', 'split'] == 'training'
val = tracks['set', 'split'] == 'validation'
test = tracks['set', 'split'] == 'test'

y_train = tracks.loc[size & (train | val), ('track', 'genre_top')]
X_train = features.loc[size & (train | val), F_SET]

y_test = tracks.loc[size & test, ('track', 'genre_top')]
X_test = features.loc[size & test, F_SET]
#--------------------------------------------------------------------------------
X_train, y_train = skl.utils.shuffle(X_train, y_train, random_state=42)
scaler = skl.preprocessing.StandardScaler(copy=False)
scaler.fit_transform(X_train)
scaler.transform(X_test)

cv = KFold(n_splits=CV, shuffle=True, random_state=42)
model = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42)

space = dict()
space['n_estimators'] = [10, 100, 500]
space['max_features'] = [2, 4, 6]

search = GridSearchCV(model, space, scoring='accuracy', cv=cv, refit=True)

result = search.fit(X_train, y_train)
best_model = result.best_estimator_

acc = best_model.score(X_test, y_test)
print('acc=%.3fS, best parameters:%s' % (acc, result.best_params_))