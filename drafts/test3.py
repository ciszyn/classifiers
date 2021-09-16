import os
import sklearn as skl
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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

columns = ['mfcc', 'chrome_cens', 'tonnetz', 'spectral_contrast', 'spectral_centroid',
        'spectral_bandwidth', 'spectral_rolloff', 'rmse', 'zcr']

train = tracks['set', 'split'] == 'training'
val = tracks['set', 'split'] == 'validation'
test = tracks['set', 'split'] == 'test'

#------------------
size = medium
F_SET = 'mfcc'
#------------------

y = tracks.loc[size, ('track', 'genre_top')]
X = features.loc[size, F_SET]

#--------------------------------------------------------------------------
print("training...")
#pipe = make_pipeline(StandardScaler(copy=False), PCA(), SVC(kernel='linear'))
#score = cross_val_score(pipe, X, y, cv=5)
y_train = tracks.loc[size & (train | val), ('track', 'genre_top')]
X_train = features.loc[size & (train | val), F_SET]

y_test = tracks.loc[size & test, ('track', 'genre_top')]
X_test = features.loc[size & test, F_SET]

X_train, y_train = skl.utils.shuffle(X_train, y_train, random_state=42)

scaler = StandardScaler(copy=False)
scaler.fit_transform(X_train)
scaler.transform(X_test)

svc = SVC(kernel="linear")
svc.fit(X_train, y_train)
score = svc.score(X_test, y_test)
print(score)