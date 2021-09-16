from tools import util
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from modelScripts import CNN, CRNN, CRNN2, CRNN3, CRNN4, CRNN4_conv11, CRNN4norm, CRNN5, CRNN6, CRNN7, CRNN8, CRNN9, CRNN10, CRNN11, CRNNsplit, PRCNN

set_size = 'small'

metadata_dir = './fma_metadata/'
AUDIO_DIR = './fma_' + set_size
spec_full_path = './full_' + set_size + '/'
spec_path = './fma_spectrograms/' + set_size + '/'

tracks = util.load(metadata_dir + 'tracks.csv')
features = util.load(metadata_dir + 'features.csv')
echonest = util.load(metadata_dir + 'echonest.csv')

subset = tracks.index[tracks['set', 'subset'] <= set_size]

features_all = features.join(echonest, how='inner').sort_index(axis=1)

tracks = tracks.loc[subset]
features_all = features.loc[subset]

train = tracks.index[tracks['set', 'split'] == 'training']
val = tracks.index[tracks['set', 'split'] == 'validation']
test = tracks.index[tracks['set', 'split'] == 'test']

genres = list(LabelEncoder().fit(tracks['track', 'genre_top']).classes_)

feature_sets = {'all': list(features.columns.levels[0])}

measures = ["accuracy", "TP_rate", "FP_rate", "Precision", "F_measure"]

classifiers_base = {
    'LR': LogisticRegression(),
    'kNN': KNeighborsClassifier(n_neighbors=200),
    'SVCrbf': SVC(kernel='rbf'),
    'SVCpoly1': SVC(kernel='poly', degree=1),
    'linSVC1': SVC(kernel="linear"),
    'linSVC2': LinearSVC(),
    'DT': DecisionTreeClassifier(max_depth=5),
    'RF': RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    'AdaBoost': AdaBoostClassifier(n_estimators=10),
    'MLP1': MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000),
    'MLP2': MLPClassifier(hidden_layer_sizes=(200, 50), max_iter=2000),
    'NB': GaussianNB(),
    'QDA': QuadraticDiscriminantAnalysis(),
}

classifiers_deep = {
    'cnn': CNN.getModel,
    'crnn': CRNN.getModel,
    'crnn2': CRNN2.getModel,
    'crnn3': CRNN3.getModel,
    'crnn4': CRNN4.getModel,
    'crnn5': CRNN5.getModel,
    'crnn6': CRNN6.getModel,
    'crnn7': CRNN7.getModel,
    'crnn8': CRNN8.getModel,
    'crnn9': CRNN9.getModel,
    'crnn10': CRNN10.getModel,
    'crnn11': CRNN11.getModel,
    'prcnn': PRCNN.getModel,
}
