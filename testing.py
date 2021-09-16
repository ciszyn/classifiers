import numpy as np
import sklearn.preprocessing
from numpy import tensordot

def normalize_mat_rows(mat):
    return sklearn.preprocessing.normalize(mat, norm="l1")

a = np.matrix('1 2; 3 4')
b = np.matrix('1 2')
print(tensordot(a, b, axes=([1], [1])))