from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import Extremes
import time
import numpy as np


def generate_matrix_size(n, d):
    x = np.random.rand(n, d)
    y_size = x.shape[0]
    y = np.random.randint(2, size=y_size)
    return x, y


def isolation_index(X):
    st = time.time()
    scores = Extremes.extreme(X)
    en = time.time()
    return en - st
