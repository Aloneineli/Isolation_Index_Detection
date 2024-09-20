from pyod.models.cof import COF
import Data.Process_data
import numpy as np
from sklearn.metrics import roc_auc_score

import Extremes_Recursion
from Data.Process_data import generate_vector_min
import Extremes
import time


def get_AUC_IsolationIndex(X, y, times=False):
	st = time.time()
	scores = Extremes.extreme(X)
	en = time.time()
	if times:
		return en - st
	n_anomalies = y.sum()
	y_pred = generate_vector_min(scores, n_anomalies)
	auc = roc_auc_score(y, y_pred)
	return auc

def get_AUC_IsolationIndexR(X, y, n, times=False):
	st = time.time()
	scores, _ = Extremes_Recursion.extreme_recursion(X, n)
	en = time.time()
	if times:
		return en - st
	n_anomalies = y.sum()
	y_pred = generate_vector_min(scores, n_anomalies)
	auc = roc_auc_score(y, y_pred)
	return auc


if __name__ == '__main__':
	X, y = Data.Process_data.pen_writing(n_anomalies=1)
	# X, y = Data.Process_data.breast_cancer(n_anomalies=1)
	print(get_AUC_IsolationIndex(X, y))
