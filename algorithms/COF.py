from pyod.models.cof import COF
import Data.Process_data
import numpy as np
from sklearn.metrics import roc_auc_score
from Data.Process_data import generate_vector
import time


def get_AUC_COF(X, y, n_neighbors, times=False):
	st = time.time()
	clf = COF(n_neighbors=n_neighbors)
	clf.fit(X)
	en = time.time()
	y_train_scores = clf.decision_scores_
	if times:
		return en - st
	n_anomalies = y.sum()
	y_pred = generate_vector(y_train_scores, n_anomalies)
	auc = roc_auc_score(y, y_pred)
	return auc


if __name__ == '__main__':
	X, y = Data.Process_data.pen_writing(n_anomalies=5)
	# X, y = Data.Process_data.breast_cancer(n_anomalies=4)
	print(get_AUC_COF(X, y, n_neighbors=20))
