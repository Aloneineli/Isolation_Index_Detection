from pyod.models.loda import LODA
import Data.Process_data
from sklearn.metrics import roc_auc_score
import numpy as np
from Data.Process_data import generate_vector
import time


def get_AUC_LODA(X, y, n_bins, times=False):
	st = time.time()
	clf = LODA(n_bins=n_bins)
	clf.fit(X)
	en = time.time()
	y_train_scores = clf.decision_scores_
	if times:
		return en - st
	n_anomalies = y.sum()
	y_pred = generate_vector(y_train_scores, n_anomalies)
	auc = roc_auc_score(y, y_pred)
	return auc

# if __name__ == '__main__':
# 	X, y = Data.Process_data.pen_writing(n_anomalies=4)
# 	# X, y = Data.Process_data.breast_cancer(n_anomalies=4)
# 	print(get_AUC_LODA(X, y, 10))
