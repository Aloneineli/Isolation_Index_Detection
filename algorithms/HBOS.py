from pyod.models.hbos import HBOS
import Data.Process_data
import numpy as np
from sklearn.metrics import roc_auc_score
from Data.Process_data import generate_vector
import time


def get_AUC_HBOS(X, y, n_bins, times=False):
	st = time.time()
	clf = HBOS(n_bins=n_bins)
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
# 	X, y = Data.Process_data.pen_writing(n_anomalies=1)
# 	# X, y = Data.Process_data.breast_cancer(n_anomalies=1)
# 	print(get_AUC_HBOS(X, y))
