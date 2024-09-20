import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import Data.Process_data
from sklearn.metrics import roc_auc_score
import time


def get_AUC_LOF(X, y, n_neighbors, times=False):
	st = time.time()
	clf = LocalOutlierFactor(n_neighbors=n_neighbors)
	y_pred = clf.fit_predict(X)
	en = time.time()
	if times:
		return en - st
	y_pred[y_pred == 1] = 0
	y_pred[y_pred == -1] = 1
	auc = roc_auc_score(y, y_pred)
	return auc

#
# if __name__ == '__main__':
# 	X, y = Data.Process_data.pen_writing(n_anomalies=5)
# 	# X, y = Data.Process_data.breast_cancer(n_anomalies=10)
# 	print(get_LOF_AUC(X, y, n_neighbors=3))
