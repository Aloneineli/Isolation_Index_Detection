from pyod.models.ecod import ECOD
import Data.Process_data
import numpy as np
from sklearn.metrics import roc_auc_score
from Data.Process_data import generate_vector


def get_AUC_ECOD(X, y):
	# this is parameter free - should talk about this in the paper
	clf = ECOD()
	clf.fit(X)
	y_train_scores = clf.decision_scores_
	n_anomalies = y.sum()
	y_pred = generate_vector(y_train_scores, n_anomalies)
	auc = roc_auc_score(y, y_pred)
	return auc


# if __name__ == '__main__':
	# X, y = Data.Process_data.pen_writing(n_anomalies=1)
	# X, y = Data.Process_data.breast_cancer(n_anomalies=3)
	# print(get_ECOD_AUC(X, y))
