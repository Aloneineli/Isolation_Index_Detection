from pyod.models.loci import LOCI
import Data.Process_data
from sklearn.metrics import roc_auc_score
import numpy as np
from Data.Process_data import generate_vector


def get_AUC_LOCI(X, y, n_neighbors):
	clf = LOCI(k=n_neighbors)
	clf.fit(X)
	y_train_scores = clf.decision_scores_
	n_anomalies = y.sum()
	y_pred = generate_vector(y_train_scores, n_anomalies)
	auc = roc_auc_score(y, y_pred)
	return auc


if __name__ == '__main__':
	X, y = Data.Process_data.pen_writing(n_anomalies=5)
# 	X, y = Data.Process_data.breast_cancer(n_anomalies=4)
	print(get_AUC_LOCI(X, y, n_neighbors=3))
