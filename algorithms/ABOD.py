import time
import pandas as pd
from pyod.models.abod import ABOD
import Data.Process_data
import numpy as np
from sklearn.metrics import roc_auc_score
from Data.Process_data import generate_vector


def get_AUC_ABOD(X, y, n_neighbors, times=False):
	st = time.time()
	clf = ABOD(n_neighbors=n_neighbors)
	X = np.array(X, dtype='float32')
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
	list_times = []
	for n in np.arange(1, 10):
		print(f'doing anomaly {n}')
		if n%2 == 0:
			X, y = Data.Process_data.pen_writing(n_anomalies=n)
		else:
			X, y = Data.Process_data.breast_cancer(n_anomalies=n)
			
		for k in [5, 10, 20, 50]:
			_, t = get_AUC_ABOD(X, y, n_neighbors=k)
			list_times.append(t)
	mean = sum(list_times) / len(list_times)
	df = pd.DataFrame({'ABOD': [mean]})
	df.to_csv(f'ABOD time.csv')