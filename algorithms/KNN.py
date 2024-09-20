from pyod.models.knn import KNN
import Data.Process_data
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
import numpy as np
from Data.Process_data import generate_vector
import time


def get_AUC_knn(X, y, n_neighbors, times=False):
	st = time.time()
	neigh = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=1)
	neigh.fit(X)
	en = time.time()
	dist, ind = neigh.kneighbors(X)
	if times:
		return en - st
	anomaly_num = y.sum()
	distances = [dist[i][n_neighbors - 1] for i in range(len(dist))]
	y_pred = generate_vector(distances, anomaly_num)
	auc = roc_auc_score(y, y_pred)
	return auc

# if __name__ == '__main__':
# 	# X, y = Data.Process_data.pen_writing(n_anomalies=1)
# 	X, y = Data.Process_data.breast_cancer(n_anomalies=2)
# 	print(get_knn_AUC(X, y, n_neighbors=10))
