import pandas as pd
import algorithms.KNN
import algorithms.LOF
import algorithms.COF
import algorithms.ECOD
import algorithms.HBOS
import algorithms.IsolationForest
import algorithms.IsolationIndex
import algorithms.LODA
import algorithms.LOCI
import algorithms.SOD
import algorithms.ABOD
import Data.Process_data
import numpy as np


def plot_auc_vs_k_times(n_anomalies: list, data: str, n_neighbors: np.array, repetition: int):
	out = pd.DataFrame()
	for idx, n in enumerate(n_anomalies):
		knn_to_plot_mat = np.zeros((repetition, n_neighbors.shape[0]), dtype='float32')
		lof_to_plot_mat = np.zeros((repetition, n_neighbors.shape[0]), dtype='float32')
		i_forest_to_plot_mat = np.zeros((repetition, n_neighbors.shape[0]), dtype='float32')
		loda_to_plot_mat = np.zeros((repetition, n_neighbors.shape[0]), dtype='float32')
		hbos_to_plot_mat = np.zeros((repetition, n_neighbors.shape[0]), dtype='float32')
		abod_to_plot_mat = np.zeros((repetition, n_neighbors.shape[0]), dtype='float32')
		sod_to_plot_mat = np.zeros((repetition, n_neighbors.shape[0]), dtype='float32')
		cof_to_plot_mat = np.zeros((repetition, n_neighbors.shape[0]), dtype='float32')
		# loci_to_plot_mat = np.zeros((repetition, n_neighbors.shape[0]), dtype='float32')
		iso_index_to_plot_mat = np.zeros((repetition, n_neighbors.shape[0]), dtype='float32')
		for i in range(0, repetition):
			print(f'doing ks of anomaly {n} and rep {i}')
			if data == 'pen_writing':
				X, y = Data.Process_data.pen_writing(n_anomalies=n)
			elif data == 'breast_cancer':
				X, y = Data.Process_data.breast_cancer(n_anomalies=n)
			else:
				return
			for k in n_neighbors:
				knn_auc = algorithms.KNN.get_AUC_knn(X, y, n_neighbors=k, times=True)
				lof_auc = algorithms.LOF.get_AUC_LOF(X, y, n_neighbors=k, times=True)
				i_forest_auc = algorithms.IsolationForest.get_AUC_IsolationForest(X, y, n_estimators=k, times=True)
				loda_auc = algorithms.LODA.get_AUC_LODA(X, y, n_bins=k, times=True)
				# loci_auc = algorithms.LOCI.get_AUC_LOCI(X, y, n_neighbors=k)
				if k == 1 or k == 2:
					hbos_auc = algorithms.HBOS.get_AUC_HBOS(X, y, n_bins=3, times=True)
					sod_auc = algorithms.SOD.get_AUC_SOD(X, y, n_neighbors=3, ref_set=2, times=True)
					abod_auc = algorithms.ABOD.get_AUC_ABOD(X, y, n_neighbors=2, times=True)
					cof_auc = algorithms.COF.get_AUC_COF(X, y, n_neighbors=2, times=True)
				else:
					hbos_auc = algorithms.HBOS.get_AUC_HBOS(X, y, n_bins=k, times=True)
					sod_auc = algorithms.SOD.get_AUC_SOD(X, y, n_neighbors=int(k), ref_set=int(k - 1), times=True)
					abod_auc = algorithms.ABOD.get_AUC_ABOD(X, y, n_neighbors=k, times=True)
					cof_auc = algorithms.COF.get_AUC_COF(X, y, n_neighbors=int(k), times=True)
				iso_index_auc = algorithms.IsolationIndex.get_AUC_IsolationIndex(X, y, times=True)
				knn_to_plot_mat[i, k - 1] = knn_auc
				lof_to_plot_mat[i, k - 1] = lof_auc
				i_forest_to_plot_mat[i, k - 1] = i_forest_auc
				loda_to_plot_mat[i, k - 1] = loda_auc
				# loci_to_plot_mat[i, k - 1] = loci_auc
				cof_to_plot_mat[i, k - 1] = cof_auc
				hbos_to_plot_mat[i, k - 1] = hbos_auc
				abod_to_plot_mat[i, k - 1] = abod_auc
				sod_to_plot_mat[i, k - 1] = sod_auc
				iso_index_to_plot_mat[i, k - 1] = iso_index_auc
		
		knn_to_plot_avg = np.mean(knn_to_plot_mat, axis=0)
		lof_to_plot_avg = np.mean(lof_to_plot_mat, axis=0)
		i_forest_to_plot_avg = np.mean(i_forest_to_plot_mat, axis=0)
		loda_to_plot_avg = np.mean(loda_to_plot_mat, axis=0)
		# loci_to_plot_avg = np.mean(loci_to_plot_mat, axis=0)
		cof_to_plot_avg = np.mean(cof_to_plot_mat, axis=0)
		hbos_to_plot_avg = np.mean(hbos_to_plot_mat, axis=0)
		abod_to_plot_avg = np.mean(abod_to_plot_mat, axis=0)
		sod_to_plot_avg = np.mean(sod_to_plot_mat, axis=0)
		iso_index_to_plot_avg = np.mean(iso_index_to_plot_mat, axis=0)
		df = pd.DataFrame({'k': n_neighbors,
		                   'knn': knn_to_plot_avg,
		                   'lof': lof_to_plot_avg,
		                   'cof': cof_to_plot_avg,
		                   # 'loci': loci_to_plot_avg,
		                   'i_forest': i_forest_to_plot_avg,
		                   'loda': loda_to_plot_avg,
		                   'hbos': hbos_to_plot_avg,
		                   'abod': abod_to_plot_avg,
		                   'sod': sod_to_plot_avg,
		                   'iso_index': iso_index_to_plot_avg})
		out = pd.concat([out, df], axis=0)
	out.to_csv(f'times_9_methods_{data}.csv')


if __name__ == '__main__':
	anomalies = [1, 2, 3, 5, 7, 10]
	n_neighbors = np.arange(1, 50, 1)
	repetition = 7
	plot_auc_vs_k_times(anomalies, 'breast_cancer', n_neighbors, repetition)