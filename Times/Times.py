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
import sklearn_algorithms


def check_times_for_methods(n, d, k, rep):
	knn = np.zeros(rep, dtype='float32')
	lof = np.zeros(rep, dtype='float32')
	isolation_forest = np.zeros(rep, dtype='float32')
	loda = np.zeros(rep, dtype='float32')
	isolation_index = np.zeros(rep, dtype='float32')
	hbos = np.zeros(rep, dtype='float32')
	# abod = np.zeros(rep, dtype='float32')
	# sod = np.zeros(rep, dtype='float32')
	# cof = np.zeros(rep, dtype='float32')
	for i in range(0, rep):
		X, y = sklearn_algorithms.generate_matrix_size(n, d)
		t_knn = algorithms.KNN.get_AUC_knn(X, y, n_neighbors=k, times=True)
		t_lof = algorithms.LOF.get_AUC_LOF(X, y, n_neighbors=k, times=True)
		t_isolation_forest = algorithms.IsolationForest.get_AUC_IsolationForest(X, y, n_estimators=k, times=True)
		t_loda = algorithms.LODA.get_AUC_LODA(X, y, n_bins=k, times=True)
		t_isolation_index = algorithms.IsolationIndex.get_AUC_IsolationIndex(X, y, times=True)
		t_hbos = algorithms.HBOS.get_AUC_HBOS(X, y, n_bins=k, times=True)
		# t_abod = algorithms.ABOD.get_AUC_ABOD(X, y, n_neighbors=k, times=True)
		# t_sod = algorithms.SOD.get_AUC_SOD(X, y, n_neighbors=int(k), ref_set=int(k - 1), times=True)
		# t_cof = algorithms.COF.get_AUC_COF(X, y, n_neighbors=int(k), times=True)
		knn[i] = t_knn
		lof[i] = t_lof
		isolation_forest[i] = t_isolation_forest
		loda[i] = t_loda
		hbos[i] = t_hbos
		# abod[i] = t_abod
		# sod[i] = t_sod
		# cof[i] = t_cof
		isolation_index[i] = t_isolation_index
	knn_mean = np.mean(knn)
	lof_mean = np.mean(lof)
	isolation_forest_mean = np.mean(isolation_forest)
	loda_mean = np.mean(loda)
	hbos_mean = np.mean(hbos)
	# abod_mean = np.mean(abod)
	# sod_mean = np.mean(sod)
	# cof_mean = np.mean(cof)
	isolation_index_mean = np.mean(isolation_index)
	row = pd.DataFrame({'N': n, 'd': d,
	                    'KNN': knn_mean,
	                    'LOF': lof_mean,
	                    'Isolation Forest': isolation_forest_mean,
	                    'LODA': loda_mean,
	                    'HBOS': hbos_mean,
	                    # 'ABOD': abod_mean,
	                    # 'SOD': sod_mean,
	                    # 'COF': cof_mean,
	                    'Isolation Index': isolation_index_mean
						},
	                   index=[0])
	return row


def make_csv(rep):
	k = 20
	out = pd.DataFrame()
	for d in [20]:
		for n in [50, 100, 200, 350, 500, 700, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000, 8000]:
		# for n in list(np.arange(100, 20000, 1000)):
			print(n, d)
			row = check_times_for_methods(int(n), d, k, rep)
			out = pd.concat([out, row], axis=0)
	out.to_csv('Times_fast_methods_new.csv')
	return


if __name__ == '__main__':
	make_csv(3)

