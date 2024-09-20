import pandas as pd
import math
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
import matplotlib.pyplot as plt


def plot_auc_vs_k(n_anomalies: list, data: str, n_neighbors: np.array, repetition: int):
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
                X, y = Data.Process_data.pen_writing(n_anomalies=n, percent=True)
                X = np.array(X, dtype='float32')
            elif data == 'breast_cancer':
                X, y = Data.Process_data.breast_cancer(n_anomalies=n, percent=True)
                X = np.array(X, dtype='float32')
            else:
                return
            for k in n_neighbors:
                knn_auc = algorithms.KNN.get_AUC_knn(X, y, n_neighbors=k)
                lof_auc = algorithms.LOF.get_AUC_LOF(X, y, n_neighbors=k)
                i_forest_auc = algorithms.IsolationForest.get_AUC_IsolationForest(X, y, n_estimators=k)
                loda_auc = algorithms.LODA.get_AUC_LODA(X, y, n_bins=k)
                # loci_auc = algorithms.LOCI.get_AUC_LOCI(X, y, n_neighbors=k)
                if k == 1 or k == 2:
                    hbos_auc = algorithms.HBOS.get_AUC_HBOS(X, y, n_bins=3)
                    sod_auc = algorithms.SOD.get_AUC_SOD(X, y, n_neighbors=3, ref_set=2)
                    abod_auc = algorithms.ABOD.get_AUC_ABOD(X, y, n_neighbors=2)
                    cof_auc = algorithms.COF.get_AUC_COF(X, y, n_neighbors=2)
                else:
                    hbos_auc = algorithms.HBOS.get_AUC_HBOS(X, y, n_bins=k)
                    sod_auc = algorithms.SOD.get_AUC_SOD(X, y, n_neighbors=int(k), ref_set=int(k - 1))
                    abod_auc = algorithms.ABOD.get_AUC_ABOD(X, y, n_neighbors=k)
                    cof_auc = algorithms.COF.get_AUC_COF(X, y, n_neighbors=int(k))
                iso_index_auc = algorithms.IsolationIndex.get_AUC_IsolationIndexR(X, y, math.ceil(X.shape[0]/80))
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
        df.to_csv(f'comparing_auc_for_10_methods_{n}%_anomalies_in{data}_data.csv')


# 	for algo in [('KNN', knn_to_plot_avg, (0, (1, 1)), 'black'),
# 	             ('LOF', lof_to_plot_avg, (0, (1, 1)), 'blue'),
# 	             ('Isolation Forest', i_forest_to_plot_avg, (0, (3, 1, 1, 1, 1, 1)), 'brown'),
# 	             ('LODA', loda_to_plot_avg, 'dotted', 'purple'),
# 	             ('HBOS', hbos_to_plot_avg, (0, (5, 1)), 'green'),
# 	             ('ABOD', abod_to_plot_avg, (0, (3, 1, 1, 1)), 'orange'),
# 	             ('SOD', sod_to_plot_avg, (0, (3, 1, 1, 1, 1, 1)), 'grey'),
# 	             ('Isolation Index', iso_index_to_plot_avg, 'solid', 'red')]:
# 		plt.subplot(3, 2, idx + 1)
# 		plt.plot(n_neighbors, algo[1], label=algo[0], linestyle=algo[2], color=algo[3])
# 	plt.xlabel('K')
# 	plt.ylabel('AUC')
# 	plt.ylim(0.0, 1.0)
# 	plt.title(f'{n} Anomalies')
# 	plt.legend(loc='lower right', fontsize='xx-small')
# plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.45)
# plt.show()


if __name__ == '__main__':
    anomalies = [1, 2, 3, 5]
    n_neighbors = np.arange(1, 100, 1)
    repetition = 10
    plot_auc_vs_k(anomalies, 'breast_cancer', n_neighbors, repetition)
