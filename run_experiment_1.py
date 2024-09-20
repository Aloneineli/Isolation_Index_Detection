import numpy as np
import generate_points
import Extremes
import mahalanobis_z_score
import comparing_between_points
import pandas as pd

if __name__ == '__main__':
	out = pd.DataFrame()
	ps = [1, 5, 10]
	for p in ps:
		ns = np.arange(100, 2500, 75)
		ds = [2, 3, 5, 7, 9]
		for n in ns:
			for d in ds:
				mean_ma = []
				iteration = 1
				while iteration <= 40:
					points, mean, cov = generate_points.define_multivariate_points_for_experiment(n, d)
					extreme_points = Extremes.extreme(points)[1][0:1, :]
					num_extremes = extreme_points.shape[0]
					ma_points, _ = mahalanobis_z_score.top_k_mahalanobis_zscore(points, mean, cov, p)
					percent_vs_mahal = comparing_between_points.percentage_matching_rows(ma_points, extreme_points)
					mean_ma.append(percent_vs_mahal)
					iteration += 1
				mean_maha = np.array(mean_ma, dtype='float32')
				success = np.mean(mean_maha)
				row = pd.DataFrame({'N': n, 'd': d, 'Success': success, 'Matching Points': p}, index=[0])
				out = pd.concat([out, row], axis=0)
				print(f'done n={n} with d={d}, success {success}, matching {p}')
	out = out.reset_index(drop=True)
	out.to_csv(f'Mahalanobis_distance_comparison.csv')
