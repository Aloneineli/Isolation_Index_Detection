import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
import Extremes
import numpy as np

if __name__ == '__main__':
	iris = datasets.load_iris()
	housing = pd.read_csv('boston_housing.csv').to_numpy()
	num = 40
	d = 2
	normal = np.random.randn(num, d)
	matrix_iris = iris.data[0:num, 0:2]
	matrix_housing = housing[0:num, 0:2]
	plt.figure(figsize=(10, 5))
	for i, matrix in enumerate([matrix_iris, matrix_housing, normal]):
		if i == 3:
			mean_cols = 0
		else:
			mean_cols = np.mean(matrix, axis=0)
		matrix = matrix - mean_cols
		x_vals = matrix[:, 0]
		y_vals = matrix[:, 1]
		if i == 0:
			subtitle_name = 'Iris Data'
			x_add = 1
			y_add = 1
		elif i == 1:
			subtitle_name = 'Housing Data'
			x_add = 0.8
			y_add = 6
		else:
			subtitle_name = 'Normal Standard Distribution'
			x_add = 2
			y_add = 2
		plt.subplot(1, 3, i + 1)
		plt.scatter(x_vals, y_vals, label='Non Extremes', color='blue')
		extremes = Extremes.extreme(matrix)[1]
		plt.scatter(extremes[:, 0], extremes[:, 1], label='Extremes', color='green')
		most_extreme = (extremes[0, 0], extremes[0, 1])
		plt.scatter(most_extreme[0], most_extreme[1], label='Most Extreme', color='red')
		plt.title(f'{subtitle_name}')
		plt.xlim(np.min(x_vals) - x_add, np.max(x_vals) + x_add)
		plt.ylim(np.min(y_vals) - y_add, np.max(y_vals) + y_add)
		plt.legend()
		plt.legend(fontsize='small', prop={'size': 8})
	plt.show()
		
		
		
		
