import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def generate_2d_normal_points_matrix(mean_x, mean_y, std_x, std_y, num_points):
	# Generate points from a 2-D normal distribution and store them in a matrix
	x = np.random.normal(mean_x, std_x, size=num_points)
	y = np.random.normal(mean_y, std_y, size=num_points)
	points = np.column_stack((x, y))
	return points


if __name__ == '__main__':
	A = generate_2d_normal_points_matrix(0, 0, 0.4, 0.05, 100)
	B = generate_2d_normal_points_matrix(0, 0, 0.05, 0.4, 100)
	M = np.row_stack((A, B))
	anomalies = np.array([[0.75, 0.75], [-0.75, 0.75]])
	M = np.row_stack([M, anomalies])
	pca = PCA(n_components=2)
	pca.fit(M)
	pca_vectors = pca.components_
	
	# Plot the data points and PCA vectors
	plt.figure(figsize=(8, 6))
	plt.scatter(M[:, 0], M[:, 1], label='Data Points')
	plt.quiver(pca.mean_[0], pca.mean_[1], pca_vectors[0, 0], pca_vectors[0, 1], angles='xy', scale_units='xy', scale=1,
	           color='r', label='PCA Vector 1')
	plt.quiver(pca.mean_[0], pca.mean_[1], pca_vectors[1, 0], pca_vectors[1, 1], angles='xy', scale_units='xy', scale=1,
	           color='g', label='PCA Vector 2')
	plt.scatter(pca.mean_[0], pca.mean_[1], color='k', label='Mean Point')
	plt.scatter(anomalies[:, 0], anomalies[:, 1], color='m', label='Anomaly')
	
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.title('PCA Vectors of Data')
	plt.legend()
	plt.grid(True)
	plt.axis('equal')
	plt.show()
