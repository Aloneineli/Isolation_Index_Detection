import numpy as np


def top_k_mahalanobis_zscore(matrix: np.array, mean: float, cov: np.array, k: int):
	"""
	
	:param matrix: points as rows and dimension as columns
	:param mean: mean of the distribution
	:param cov: covariance matrix d*d that points were created by
	:param k: number of top score points to return
	:return: top k points in each score method (mahalanobis and z score)
	"""
	mahalanobis_distances = []
	for row in matrix:
		diff = row - mean
		mahalanobis_dist = np.sqrt(np.dot(np.dot(diff, np.linalg.inv(cov)), diff.T))
		mahalanobis_distances.append(mahalanobis_dist)
	
	# Calculate z-score for each row
	z_scores = np.abs((matrix - mean) / np.std(matrix, axis=0))
	
	# Find the indices of the rows with the largest Mahalanobis distance and z-score
	top_k_mahalanobis_indices = np.argsort(mahalanobis_distances)[-k:]
	top_k_zscore_indices = np.argsort(np.max(z_scores, axis=1))[-k:]
	
	# Extract the top k rows
	top_k_mahalanobis_rows = matrix[top_k_mahalanobis_indices]
	top_k_zscore_rows = matrix[top_k_zscore_indices]
	
	return top_k_mahalanobis_rows, top_k_zscore_rows
