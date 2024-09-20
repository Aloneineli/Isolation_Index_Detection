import numpy as np


def generate_random_positive_diagonal_matrix(n):
	diag = np.abs(np.random.rand(n)) * 10  # Generate random positive values
	return np.diag(diag)


def gram_schmidt(vectors):
	num_vecs = vectors.shape[0]
	basis = np.zeros_like(vectors)
	basis[0] = vectors[0] / np.linalg.norm(vectors[0])
	for i in range(1, num_vecs):
		vec = vectors[i]
		projection = np.zeros_like(vec)
		for j in range(i):
			projection += np.dot(vec, basis[j]) * basis[j]
		basis[i] = (vec - projection) / np.linalg.norm(vec - projection)
	return basis


def generate_random_orthogonal_matrix(n):
	random_vector = np.random.rand(n)
	orthogonal_vectors = gram_schmidt(np.vstack((random_vector, np.random.rand(n - 1, n))))
	return orthogonal_vectors.T


def rand_cov_matrix(n):
	# Generate random orthogonal matrix
	random_orthogonal_matrix = generate_random_orthogonal_matrix(n)
	
	# Generate random positive diagonal matrix
	random_positive_diagonal_matrix = generate_random_positive_diagonal_matrix(n)
	
	# Compute D^T * M * D
	result = np.dot(np.dot(random_positive_diagonal_matrix.T, random_orthogonal_matrix),
	                np.dot(random_orthogonal_matrix.T, random_positive_diagonal_matrix))
	return result
