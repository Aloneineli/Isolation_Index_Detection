import numpy as np
import cov_random


def generate_standard_gauss_points(n: int, d: int):
	"""
	:param n: number of point to generate
	:param d: dimension of points
	:return: points. matrix nxd
	"""
	points = np.random.randn(n, d)
	return points


def generate_multivariate_points(n: int, d: int, mean: float, cov: np.array):
	uncorrelated_points = np.random.normal(size=(n, d))
	L = np.linalg.cholesky(cov)
	correlated_points = np.dot(uncorrelated_points, L.T)
	points = correlated_points + mean
	return points


def generate_positive_definite_correlation_matrix(d: int):
	while True:
		A = np.random.rand(d, d)
		A = 0.5 * (A + A.T)
		eigvals, eigvecs = np.linalg.eigh(A)
		if np.all(eigvals > 0):
			cov = np.dot(np.dot(eigvecs, np.diag(np.sqrt(eigvals))), eigvecs.T)
			return cov


def define_multivariate_points_for_experiment(n: int, d: int):
	"""This function creates a matrix of n*d with a random cov matrix and zero mean
	
	:param n: number of points
	:param d: dimension
	:return:
	"""
	cov = cov_random.rand_cov_matrix(d)
	# cov = np.eye(d)
	mean = 0
	points = generate_multivariate_points(n, d, mean, cov)
	return points, mean, cov


	
	
