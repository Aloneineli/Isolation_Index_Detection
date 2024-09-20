import math
import pandas as pd
import random
from Data.Breast_Cancer import get_breast_cancer
from Data.Pen_Writing import get_pen_writing
import numpy as np
from ucimlrepo import fetch_ucirepo


def breast_cancer(n_anomalies=1, percent=False):
	"""
	This function generates breast cancer data with targets. the input set definr how much from
	Malignant we are taking
	:param n_anomalies: number of anomalies in the data
	:param percent: whether to generate a percentage or fixed amount of anomalies
	:return: features and targets of the processed data as series
	"""
	X = None
	y = None
	while (X is None) and (y is None):
		try:
			X, y = get_breast_cancer()
		except:
			print('pulling data wrong')

	cols = X.columns
	mat = pd.concat([X, y], axis=1)
	mat_B = mat[mat['Diagnosis'] == 'B'].reset_index(drop=True)
	mat_M = mat[mat['Diagnosis'] == 'M'].reset_index(drop=True)
	if percent:
		n_anomalies = math.ceil((X.shape[0] / 100) * n_anomalies)
	anomaly = mat_M.sample(n=n_anomalies)
	out = pd.concat([mat_B, anomaly], axis=0).reset_index(drop=True)
	out['Diagnosis'] = out['Diagnosis'].map({'B': 0, 'M': 1})
	return out[cols].to_numpy(), out['Diagnosis'].to_numpy()


def pen_writing(n_anomalies=1, percent=False):
	"""
	This function generates digit handwritten data with targets. the number of anomalies are set to be
	the number of digits that are not the normal instance. for example if n_anomalies=2 then the data will consists
	of n-2 from same digits and 2 from digits that are not that one.
	:param n_anomalies: number of anomalies in the data
	:param percent: whether to generate a percentage or fixed amount of anomalies
	:return: features and targets of the processed data as series
	"""
	X, y = get_pen_writing()
	cols = X.columns
	mat = pd.concat([X, y], axis=1)
	random_number_normal = random.randint(0, 9)
	random_number_anomaly = random.randint(0, 9)
	while random_number_anomaly == random_number_normal:
		random_number_anomaly = random.randint(0, 9)
	mat_normal = mat[mat['Class'] == random_number_normal].reset_index(drop=True)
	mat_anomaly = mat[mat['Class'] != random_number_normal].reset_index(drop=True)
	if percent:
		n_anomalies = math.ceil((X.shape[0] / 100) * n_anomalies)
	anomaly = mat_anomaly.sample(n=n_anomalies)
	out = pd.concat([mat_normal, anomaly], axis=0).reset_index(drop=True)
	if n_anomalies > mat_normal.shape[0]:
		return "ERROR"
	out = out.drop(out.index[:n_anomalies])
	out = out.reset_index(drop=True)
	out['Class'] = out['Class'].apply(lambda x: 0 if x == random_number_normal else 1)
	return out[cols].to_numpy(), out['Class'].to_numpy()


def generate_vector(y_pred, k):
	distances_array = np.array(y_pred)
	top_k_indices = np.argsort(distances_array)[-k:]
	vector_array = np.zeros_like(distances_array)
	vector_array[top_k_indices] = 1
	return vector_array


def generate_vector_min(y_pred, k):
	max_anomalies = (y_pred != np.inf).sum()
	n_anomalies_to_take = np.minimum(max_anomalies, k)
	distances_array = np.array(y_pred)
	min_k_indices = np.argsort(distances_array)[:n_anomalies_to_take]
	vector_array = np.zeros_like(distances_array)
	vector_array[min_k_indices] = 1
	return vector_array


if __name__ == '__main__':
	# fetch dataset
	pen_based_recognition_of_handwritten_digits = fetch_ucirepo(id=81)

	# data (as pandas dataframes)
	X = pen_based_recognition_of_handwritten_digits.data.features
	y = pen_based_recognition_of_handwritten_digits.data.targets

	# metadata
	print(pen_based_recognition_of_handwritten_digits.metadata)

	# variable information
	print(pen_based_recognition_of_handwritten_digits.variables)