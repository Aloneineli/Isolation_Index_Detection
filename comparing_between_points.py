import numpy as np


def percentage_matching_rows(m, extremes):
	num_matching_rows = 0
	for row_m in m:
		if np.array_equal(row_m, extremes[0, :]):
			num_matching_rows += 1
			break  # No need to continue searching for matches once found
	return num_matching_rows
