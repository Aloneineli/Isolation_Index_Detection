import numpy as np


def isolation_index(point: float, points: np.array, extreme=True):
	"""
	
	:param point: real number for which we want to find the isolation index.
	:param points: real numbers that are on the same line.
	:param extreme: for future use if point is not an extreme on the line of points
	:return: isolation index of the point.
	"""
	if (points.max() != point) and (points.min() != point):
		return f'{point} is not an extreme point'
	max_point = False
	min_point = False
	if points.max() == point:
		max_point = True
		idx = -2
	else:
		min_point = True
		idx = 1
	ascending_points = np.sort(points)
	nearest_point = ascending_points[idx]
	distance = abs(nearest_point - point)
	if distance == 0:
		isolation_index_point = np.inf
	else:
		isolation_index_point = 1 / distance
	return isolation_index_point
	


