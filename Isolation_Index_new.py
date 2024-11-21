import numpy as np
import numpy as np
from numba import jit, prange

@jit(nopython=True)
def isolation_index(point: float, points: np.array, extreme=True):
    """
    :param point: real number for which we want to find the isolation index.
    :param points: real numbers that are on the same line.
    :param extreme: for future use if point is not an extreme on the line of points
    :return: isolation index of the point or NaN if it's not extreme.
    """
    # Check if the point is an extreme point
    max_point = (point == points.max())
    min_point = (point == points.min())

    if not max_point and not min_point:
        return np.nan  # Use NaN to indicate it's not an extreme point

    # Calculate distances from the point
    distances = np.abs(points - point)

    # Set the distance to the point itself to infinity to ignore it
    distances[distances == 0] = np.inf

    # Find nearest point distance
    nearest_distance = np.min(distances)

    # Calculate isolation index
    return np.inf if nearest_distance == 0 else 1 / nearest_distance