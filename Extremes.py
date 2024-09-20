import numpy as np
from Isolation_Index import isolation_index
import Data.Process_data


def extreme(m: np.array):
    """

    :param m: rows are points and columns are features
    :return: vector of scores  - a score for each row in m while inf means it's not extreme
    and rows in order where first row has the smallest II index
    """
    num_points = m.shape[0]
    dim = m.shape[1]
    row_norms = np.linalg.norm(m, axis=1)
    normalized_matrix = m / row_norms[:, np.newaxis]
    normalized_matrix = normalized_matrix.T
    scalar_matrix = np.matmul(m, normalized_matrix)
    diagonal_elements = np.diag(scalar_matrix)
    max_values = np.max(scalar_matrix, axis=0)
    min_values = np.min(scalar_matrix, axis=0)
    max_columns = np.where(diagonal_elements == max_values)[0]
    min_columns = np.where(diagonal_elements == min_values)[0]
    combined_columns = np.concatenate([max_columns, min_columns])
    cols = list(set(combined_columns))

    scores = {}
    for column in cols:
        points = scalar_matrix[:, column]
        point = scalar_matrix[column, column]
        score = isolation_index(point, points)
        scores[column] = score
    sorted_scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}
    sorted_keys = list(sorted_scores.keys())
    sorted_rows = m[sorted_keys]
    scores_vector = np.zeros(m.shape[0], dtype='float32')
    scores_vector[sorted_keys] = np.array(list(sorted_scores.values()), dtype='float32')
    scores_vector[scores_vector == 0.0] = np.inf
    # the most anomaly is the first one is sorted rows
    return scores_vector, sorted_rows


if __name__ == '__main__':
    X, y = Data.Process_data.pen_writing(n_anomalies=4)
    # X, y = Data.Process_data.breast_cancer(n_anomalies=4)
    scores, rows = extreme(X)
    idx = np.where(scores != np.inf)[0]
