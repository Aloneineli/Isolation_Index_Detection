import numpy as np
from Isolation_Index_new import isolation_index
# from Isolation_Index import isolation_index
import time
import numpy as np
from numba import jit, prange


def scores_to_dict(cols, scores_array):
    return {int(cols[i]): scores_array[i] for i in range(len(cols))}

@jit(nopython=True, parallel=True)  # Enable parallel execution
def compute_scores(cols, scalar_matrix):
    scores = np.zeros(len(cols), dtype=np.float32)  # Use a NumPy array for scores
    for idx in prange(len(cols)):  # Use prange for parallel execution
        column = cols[idx]
        points = scalar_matrix[:, column]
        point = scalar_matrix[column, column]
        score = isolation_index(point, points)  # Now using the optimized function
        scores[idx] = score
    return scores


def extreme(m: np.array):
    """
    :param m: rows are points and columns are features
    :return: vector of scores - a score for each row in m while inf means it's not extreme
    and rows in order where first row has the smallest II index
    """
    # start_time = time.time()
    m = m.astype(np.float32)
    start_norm_time = time.time()
    row_norms = np.linalg.norm(m, axis=1)
    # print(f"Norm calculation time: {time.time() - start_norm_time:.6f} seconds")

    # Normalize the matrix
    # start_normalize_time = time.time()
    normalized_matrix = m / row_norms[:, np.newaxis]
    normalized_matrix = normalized_matrix.astype(np.float32)
    normalized_matrix = normalized_matrix.T
    # print(f"Normalization time: {time.time() - start_normalize_time:.6f} seconds")

    # Matrix multiplication
    # start_multiply_time = time.time()
    scalar_matrix = np.matmul(m, normalized_matrix)
    # print(f"Matrix multiplication time: {time.time() - start_multiply_time:.6f} seconds")

    # Extract diagonal elements
    # start_diag_time = time.time()
    diagonal_elements = np.diag(scalar_matrix)
    # print(f"Diagonal extraction time: {time.time() - start_diag_time:.6f} seconds")

    # Max and Min calculations
    # start_max_min_time = time.time()
    max_values = np.max(scalar_matrix, axis=0)
    min_values = np.min(scalar_matrix, axis=0)
    # print(f"Max/Min calculations time: {time.time() - start_max_min_time:.6f} seconds")

    # Identify extreme columns
    # start_extreme_cols_time = time.time()
    max_columns = np.where(diagonal_elements == max_values)[0]
    min_columns = np.where(diagonal_elements == min_values)[0]
    combined_columns = np.concatenate([max_columns, min_columns])
    cols = list(set(combined_columns))
    # print(f"Extreme columns identification time: {time.time() - start_extreme_cols_time:.6f} seconds")

    # Calculate scores
    # start_score_time = time.time()
    # scores = {i: 0.0 for i in cols}
    # for column in cols:
    #     points = scalar_matrix[:, column]
    #     point = scalar_matrix[column, column]
    #     score = isolation_index(point, points)
    #     scores[column] = score
    scores_array = compute_scores(cols, scalar_matrix)
    scores = scores_to_dict(cols, scores_array)

    # print(f"Score calculation time: {time.time() - start_score_time:.6f} seconds")

    # Sort scores
    # start_sort_time = time.time()
    sorted_scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}
    sorted_keys = list(sorted_scores.keys())
    sorted_rows = m[sorted_keys]
    # print(f"Sorting time: {time.time() - start_sort_time:.6f} seconds")

    # Prepare scores vector
    # start_vector_time = time.time()
    scores_vector = np.zeros(m.shape[0], dtype='float32')
    scores_vector[sorted_keys] = np.array(list(sorted_scores.values()), dtype='float32')
    scores_vector[scores_vector == 0.0] = np.inf
    # print(f"Scores vector preparation time: {time.time() - start_vector_time:.6f} seconds")

    # total_time = time.time() - start_time
    # print(f"Total execution time: {total_time:.6f} seconds")

    return scores_vector, sorted_rows