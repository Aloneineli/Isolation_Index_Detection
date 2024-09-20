import numpy as np
import pandas as pd
import Data.Process_data

from Extremes import extreme


def get_extreme_proportion_from_anomalies(X, y):
    anomaly_rows = get_anomaly_rows(X, y)
    scores, rows = extreme(X)
    cnt = count_identical_rows(rows, anomaly_rows)
    return (cnt / y.sum()) * 100


def count_identical_rows(A, B):
    # Ensure both matrices have the same number of columns
    if A.shape[1] != B.shape[1]:
        raise ValueError("Matrices must have the same number of columns")

    # Convert rows to sets of tuples for comparison
    set_A = set(map(tuple, A))
    set_B = set(map(tuple, B))

    # Find the intersection of the sets
    identical_rows = set_A.intersection(set_B)

    # Return the number of identical rows
    return len(identical_rows)


def get_anomaly_rows(X, y):
    mask = y == 1
    out = X[mask]
    return out


def get_results_for_extremes_in_data(anomalies, iterations):
    for data in ['breast', 'pen']:
        matrix = np.zeros((iterations, len(anomalies)))
        for anomaly_idx, num_anomalies in enumerate(anomalies):
            print(f'we are at anomaly {num_anomalies}')
            for iter in range(iterations):
                print(f' iteration num {iter}')
                if data == 'breast':
                    X, y = Data.Process_data.breast_cancer(num_anomalies)
                else:
                    X, y = Data.Process_data.pen_writing(num_anomalies)
                percent = get_extreme_proportion_from_anomalies(X, y)
                print(percent)
                matrix[iter, anomaly_idx] = percent
        matrix = pd.DataFrame(matrix, columns=anomalies)
        matrix.to_csv(f'extremes in {data} data.csv')


if __name__ == '__main__':
    anomalies = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    iterations = 20
    get_results_for_extremes_in_data(anomalies, iterations)
