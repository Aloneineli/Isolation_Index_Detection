import numpy as np
from Isolation_Index import isolation_index
from Extremes import extreme
import Data.Process_data
import time


def extreme_recursion(m: np.array, n: int):
    start = time.time()
    out = []
    m_init = m.copy()
    def recursion(m_init, m, n):
        if m.shape[0] <= m_init.shape[0] - n:
            return out
        else:
            row = extreme(m)[1][0]
            out.append(row)

            idx_anomaly = np.where((m == row).all(axis=1))[0]
            return recursion(m_init, np.delete(m, idx_anomaly, axis=0), n)

    recursion(m_init, m, n)
    scores_ = list(range(len(out)))
    out = np.array(out)

    # Create an array of np.inf with the same number of rows as m_init
    final_scores = np.full(m_init.shape[0], np.inf)
    final_scores = np.array(final_scores, dtype='float32')

    # Fill in the scores for the rows present in out
    for i, row in enumerate(out):
        idx = np.where((m_init == row).all(axis=1))[0]
        final_scores[idx] = scores_[i]
    end = time.time()
    return final_scores, out


if __name__ == '__main__':
    # X, y = Data.Process_data.pen_writing(n_anomalies=4)
    n_anom = 4
    X, y = Data.Process_data.breast_cancer(n_anomalies=n_anom)
    scores, rows = extreme_recursion(X, n_anom)
    print(scores)
    idx = np.where(scores != np.inf)[0]
