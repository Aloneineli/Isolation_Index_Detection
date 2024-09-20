import Data.mnist
from matplotlib import pyplot as plt
import numpy as np
from pyod.models.abod import ABOD


def find_most_anomalous_rows_abod(X, k=5, n_anomalous=1):
    # Step 1: Fit the ABOD model
    X = X.astype(np.float64)
    clf = ABOD(n_neighbors=k)
    clf.fit(X)

    # Step 2: Compute the ABOD scores
    abod_scores = clf.decision_scores_

    # Step 3: Identify the most anomalous rows
    most_anomalous_indices = np.argsort(-abod_scores)[:n_anomalous]
    most_anomalous_rows = X[most_anomalous_indices]

    return most_anomalous_rows, abod_scores[most_anomalous_indices]


def find_optimal_k_abod(X, k_values, n_anomalous=1):
    best_k = None
    best_score = -np.inf
    best_row = None

    for k in k_values:
        rows, scores = find_most_anomalous_rows_abod(X, k=k, n_anomalous=n_anomalous)
        if scores[0] > best_score:
            best_score = scores[0]
            best_k = k
            best_row = rows[0]

    return best_k, best_row


if __name__ == '__main__':
    X, y = Data.mnist.get_data(main_class=2, anomaly_class=4, num_main=99, num_anomalies=1, anomaly_place=3)
    # Define the values of k to check
    k_values = [3, 5, 7, 10, 20, 30]

    # Find the optimal k
    optimal_k, optimal_row = find_optimal_k_abod(X, k_values, n_anomalous=1)
    print(f"Optimal k: {optimal_k}")

    # Plot the most anomalous image using the optimal k
    X_reshaped = optimal_row.reshape((28, 28))
    plt.imshow(X_reshaped, cmap='gray')  # Use 'gray' colormap for a grayscale image
    plt.colorbar()  # Add a colorbar to show the intensity scale
    plt.title("Most Anomalous Handwritten Digit Image")
    plt.show()
