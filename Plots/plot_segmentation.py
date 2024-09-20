import numpy as np
from matplotlib import pyplot as plt
import Data.segmentation
from algorithms_1.cof import find_most_anomalous_rows_cof
from algorithms_1.abod import find_most_anomalous_rows_abod
from algorithms_1.hbos import find_most_anomalous_rows_hbos
from algorithms_1.isolation_forest import find_most_anomalous_rows_iforest
from algorithms_1.knn import find_most_anomalous_rows_knn
from algorithms_1.loda import find_most_anomalous_rows_loda
from algorithms_1.lof import find_most_anomalous_rows_lof
from algorithms_1.sod import find_most_anomalous_rows_sod
import Extremes

if __name__ == '__main__':
    # Generate the data
    # X, y = Data.mnist.get_data(main_class=2, anomaly_class=4, num_main=99, num_anomalies=1, anomaly_place=3) #ii stands a bit
    # X, y = Data.mnist.get_data(main_class=7, anomaly_class=4, num_main=99, num_anomalies=1, anomaly_place=3)# ii a bit better
    # X, y = Data.mnist.get_data(main_class=1, anomaly_class=3, num_main=99, num_anomalies=1, anomaly_place=3) #ii a bit better
    # X, y = Data.mnist.get_data(main_class=9, anomaly_class=2, num_main=99, num_anomalies=1, anomaly_place=3) #same for all
    X, y = Data.segmentation.get_data(main_class=2, anomaly_class=4, num_main=20, num_anomalies=1, anomaly_place=3)

    # Define methods and their corresponding functions
    methods = {
        "COF": find_most_anomalous_rows_cof,
        "k-NN": find_most_anomalous_rows_knn,
        "LOF": find_most_anomalous_rows_lof,
        "HBOS": find_most_anomalous_rows_hbos,
        "LODA": find_most_anomalous_rows_loda,
        "iForest": find_most_anomalous_rows_iforest,
        "ABOD": find_most_anomalous_rows_abod,
        "Isolation Index": Extremes.extreme,
        "SOD": find_most_anomalous_rows_sod,
    }

    # Find the most anomalous rows using k-NN
    knn_anomalies, knn_scores = find_most_anomalous_rows_knn(X, k=10, n_anomalous=99)

    # Select the last 9 (most regular) samples
    most_regular_samples = knn_anomalies[-9:]

    # Plot the last 9 samples (most regular) in a 3x3 grid
    fig_samples, axes_samples = plt.subplots(3, 3, figsize=(10, 10))
    for i, row in enumerate(most_regular_samples):
        if row.size != 784:  # Ensure row is reshaped only when it's a valid image
            continue
        sample = row.reshape((28, 28))
        ax = axes_samples[i // 3, i % 3]
        ax.imshow(sample, cmap='gray')
        ax.set_title(f"sample {i + 1}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    # Plot anomalies in a 3x3 grid
    fig_anomalies, axes_anomalies = plt.subplots(3, 3, figsize=(10, 10))

    # Plot most anomalous image for each method
    for i, (method, func) in enumerate(methods.items()):
        if i >= 9:
            break  # Break the loop if we exceed 9 methods
        # if method == "iForest":
        #     row, _ = func(X, k=10, n_anomalous=1)  # Handle Isolation Forest to get anomalies
        elif method == "Isolation Index":
            row = func(X)[1][0]  # Handle "Isolation Index" method to get anomalies
        else:
            row, _ = func(X, k=20, n_anomalous=1)
            if isinstance(row, list) and len(row) > 0:
                row = row[0]  # Ensure row is a single anomaly if returned in a list
        if row.size != 784:  # Ensure row is reshaped only when it's a valid image
            continue
        X_reshaped = row.reshape((28, 28))
        ax = axes_anomalies[i // 3, i % 3]
        ax.imshow(X_reshaped, cmap='gray')

        # Enhance "Isolation Index" method's subplot
        if method == "Isolation Index":
            ax.set_title(f"{method}\nAnomaly", fontsize=14, fontweight='bold')  # Increase font size and make bold
            # Draw a red square around the subplot
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(2)

        else:
            ax.set_title(f"{method}\nAnomaly")

        ax.axis('off')

    plt.tight_layout()
    plt.show()
