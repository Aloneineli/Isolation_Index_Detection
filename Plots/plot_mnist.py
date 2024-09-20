import numpy as np
from matplotlib import pyplot as plt
import Data.mnist
from algorithms_1.cof import find_most_anomalous_rows_cof, find_optimal_k_cof
from algorithms_1.abod import find_most_anomalous_rows_abod, find_optimal_k_abod
from algorithms_1.hbos import find_most_anomalous_rows_hbos, find_optimal_k_hbos
from algorithms_1.isolation_forest import find_most_anomalous_rows_iforest, find_optimal_k_iforest
from algorithms_1.knn import find_most_anomalous_rows_knn, find_optimal_k_knn
from algorithms_1.loda import find_most_anomalous_rows_loda, find_optimal_k_loda
from algorithms_1.lof import find_most_anomalous_rows_lof, find_optimal_k_lof
from algorithms_1.sod import find_most_anomalous_rows_sod, find_optimal_k_sod
import Extremes

if __name__ == '__main__':
    for main_cl, ano_cl in [(8, 9), (0, 1), (5, 3), (1, 7), (6, 7), (2, 3), (0, 8)]:
        X, y = Data.mnist.get_data(main_class=main_cl, anomaly_class=ano_cl, num_main=99, num_anomalies=1, anomaly_place=3)

        # Define methods and their corresponding functions
        methods = {
            "COF": [find_most_anomalous_rows_cof, find_optimal_k_cof],
            "k-NN": [find_most_anomalous_rows_knn, find_optimal_k_knn],
            "LOF": [find_most_anomalous_rows_lof, find_optimal_k_lof],
            "HBOS": [find_most_anomalous_rows_hbos, find_optimal_k_hbos],
            "LODA": [find_most_anomalous_rows_loda, find_optimal_k_loda],
            "iForest": [find_most_anomalous_rows_iforest, find_optimal_k_iforest],
            "ABOD": [find_most_anomalous_rows_abod, find_optimal_k_abod],
            "Isolation Index": [Extremes.extreme, 0.0],
            "SOD": [find_most_anomalous_rows_sod, find_optimal_k_sod],
        }

        # Find the most anomalous rows using k-NN
        knn_anomalies, knn_scores = find_most_anomalous_rows_knn(X, k=10, n_anomalous=99)

        # Select the last 9 (most regular) samples
        most_regular_samples = knn_anomalies[-9:]

        # Plot the last 9 samples (most regular) in a 3x3 grid
        fig_samples, axes_samples = plt.subplots(3, 3, figsize=(10, 10))
        fig_samples.suptitle("Normal Samples", fontsize=16)
        for i, row in enumerate(most_regular_samples):
            if row.size != 784:  # Ensure row is reshaped only when it's a valid image
                continue
            sample = row.reshape((28, 28))
            ax = axes_samples[i // 3, i % 3]
            ax.imshow(sample, cmap='gray')
            ax.set_title(f"sample {i + 1}")
            ax.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'samples - main class {main_cl} anomaly {ano_cl}.png')

        # Plot anomalies in a 3x3 grid
        fig_anomalies, axes_anomalies = plt.subplots(3, 3, figsize=(10, 10))
        fig_anomalies.suptitle("Anomalies", fontsize=16)

        # Plot most anomalous image for each method
        for i, (method, funcs) in enumerate(methods.items()):
            if i >= 9:
                break  # Break the loop if we exceed 9 methods
            elif method == "Isolation Index":
                row = funcs[0](X)[1][0]  # Handle "Isolation Index" method to get anomalies
            else:
                best_k_, _ = funcs[1](X, [3, 5, 7, 10, 20, 30], n_anomalous=1)
                row, _ = funcs[0](X, k=best_k_, n_anomalous=1)
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

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'anomalies - main class {main_cl} anomaly {ano_cl}.png')
        print(f'done {main_cl} anomaly {ano_cl}.png')
