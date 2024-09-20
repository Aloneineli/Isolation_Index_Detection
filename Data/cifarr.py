import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from keras.datasets import cifar10
import pandas as pd

def get_cifar_data(main_class, anomaly_class, num_main=100, num_anomalies=1, anomaly_place=1):
    # Load CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0).flatten()

    # Initialize lists to hold the selected data
    X_selected = []
    y_selected = []

    # Loop through each class (0-9) and select the first 100 instances
    for digit in range(10):
        # Find the indices of the current class
        digit_indices = np.where(y == digit)[0]

        # Select the first 100 instances of the current class
        selected_indices = digit_indices[:100]

        # Add the selected data to the lists
        X_selected.append(X[selected_indices])
        y_selected.append(y[selected_indices])

    # Convert lists to arrays and concatenate
    X_selected = np.concatenate(X_selected, axis=0)
    y_selected = np.concatenate(y_selected, axis=0)

    final_digit_indices_main = np.where(y_selected == main_class)[0]
    final_selected_indices_main = final_digit_indices_main[:num_main]
    final_x_selected_main = X_selected[final_selected_indices_main]
    final_y_selected_main = y_selected[final_selected_indices_main]

    final_digit_indices_anomaly = np.where(y_selected == anomaly_class)[0]
    final_selected_indices_anomaly = final_digit_indices_anomaly[anomaly_place - 1:anomaly_place - 1 + num_anomalies]
    final_x_selected_anomaly = X_selected[final_selected_indices_anomaly]
    final_y_selected_anomaly = y_selected[final_selected_indices_anomaly]

    out_x = np.concatenate((final_x_selected_main, final_x_selected_anomaly), axis=0)
    out_y = np.concatenate((final_y_selected_main, final_y_selected_anomaly), axis=0)
    return out_x, out_y

if __name__ == '__main__':
    # Generate the data
    X, y = get_cifar_data(main_class=1, anomaly_class=4, num_main=99, num_anomalies=1, anomaly_place=3)

    # Plot samples in a 3x3 grid
    fig_samples, axes_samples = plt.subplots(3, 3, figsize=(10, 10))
    samples_indices = np.random.choice(range(1, 100), 9, replace=False)
    for i, idx in enumerate(samples_indices):
        sample = X[idx].reshape((32, 32, 3))
        ax = axes_samples[i // 3, i % 3]
        ax.imshow(sample)
        ax.set_title(f"Sample {i + 1}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()
