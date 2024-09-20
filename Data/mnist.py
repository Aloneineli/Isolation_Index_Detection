import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as img
from sklearn.datasets import fetch_openml
import pandas as pd


def get_data(main_class, anomaly_class, num_main=100, num_anomalies=1, anomaly_place=1):
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist['data'], mnist['target'].astype(int)

    # Initialize lists to hold the selected data
    X_selected = []
    y_selected = []

    # Loop through each digit (0-9) and select the first 100 instances
    for digit in range(10):
        # Find the indices of the current digit
        digit_indices = np.where(y == digit)[0]

        # Select the first 100 instances of the current digit
        if num_main >= len(digit_indices):
            num_main = len(digit_indices)
        selected_indices = digit_indices[:num_main]

        # Add the selected data to the lists
        X_selected.append(X.iloc[selected_indices])
        y_selected.append(y.iloc[selected_indices])

    # Convert lists to arrays and concatenate
    X_selected = pd.concat(X_selected).to_numpy()
    y_selected = pd.concat(y_selected).to_numpy()

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



