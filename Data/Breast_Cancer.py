from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd

def get_breast_cancer():
    # fetch dataset
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

    # data (as pandas dataframes)
    X = breast_cancer_wisconsin_diagnostic.data.features
    y = breast_cancer_wisconsin_diagnostic.data.targets
    return X, y


def get_data(main_class='B', anomaly_class='M', num_main=100, num_anomalies=1, anomaly_place=1):
    X, y = get_breast_cancer()


    # Initialize lists to hold the selected data
    X_selected = []
    y_selected = []

    for digit in ['B', 'M']:
        # Find the indices of the current digit
        digit_indices = np.where(y == digit)[0]


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
