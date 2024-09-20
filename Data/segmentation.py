import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Fetch the Image Segmentation dataset
image_segmentation = fetch_ucirepo(id=50)

# Data (as pandas dataframes)
X = image_segmentation.data.features
y = image_segmentation.data.targets

def get_data(main_class, anomaly_class, num_main=20, num_anomalies=1, anomaly_place=1):
    # Ensure the target labels are in a proper format (flatten if necessary)
    if isinstance(y, pd.DataFrame):
        y_series = y.iloc[:, 0]  # Assuming y has a single column
    else:
        y_series = pd.Series(y)

    # Convert target labels to numeric
    y_numeric = pd.factorize(y_series)[0]

    # Initialize lists to hold the selected data
    X_selected = []
    y_selected = []

    # Loop through each class and select the first 100 instances
    for class_label in np.unique(y_numeric):
        # Find the indices of the current class
        class_indices = np.where(y_numeric == class_label)[0]

        # Select the first 100 instances of the current class
        selected_indices = class_indices[:num_main]

        # Add the selected data to the lists
        X_selected.append(X.iloc[selected_indices])
        y_selected.append(y_numeric[selected_indices])

    # Convert lists to arrays and concatenate
    X_selected = pd.concat(X_selected).to_numpy()
    y_selected = np.concatenate(y_selected)

    # Select the main class and anomaly class data
    final_class_indices_main = np.where(y_selected == main_class)[0]
    final_selected_indices_main = final_class_indices_main[:num_main]
    final_x_selected_main = X_selected[final_selected_indices_main]
    final_y_selected_main = y_selected[final_selected_indices_main]

    final_class_indices_anomaly = np.where(y_selected == anomaly_class)[0]
    final_selected_indices_anomaly = final_class_indices_anomaly[anomaly_place - 1:anomaly_place - 1 + num_anomalies]
    final_x_selected_anomaly = X_selected[final_selected_indices_anomaly]
    final_y_selected_anomaly = y_selected[final_selected_indices_anomaly]

    # Concatenate the main class and anomaly data
    out_x = np.concatenate((final_x_selected_main, final_x_selected_anomaly), axis=0)
    out_y = np.concatenate((final_y_selected_main, final_y_selected_anomaly), axis=0)
    return out_x, out_y

# Example usage
main_class = 0  # Assuming 'brickface' is encoded as 0
anomaly_class = 1  # Assuming 'sky' is encoded as 1
X_out, y_out = get_data(main_class, anomaly_class, num_main=100, num_anomalies=1, anomaly_place=1)

print("Selected data shape:", X_out.shape)
print("Selected labels shape:", y_out.shape)
