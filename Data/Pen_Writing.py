from ucimlrepo import fetch_ucirepo
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import pandas as pd
from scipy.interpolate import interp1d

def get_pen_writing():
    # fetch dataset
    pen_based_recognition_of_handwritten_digits = fetch_ucirepo(id=81)

    # data (as pandas dataframes)
    X = pen_based_recognition_of_handwritten_digits.data.features
    y = pen_based_recognition_of_handwritten_digits.data.targets
    return X, y


def get_data(main_class, anomaly_class, num_main=100, num_anomalies=1, anomaly_place=1):
    X, y = get_pen_writing()

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



def plot_digit_as_image(row, image_size=100):
    # Extract x and y coordinates from the row
    x_coords = row[:8]
    y_coords = row[8:16]
    # label = row[16]

    # Normalize coordinates to fit within the image size
    x_coords = np.interp(x_coords, (0, 100), (0, image_size-1))
    y_coords = np.interp(y_coords, (0, 100), (0, image_size-1))

    # Interpolate to create a smooth path
    num_points = 100  # Number of points to interpolate
    t = np.linspace(0, 1, len(x_coords))
    t_interpolated = np.linspace(0, 1, num_points)

    x_interpolated = interp1d(t, x_coords, kind='linear')(t_interpolated)
    y_interpolated = interp1d(t, y_coords, kind='linear')(t_interpolated)

    # Create a blank image
    image = np.zeros((image_size, image_size))

    # Draw the path on the image
    for i in range(num_points - 1):
        x0, y0 = int(x_interpolated[i]), int(y_interpolated[i])
        x1, y1 = int(x_interpolated[i+1]), int(y_interpolated[i+1])
        image = draw_line(image, x0, y0, x1, y1)

    # Plot the image
    plt.imshow(image, cmap='gray')
    plt.title(f'Digit: h')
    plt.axis('off')
    plt.show()

def draw_line(image, x0, y0, x1, y1):
    """Draw a line on the image from (x0, y0) to (x1, y1) using Bresenham's algorithm."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        image[y0, x0] = 255
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return image

if __name__ == "__main__":
    main_cl = 2
    ano_cl = 4
    num_main = 99
    num_anomalies = 1
    anomaly_place = 3
    df = get_data(main_cl, ano_cl, num_main, num_anomalies, anomaly_place)
    example_row = list(df[0][5])
    plot_digit_as_image(example_row)