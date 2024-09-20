import Extremes
import Data.mnist
from matplotlib import pyplot as plt

if __name__ == '__main__':
    X, y = Data.mnist.get_data(main_class=2, anomaly_class=4, num_main=99, num_anomalies=1, anomaly_place=3)
    # X, y = Data.mnist.get_data(main_class=9, anomaly_class=2, num_main=100, num_anomalies=3)
    # X, y = Data.mnist.get_data(main_class=6, anomaly_class=2, num_main=100, num_anomalies=3)
    # X, y = Data.mnist.get_data(main_class=7, anomaly_class=4, num_main=100, num_anomalies=3)
    # X, y = Data.mnist.get_data(main_class=4, anomaly_class=5, num_main=100, num_anomalies=3)
    a, rows = Extremes.extreme(X)
    print(a)
    X_reshaped = rows[0].reshape((28, 28))
    plt.imshow(X_reshaped, cmap='gray')  # Use 'gray' colormap for a grayscale image
    plt.colorbar()  # Add a colorbar to show the intensity scale
    plt.title("Handwritten Digit Image")
    plt.show()