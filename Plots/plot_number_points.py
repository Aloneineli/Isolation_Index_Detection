import numpy as np
import random
import pandas as pd
from matplotlib import pyplot as plt
import Data.mnist
import Data.Breast_Cancer
import Data.Pen_Writing
from algorithms_1.cof import find_most_anomalous_rows_cof, find_optimal_k_cof
from algorithms_1.abod import find_most_anomalous_rows_abod, find_optimal_k_abod
from algorithms_1.hbos import find_most_anomalous_rows_hbos, find_optimal_k_hbos
from algorithms_1.isolation_forest import find_most_anomalous_rows_iforest, find_optimal_k_iforest
from algorithms_1.knn import find_most_anomalous_rows_knn, find_optimal_k_knn
from algorithms_1.loda import find_most_anomalous_rows_loda, find_optimal_k_loda
from algorithms_1.lof import find_most_anomalous_rows_lof, find_optimal_k_lof
from algorithms_1.sod import find_most_anomalous_rows_sod, find_optimal_k_sod
import Extremes


def run_experiment(data, epochs, num_main_values):
    # Initialize results dictionary with keys as method names and values as dictionaries to store results for each num_main
    results = {method_: {num_main: [] for num_main in num_main_values} for method_ in
               ["COF", "k-NN", "LOF", "HBOS", "LODA", "iForest", "ABOD", "Isolation Index", "SOD"]}

    for num_main in num_main_values:
        for i in range(epochs):
            print(f"num_main: {num_main}, Iteration {i + 1}/{epochs}")
            if (data == 'mnist') or (data == 'pen_writing'):
                main_cl = random.randint(0, 9)
                ano_cl = random.choice([i for i in range(10) if i != main_cl])
                anomaly_place = random.randint(1, 180)
                if data == 'mnist':
                    X, y = Data.mnist.get_data(main_class=main_cl, anomaly_class=ano_cl, num_main=num_main,
                                               num_anomalies=1,
                                               anomaly_place=anomaly_place)
                else:
                    X, y = Data.Pen_Writing.get_data(main_class=main_cl, anomaly_class=ano_cl, num_main=num_main,
                                                     num_anomalies=1,
                                                     anomaly_place=anomaly_place)
            elif data == 'breast_cancer':
                main_cl = 'B'
                ano_cl = 'M'
                anomaly_place = random.randint(1, 180)
                X, y = Data.Breast_Cancer.get_data(main_class=main_cl, anomaly_class=ano_cl, num_main=num_main,
                                                   num_anomalies=1,
                                                   anomaly_place=anomaly_place)

            methods = {
                "COF": (find_most_anomalous_rows_cof, find_optimal_k_cof),
                "k-NN": (find_most_anomalous_rows_knn, find_optimal_k_knn),
                "LOF": (find_most_anomalous_rows_lof, find_optimal_k_lof),
                "HBOS": (find_most_anomalous_rows_hbos, find_optimal_k_hbos),
                "LODA": (find_most_anomalous_rows_loda, find_optimal_k_loda),
                "iForest": (find_most_anomalous_rows_iforest, find_optimal_k_iforest),
                "ABOD": (find_most_anomalous_rows_abod, find_optimal_k_abod),
                "Isolation Index": (Extremes.extreme, None),  # No k for Isolation Index
                "SOD": (find_most_anomalous_rows_sod, find_optimal_k_sod),
            }

            for method, (func, find_optimal_k) in methods.items():
                if method == "Isolation Index":
                    _, rows = func(X)
                    row = rows[0]
                else:
                    best_k_, _ = find_optimal_k(X, [3, 5, 7, 10, 15, 20], n_anomalous=1)
                    row, _ = func(X, k=best_k_, n_anomalous=1)
                    if isinstance(row, list) and len(row) > 0:
                        row = row[0]
                if data == 'mnist':
                    if row.size != 784:
                        continue
                elif data == 'breast_cancer':
                    if row.size != 30:
                        continue
                elif data == 'pen_writing':
                    if row.size != 16:
                        continue

                row_index = np.where((X == row).all(axis=1))[0][0]
                is_anomaly = y[row_index] == ano_cl
                results[method][num_main].append(is_anomaly)

    # Compute average success rate for each method and num_main
    averages = {method: {num_main: np.mean(results[method][num_main]) * 100 for num_main in num_main_values} for method
                in results}

    # Convert results to DataFrame
    df_ = pd.DataFrame(averages)
    return df_


def plot_results(df, title, x_ticks):
    # Define line styles and colors
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '--']
    colors = ['b', 'g', 'm', 'c', 'y', 'k', 'orange', 'purple', 'b']

    # Start plotting
    plt.figure(figsize=(12, 8))

    for i, column in enumerate(df.columns):
        if column == "Isolation Index":
            plt.plot(df.index, df[column], linestyle='-', color='red', linewidth=2, label=column)
        else:
            plt.plot(df.index, df[column], line_styles[i % len(line_styles)], color=colors[i % len(colors)],
                     label=column)
    plt.xticks(ticks=x_ticks)  # Set x-ticks based on dataset
    plt.yticks(ticks=range(0, 101, 10))
    plt.xlabel('num main class')
    plt.ylabel('Average success rate (%)')
    plt.title(title)
    plt.legend()
    # plt.grid(True)

    # Save the plot as an image
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()


if __name__ == "__main__":
    epochs = 3

    # Define num_main_values for each dataset
    num_main_values_dict = {
        'breast_cancer': range(25, 350, 30),
        'pen_writing': range(25, 750, 50),
        'mnist': range(50, 3000, 250)
    }

    x_ticks_dict = {
        'breast_cancer': range(25, 350, 30),
        'pen_writing': range(25, 650, 50),
        'mnist': range(50, 3000, 250)
    }

    for data in ['Breast Cancer', 'Pen Writing', 'MNIST']:
        if data == 'MNIST':
            data_temp = 'mnist'
        elif data == 'Pen Writing':
            data_temp = 'pen_writing'
        elif data == 'Breast Cancer':
            data_temp = 'breast_cancer'

        df_temp = run_experiment(data=data_temp, epochs=epochs, num_main_values=num_main_values_dict[data_temp])
        print(df_temp)
        df_temp.to_csv(f"experiment_results_{data}_{epochs}_num_main_class.csv", index=True)
        plot_results(df_temp, f"{data} Data num main class", x_ticks_dict[data_temp])
