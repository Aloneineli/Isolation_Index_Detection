import math
import time
import numpy as np
import random
import pandas as pd
from matplotlib import pyplot as plt
import Data.mnist
import Data.Breast_Cancer
import Data.Pen_Writing
from algorithms_1.cof import find_most_anomalous_rows_cof
from algorithms_1.abod import find_most_anomalous_rows_abod
from algorithms_1.hbos import find_most_anomalous_rows_hbos
from algorithms_1.isolation_forest import find_most_anomalous_rows_iforest
from algorithms_1.knn import find_most_anomalous_rows_knn
from algorithms_1.loda import find_most_anomalous_rows_loda
from algorithms_1.lof import find_most_anomalous_rows_lof
from algorithms_1.sod import find_most_anomalous_rows_sod
import Extremes
import Extremes_new


def run_experiment(data, epochs, num_main_values):
    # Initialize results dictionary with keys as method names and values as dictionaries to store results for each k
    results = {method_: {k: [] for k in range(3, 100, 6)} for method_ in
               ["COF", "k-NN", "LOF", "HBOS", "LODA", "iForest", "ABOD", "Isolation Index", "SOD"]}

    for num_main in num_main_values:
        for i in range(epochs):
            start = time.time()
            print(f"num_main: {num_main}, Iteration {i + 1}/{epochs}")
            if (data == 'mnist') or (data == 'pen_writing'):
                main_cl = random.randint(0, 9)
                ano_cl = random.choice([i for i in range(10) if i != main_cl])
                anomaly_place = random.randint(1, 50)
                if data == 'mnist':
                    X, y = Data.mnist.get_data(main_class=main_cl, anomaly_class=ano_cl, num_main=num_main,
                                               num_anomalies=math.floor(num_main/100),
                                               anomaly_place=anomaly_place)
                else:
                    X, y = Data.Pen_Writing.get_data(main_class=main_cl, anomaly_class=ano_cl, num_main=num_main,
                                                     num_anomalies=math.floor(num_main/100),
                                                     anomaly_place=anomaly_place)
            elif data == 'breast_cancer':
                main_cl = 'B'
                ano_cl = 'M'
                anomaly_place = random.randint(1, 50)
                X, y = Data.Breast_Cancer.get_data(main_class=main_cl, anomaly_class=ano_cl, num_main=num_main,
                                                   num_anomalies=math.floor(num_main/100),
                                                   anomaly_place=anomaly_place)

            methods = {
                "COF": find_most_anomalous_rows_cof,
                "k-NN": find_most_anomalous_rows_knn,
                "LOF": find_most_anomalous_rows_lof,
                "HBOS": find_most_anomalous_rows_hbos,
                "LODA": find_most_anomalous_rows_loda,
                "iForest": find_most_anomalous_rows_iforest,
                "ABOD": find_most_anomalous_rows_abod,
                "Isolation Index": Extremes_new.extreme,
                "SOD": find_most_anomalous_rows_sod,
            }

            for k in range(3, 100, 6):
                for method, func in methods.items():
                    if method == "Isolation Index":
                        _, rows = func(X)  # Handle "Isolation Index" method to get anomalies
                        row = rows[0]
                        row = row[np.newaxis, :]
                    else:
                        row, _ = func(X, k=k, n_anomalous=1)
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
                    if method == 'Isolation Index':
                        row_index = np.where((X == row))[0][0]
                    else:
                        row_index = np.where((X == row).all(axis=1))[0][0]
                    is_anomaly = y[row_index] == ano_cl
                    results[method][k].append(is_anomaly)
            end = time.time()
            print(f'took {end - start} seconds')

    # Compute average success rate for each method and k
    averages = {method: {k: np.mean(results[method][k]) * 100 for k in range(3, 100, 6)} for method in results}

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
    plt.xlabel('k value')
    plt.ylabel('Average success rate (%)')
    plt.title(title)
    plt.legend()
    # plt.grid(True)

    # Save the plot as an image
    # plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()


if __name__ == "__main__":
    epochs = 5

    num_main_values_dict = {
        'breast_cancer': [1000],
        'pen_writing': [1000],
        'mnist': [150]
    }
    x_ticks_dict = {
        'breast_cancer': range(0, 101, 10),
        'pen_writing': range(0, 101, 10),
        'mnist': range(0, 101, 10)
    }

    for data in ['Breast Cancer', 'MNIST', 'Pen Writing']:
        if data == 'MNIST':
            data_temp = 'mnist'
        elif data == 'Pen Writing':
            data_temp = 'pen_writing'
        elif data == 'Breast Cancer':
            data_temp = 'breast_cancer'

        for num_main in num_main_values_dict[data_temp]:
            df_temp = run_experiment(data=data_temp, epochs=epochs, num_main_values=[num_main])
            print(f"Results for {data} with num_main={num_main}:")
            print(df_temp)
            df_temp.to_csv(f"experiment_results_{data}_with sample_of_{num_main}_avg_of_{epochs}.csv", index=True)
            plot_results(df_temp, f"{data} Data", x_ticks_dict[data_temp])
