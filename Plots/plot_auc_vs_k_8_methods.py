import os.path
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = 'breast_cancer'
    path = os.path.join(os.getcwd())
    plt.figure(figsize=(12, 10))
    for idx, n in enumerate([1, 2, 3, 5]):
        p = os.path.join(path, f'comparing_auc_for_10_methods_{n}%_anomalies_in{data}_data.csv')
        df = pd.read_csv(p)
        n_neighbors = df['k']
        knn = df['knn']
        lof = df['lof']
        cof = df['cof']
        i_forest = df['i_forest']
        loda = df['loda']
        hbos = df['hbos']
        abod = df['abod']
        sod = df['sod']
        iso_index = df['iso_index']
        for algo in [('KNN', knn, (0, (1, 1)), 'black'),
                     ('LOF', lof, (0, (1, 1)), 'blue'),
                     ('COF', cof, (0, (1, 1)), 'yellow'),
                     ('Isolation Forest', i_forest, (0, (3, 1, 1, 1, 1, 1)), 'brown'),
                     ('LODA', loda, 'dotted', 'purple'),
                     ('HBOS', hbos, (0, (5, 1)), 'green'),
                     ('ABOD', abod, (0, (3, 1, 1, 1)), 'orange'),
                     ('SOD', sod, (0, (3, 1, 1, 1, 1, 1)), 'grey'),
                     ('Isolation Index', iso_index, 'solid', 'red')]:
            plt.subplot(3, 2, idx + 1)
            plt.plot(n_neighbors, algo[1], label=algo[0], linestyle=algo[2], color=algo[3])
            plt.xlabel('K')
            plt.ylabel('AUC')
            plt.ylim(0.0, 1.0)
            plt.title(f'{n}% Anomalies')
            plt.legend(loc='lower right', fontsize='xx-small')
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.45)
    plt.suptitle(' Pen Writing Data Set ', fontsize=25)
    plt.savefig('Pen_Writing_Data_Set_(1,2,3,5)%_anomalies_9_methods.png')
