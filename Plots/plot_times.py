if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd
    import os

    plt.figure(figsize=(6, 4))
    path = os.path.join('..', 'Times', 'Times_9_methods_big_ns_for_plot.csv')
    df = pd.read_csv(path)
    df = df.reset_index(drop=True).iloc[2:]
    n = df['N']
    knn = df['KNN']
    lof = df['LOF']
    i_f = df['Isolation Forest']
    loda = df['LODA']
    hbos = df['HBOS']
    abod = df['ABOD']
    sod = df['SOD']
    cof = df['COF']
    i_i = df['Isolation Index']
    avg = df['Avg']

    for algo in [('KNN', knn, (0, (1, 1)), 'black'),
                 ('LOF', lof, (0, (1, 1)), 'blue'),
                 ('COF', cof, (0, (1, 1)), 'yellow'),
                 ('Isolation Forest', i_f, (0, (3, 1, 1, 1, 1, 1)), 'brown'),
                 ('LODA', loda, 'dotted', 'purple'),
                 ('HBOS', hbos, (0, (5, 1)), 'green'),
                 ('ABOD', abod, (0, (3, 1, 1, 1)), 'orange'),
                 ('SOD', sod, (0, (3, 1, 1, 1, 1, 1)), 'grey'),
                 ('Isolation Index', i_i, 'solid', 'red')]:
        plt.plot(n, algo[1], label=algo[0], linestyle=algo[2], color=algo[3])

    # Add the avg series with a thicker black line and no label
    plt.plot(n, avg, linestyle='solid', color='black', linewidth=2)

    # Add text near the avg line
    text_x = n.iloc[-1]  # X-coordinate for the text
    text_y = avg.iloc[-1]  # Y-coordinate for the text
    plt.text(text_x, text_y, 'Avg', color='black', verticalalignment='bottom', horizontalalignment='right')

    plt.xlabel('N')
    plt.ylabel('Seconds')
    plt.title('Complexity Times')
    plt.legend(loc='upper left', fontsize='small')
    plt.savefig('Times_9_methods_big_ns.png')
