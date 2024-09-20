import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
	ps = [1, 5, 10]
	ds = [2, 3, 5, 7, 9]
	df = pd.read_csv('Mahalanobis_distance_comparison.csv')
	plt.figure(figsize=(17, 6))
	for i, match in enumerate([1, 5, 10]):
		plt.subplot(1, 3, i + 1)
		df_temp = df[df['Matching Points'] == match]
		for d in ds:
			df_temp_d = df_temp[df_temp['d'] == d]
			x_vals = df_temp_d['N']
			y_vals = df_temp_d['Success']
			plt.plot(x_vals, y_vals, label=f'd={d}')
		plt.xlabel('N')
		plt.ylabel('Success')
		plt.title(f'Success Rate For Matching Points={match}')
		plt.legend()
	plt.show()
			
			
			
		