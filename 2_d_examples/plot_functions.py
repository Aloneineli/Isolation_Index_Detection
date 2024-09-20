import matplotlib.pyplot as plt
from synthetic_2d_data import *
import Extremes


def plot_points(dictionary, n, d, avg):
	total_elements = sum(len(v) for v in dictionary.values())
	plt.figure(figsize=(10, 5))
	for group_num, i in zip(list(dictionary.keys()), list(range(0, total_elements))):
		if group_num == 1:
			elem_in_group = 3
			shift = np.array([5, 5])
			shifts = (shift,)
			points = generate_all_2_d_synthetic_points(n, d, avg,
			                                           groups=len(shifts),
			                                           shifts=shifts,
			                                           num_each_group=elem_in_group)
			last_k_points = len(shifts) * elem_in_group
			scores = Extremes.extreme(points)
			idx_sorted = get_iso_from_low_to_high(scores)

		elif group_num == 2:
			for elem_in_group in dictionary[2]:
				shift_2_1 = np.array([5, 5])
				shift_2_2 = np.array([-5, 5])
				shifts = (shift_2_1, shift_2_2)
				points = generate_all_2_d_synthetic_points(n, d, avg,
				                                           groups=len(shifts),
				                                           shifts=shifts,
				                                           num_each_group=elem_in_group)
				last_k_points = len(shifts) * elem_in_group
				scores = Extremes.extreme(points)
				idx_sorted = get_iso_from_low_to_high(scores)
		elif group_num == 3:
			for elem_in_group in dictionary[3]:
				shift_3_1 = np.array([5, 5])
				shift_3_2 = np.array([-5, 5])
				shift_3_3 = np.array([-5, -5])
				shifts = (shift_3_1, shift_3_2, shift_3_3)
				points = generate_all_2_d_synthetic_points(n, d, avg,
				                                           groups=len(shifts),
				                                           shifts=shifts,
				                                           num_each_group=elem_in_group)
				last_k_points = len(shifts) * elem_in_group
				scores = Extremes.extreme(points)
				idx_sorted = get_iso_from_low_to_high(scores)
		else:
			print('problem')

		plt.subplot(1, 3, i + 1)
		normal_points = points[:-last_k_points]
		anomaly_points = points[-last_k_points:]
		plt.scatter(normal_points[:, 0], normal_points[:, 1], c='b', label='Normal Points')
		plt.scatter(anomaly_points[:, 0], anomaly_points[:, 1], c='r', label='Anomaly Points')
		for idx_plot, idx in enumerate(idx_sorted):
			plt.text(points[idx, 0], points[idx, 1], str(idx_plot + 1), color='green')
		plt.ylim(-8.0, 8.0)
		plt.xlim(-8.0, 8.0)
		plt.title(f'({group_num})')
		plt.legend(loc='lower right', fontsize='xx-small')
	plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.45)
	plt.show()
			
				
				



if __name__ == '__main__':
	dictionary = {1: [1, 3, 5], 2: [1], 3: [1]}
	n = 30
	d = 2
	avg = 0.0
	plot_points(dictionary, n, d, avg)
	