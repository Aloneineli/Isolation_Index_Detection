import numpy as np


def generate_main_points(n, d, mean, stddev=1):
	initial_points = np.random.normal(mean, stddev, (n, d))
	return initial_points


def generate_anomalies(groups, shifts, num_each_group, d, stddev=0.5):
	n_points = num_each_group * groups
	out = np.zeros((n_points, d), dtype='float32')
	for gr_idx in range(groups):
		points = np.random.normal(shifts[gr_idx], stddev, (num_each_group, d))
		out[gr_idx * num_each_group:gr_idx * num_each_group + num_each_group, :] = points
	return out


def generate_all_2_d_synthetic_points(n, d, mean, groups, shifts, num_each_group, stddev_normal=1, stddev_anomaly=0.5):
	initial_points = np.random.normal(mean, stddev_normal, (n, d))
	anomaly_points = generate_anomalies(groups, shifts, num_each_group, d, stddev_anomaly)
	out = np.concatenate([initial_points, anomaly_points])
	num_anomaly = num_each_group * groups
	return out


def get_iso_from_low_to_high(scores):
	not_infinite_mask = np.isfinite(scores)
	non_inf_indices = np.arange(len(scores))[not_infinite_mask]
	sorted_indices_by_value = non_inf_indices[np.argsort(scores[not_infinite_mask])]
	return sorted_indices_by_value


if __name__ == '__main__':
	n = 30
	d = 2
	groups = 3
	shift_1 = np.array([100, 100])
	shift_2 = np.array([200, 200])
	shift_3 = np.array([300, 300])
	mean = 0
	num_each_group = 5
	e = generate_all_2_d_synthetic_points(n, d, mean,
	                                      groups=3,
	                                      shifts=(shift_1, shift_2, shift_3),
	                                      num_each_group=num_each_group)
