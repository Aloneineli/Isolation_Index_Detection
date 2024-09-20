import numpy as np
import matplotlib.pyplot as plt
import Extremes
import generate_points


def plot_points_2d(points, mark_extreme=False):
	plt.scatter(points[:, 0], points[:, 1], color='blue')
	if mark_extreme:
		plt.scatter(points[0, 0], points[0, 1], color='red')  # First point in red
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.xlim(-4, 4)
	plt.ylim(-4, 4)
	plt.axhline(0, color='black', linewidth=0.5)
	plt.axvline(0, color='black', linewidth=0.5)
	plt.title('2D Points with Standard Normal Distribution')
	plt.show()



if __name__ == '__main__':
	m = generate_points.generate_standard_gauss_points(128)
	print(m)
	plot_points_2d(m)
	extreme_points = Extremes.extreme(m)[1]
	plot_points_2d(extreme_points, mark_extreme=True)
