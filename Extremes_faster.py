import numpy as np
import matplotlib.pyplot as plt
import time


def find_extreme_points(m):
    start_time = time.time()

    # Step 1: Center the points by subtracting the mean of each dimension
    m_mean = np.mean(m, axis=0)
    m_centered = m - m_mean

    # Step 2: Calculate norms (self-projections) for all points
    norms = np.linalg.norm(m_centered, axis=1)

    # Step 3: Sort indices by descending norm values
    sorted_indices = np.argsort(-norms)
    available = np.ones(len(m_centered), dtype=bool)

    extreme_points = []
    total_time_projection = 0
    # Main loop to find extreme points
    while np.any(available):
        # Step 4: Find the top available point (un-normalized)
        top_idx = sorted_indices[available[sorted_indices]][0]
        top_point = m_centered[top_idx]
        extreme_points.append(top_point)

        # Step 5: Project the top point onto all other remaining points (except itself)
        remaining_indices = np.where(available)[0]
        remaining_indices = remaining_indices[remaining_indices != top_idx]

        # Step 6: Project the top point onto the remaining points
        start_time = time.time()
        projections = np.dot(m_centered[remaining_indices], top_point) / norms[remaining_indices]
        total_time_projection += time.time() - start_time

        # Step 7: Eliminate points where the projection is larger than their own norm
        available[remaining_indices] = projections < norms[remaining_indices]

        # Mark the current top point as used
        available[top_idx] = False

    print(f"Total time for projecting top points: {total_time_projection:.6f} seconds")
    # Step 8: Add the mean back to all extreme points
    print(len(extreme_points))
    extreme_points = np.array(extreme_points) + m_mean

    return extreme_points


# Generate 100 points with mean zero and std of 3
mean = 0
std_dev = 3
num_points = 8000
points = np.random.normal(mean, std_dev, (num_points, 20))  # 2D points for easy plotting

# Measure time to find extreme points
start_time = time.time()
extreme_points = find_extreme_points(points)
print(f"Total time to find extreme points: {time.time() - start_time:.6f} seconds")

# Plot all points
plt.figure(figsize=(10, 6))
plt.scatter(points[:, 0], points[:, 1], color='blue', label='Regular Points')  # Regular points in blue
plt.scatter(extreme_points[:, 0], extreme_points[:, 1], color='red', label='Extreme Points')  # Extreme points in red

# Adding labels and legend
plt.title('Points with Extreme Points Highlighted')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.axhline(0, color='gray', lw=0.5, ls='--')  # x-axis
plt.axvline(0, color='gray', lw=0.5, ls='--')  # y-axis
plt.legend()
plt.grid()
plt.show()
