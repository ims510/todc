import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate data for the first cluster (left cloud)
cluster1_x = np.random.normal(loc=-4, scale=0.8, size=300)
cluster1_y = np.random.normal(loc=0, scale=0.8, size=300)

# Generate data for the second cluster (right cloud)
cluster2_x = np.random.normal(loc=4, scale=0.8, size=300)
cluster2_y = np.random.normal(loc=0, scale=0.8, size=300)

# Assign colors to the points
colors1 = ['green'] * 200 + ['red'] * 100  # First cluster: 2/3 green, 1/3 red
colors2 = ['blue'] * 200 + ['red'] * 100  # Second cluster: 2/3 blue, 1/3 red

# Adjust the red points to be closer to each other
cluster1_x[-100:] += 1.5  # Shift red points in the first cluster slightly to the right
cluster2_x[-100:] -= 1.5  # Shift red points in the second cluster slightly to the left


# Plot the first cluster
plt.scatter(cluster1_x, cluster1_y, c=colors1, alpha=0.7)

# Plot the second cluster
plt.scatter(cluster2_x, cluster2_y, c=colors2, alpha=0.7)

# Add labels and legend
plt.title("Clusters with Shared Features (Red)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.grid(True)

# Show the plot
plt.show()