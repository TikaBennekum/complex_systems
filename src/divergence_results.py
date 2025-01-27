import numpy as np
import matplotlib.pyplot as plt

# Load the difference matrices
mean_water_diff = np.load('data/mean_water_height_diff.npy')
mean_ground_diff = np.load('data/mean_ground_height_diff.npy')

# Taking the logarithm of the mean differences
log_mean_water_diff = np.log(mean_water_diff + 1e-10)  # Adding a small value to avoid log(0)
log_mean_ground_diff = np.log(mean_ground_diff + 1e-10)  # Adding a small value to avoid log(0)

# Truncate the log differences to the first 200 time steps for Lyapunov exponent computation
log_mean_ground_diff_trunc = log_mean_ground_diff[:200]
log_mean_water_diff_trunc = log_mean_water_diff[:200]

# Generate time steps for truncated range and full range
time_steps_trunc = np.arange(len(log_mean_ground_diff_trunc))  # Truncated time steps
time_steps_full = np.arange(len(mean_water_diff))  # Full range time steps

# Fit a linear model to the truncated log differences
slope_water, intercept_water = np.polyfit(time_steps_trunc, log_mean_water_diff_trunc, 1)
slope_ground, intercept_ground = np.polyfit(time_steps_trunc, log_mean_ground_diff_trunc, 1)

# The slopes are estimates of the Lyapunov exponents
lyapunov_exponent_water = slope_water
lyapunov_exponent_ground = slope_ground

print(f"Lyapunov Exponent for Water Height (Truncated): {lyapunov_exponent_water}")
print(f"Lyapunov Exponent for Ground Height (Truncated): {lyapunov_exponent_ground}")

# Create subplots for water height difference and terrain height difference
fig, axs = plt.subplots(2, 1, figsize=(8, 8))

# Subplot for Water Height Difference
axs[0].plot(time_steps_full, mean_water_diff, color="blue", label="Water Height Difference")
axs[0].set_title("Water Height Difference Over Time")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Mean Absolute Difference")
axs[0].legend()
axs[0].grid(True)

# Subplot for Terrain Height Difference
axs[1].plot(time_steps_full, mean_ground_diff, color="green", label="Ground Height Difference")
axs[1].set_title("Ground Height Difference Over Time")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Mean Absolute Difference")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
