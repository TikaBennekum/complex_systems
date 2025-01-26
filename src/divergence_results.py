import numpy as np
import matplotlib.pyplot as plt

# Load the difference matrices
mean_water_diff = np.load('data/mean_water_height_diff.npy')
mean_ground_diff = np.load('data/mean_ground_height_diff.npy')

# Taking the logarithm of the mean differences
log_mean_water_diff = np.log(mean_water_diff + 1e-10)  # Adding a small value to avoid log(0)
log_mean_ground_diff = np.log(mean_ground_diff + 1e-10)  # Adding a small value to avoid log(0)

# Generate time steps
time_steps = np.arange(len(mean_water_diff))

# Fit a linear model to the log differences
slope_water, intercept_water = np.polyfit(time_steps, log_mean_water_diff, 1)
slope_ground, intercept_ground = np.polyfit(time_steps, log_mean_ground_diff, 1)

# The slopes are estimates of the Lyapunov exponents
lyapunov_exponent_water = slope_water
lyapunov_exponent_ground = slope_ground

print(f"Lyapunov Exponent for Water Height: {lyapunov_exponent_water}")
print(f"Lyapunov Exponent for Ground Height: {lyapunov_exponent_ground}")


plt.plot(mean_water_diff, label="Water Height Difference")
plt.plot(mean_ground_diff, label="Ground Height Difference")
plt.xlabel("Epoch")
plt.ylabel("Mean Absolute Difference")
plt.title("Differences in Water and Ground Height Over Time")
plt.xlim(0, 1000)
plt.yscale('log')  
plt.legend()
plt.show()
