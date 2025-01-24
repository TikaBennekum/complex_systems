# Existing code
from CA import *
from initial_state_generation import generate_initial_slope
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Define the grid dimensions and initial ground height
    width, height, ground_height = 21, 101, 101 * 0.1

    # Generate the initial terrain with a slope and some noise
    initial_state = generate_initial_slope(height, width, ground_height, noise_amplitude=0.1, noise_type='white')

    # Perturbed terrain: Add a small bump under the water source
    perturbed_state = initial_state.copy()
    perturbed_state[1, width // 2, GROUND_HEIGHT] += -0.1  # Add a small bump/hole to the terrain

    # Visualize the initial terrain difference
    terrain_difference_initial = perturbed_state[:, :, GROUND_HEIGHT] - initial_state[:, :, GROUND_HEIGHT]

    # Plot the initial terrain differences before simulation
    plt.figure(figsize=(12, 5))
    # [Your existing plotting code here]
    plt.savefig('data/initial_terrain_comparison.png')
    plt.show()

    # First simulation: Unmodified terrain
    ca1 = CA(width, height, initial_state.copy(), neighbor_list=BOTTOM_NEIGHBORS)
    ca1.grid[0, width // 2, WATER_HEIGHT] = 50  # Water source at the center
    water_heights_unmodified = []  # Store water heights for each timestep
    for _ in range(2000):
        ca1.update_grid()
        water_heights_unmodified.append(ca1.grid[:, :, WATER_HEIGHT].copy())

    # Second simulation: Slight terrain change under the water source
    ca2 = CA(width, height, perturbed_state, neighbor_list=BOTTOM_NEIGHBORS)
    ca2.grid[0, width // 2, WATER_HEIGHT] = 50  # Same water source position
    water_heights_modified = []  # Store water heights for each timestep
    for _ in range(2000):
        ca2.update_grid()
        water_heights_modified.append(ca2.grid[:, :, WATER_HEIGHT].copy())

    # Calculate water height differences for each timestep
    water_height_differences = []
    for t in range(2000):
        difference = water_heights_unmodified[t] - water_heights_modified[t]
        water_height_differences.append(difference)

    # Visualize the differences over time (for a few selected timesteps)
    plt.figure(figsize=(12, 5))
    for t in range(0, 2000, 400):  # Plot every 400 timesteps
        plt.subplot(1, 5, t // 400 + 1)
        plt.title(f"Water Height Difference at Timestep {t}")
        plt.imshow(water_height_differences[t], cmap="coolwarm", interpolation="none")
        plt.colorbar(label="Height Difference")
    plt.tight_layout()
    plt.savefig('data/water_height_differences_over_time.png')
    plt.show()
