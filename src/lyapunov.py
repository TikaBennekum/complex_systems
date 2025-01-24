import numpy as np
import matplotlib.pyplot as plt
from CA import *
from initial_state_generation import generate_initial_slope

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

    # First simulation: Unmodified terrain
    ca1 = CA(width, height, initial_state.copy(), neighbor_list=BOTTOM_NEIGHBORS)
    ca1.grid[0, width // 2, WATER_HEIGHT] = 50  # Water source at the center
    ca1.run_simulation(num_epochs=1000, show_live=False, save_nth=1)  # Save every 10th frame
    unmodified_states = ca1.saved_grids  # Store saved states for unmodified terrain

    # Second simulation: Slight terrain change under the water source
    ca2 = CA(width, height, perturbed_state, neighbor_list=BOTTOM_NEIGHBORS)
    ca2.grid[0, width // 2, WATER_HEIGHT] = 50  # Same water source position
    ca2.run_simulation(num_epochs=1000, show_live=False, save_nth=1)  # Save every 10th frame
    modified_states = ca2.saved_grids  # Store saved states for modified terrain

    # Calculate the differences in water height for each saved state
    water_height_differences = []
    for t in range(unmodified_states.shape[0]):
        difference = unmodified_states[t, :, :, WATER_HEIGHT] - modified_states[t, :, :, WATER_HEIGHT]
        abs_difference = np.abs(difference)
        water_height_differences.append(np.mean(abs_difference))  # Mean difference across the grid

    # Plot the differences over time
    plt.figure(figsize=(10, 5))
    plt.plot(water_height_differences, label="Water Height Difference", color="blue")
    plt.title("Average Water Height Difference Over Time")
    plt.xlabel("Time Step (Every 10th Iteration)")
    plt.ylabel("Average Water Height Difference")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('data/water_height_difference_plot.png')
    plt.show()
    