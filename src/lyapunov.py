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

    # First simulation: Unperturbed system (warming up)
    ca1 = CA(width, height, initial_state.copy(), neighbor_list=BOTTOM_NEIGHBORS)
    ca1.grid[0, width // 2, WATER_HEIGHT] = 50  # Water source at the center
    ca1.run_simulation(num_epochs=120, show_live=False, save_nth=1)  # Run for 120 epochs to "warm up"
    warmed_up_states = ca1.saved_grids[-1]  # Use the last state after warming up as the starting point

    # Perturb the warmed-up state for the second system
    perturbed_initial_state = warmed_up_states.copy()
    perturbed_initial_state[1, width // 2, GROUND_HEIGHT] += 0.01  # Add a small perturbation

    # Continue both simulations from the warmed-up state
    ca1 = CA(width, height, warmed_up_states.copy(), neighbor_list=BOTTOM_NEIGHBORS)
    ca2 = CA(width, height, perturbed_initial_state.copy(), neighbor_list=BOTTOM_NEIGHBORS)

    # Run the simulations for additional epochs (trajectories)
    ca1.run_simulation(num_epochs=1000, show_live=False, save_nth=1)  # Continue the unperturbed system
    ca2.run_simulation(num_epochs=1000, show_live=False, save_nth=1)  # Run the perturbed system

    # Retrieve saved states
    unmodified_states = ca1.saved_grids
    perturbed_states = ca2.saved_grids

    assert unmodified_states.shape == perturbed_states.shape, \
    f"State dimensions do not match: {unmodified_states.shape} vs {perturbed_states.shape}"

    assert unmodified_states[0].shape == ca1.grid.shape, \
    f"State dimensions do not match: {unmodified_states[0].shape} vs {ca1.grid.shape}"

    water_height_unmod = unmodified_states[:, :, :, 0]  # Shape: (10, 101, 21)
    water_height_perturbed = perturbed_states[:, :, :, 0]  # Shape: (10, 101, 21)

    ground_height_unmod = unmodified_states[:, :, :, 1]  # Shape: (10, 101, 21)
    ground_height_perturbed = perturbed_states[:, :, :, 1]  # Shape: (10, 101, 21)

    water_diff = np.abs(water_height_unmod - water_height_perturbed)

    ground_diff = np.abs(ground_height_unmod - ground_height_perturbed)

    mean_water_diff = np.mean(water_diff, axis=(1, 2))  # Shape: (10,)

    mean_ground_diff = np.mean(ground_diff, axis=(1, 2))  # Shape: (10,)


    plt.plot(mean_water_diff, label="Water Height Difference")
    plt.plot(mean_ground_diff, label="Ground Height Difference")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Difference")
    plt.title("Differences in Water and Ground Height Over Time")
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.legend()
    plt.show()




