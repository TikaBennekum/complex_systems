import numpy as np
import matplotlib.pyplot as plt
from cpp_modules.fastCA import simulate  # Import the simulate function from the C++ module
from initial_state_generation import generate_initial_slope

GROUND_HEIGHT = 0
WATER_HEIGHT = 1

def generate_initial_state(width, height, ground_height):
    """Generate the initial terrain with a slope and some noise."""
    return generate_initial_slope(height, width, ground_height, noise_amplitude=0.1, noise_type='white')

def plot_initial_difference(initial_state, perturbed_initial_state):
    """Plot the initial difference in ground height."""
    initial_diff = np.abs(initial_state - perturbed_initial_state)
    plt.imshow(initial_diff[:, :, GROUND_HEIGHT], cmap='hot', interpolation='nearest')
    plt.title("Initial Ground Height Difference")
    plt.colorbar()
    plt.show()

def run_simulation(grids, params, num_steps=1000):
    """Run the simulation using the C++ function."""
    # Call the C++ simulation function
    simulate(grids, params)  # Pass the grids and parameters to the C++ function

def analyze_results(unmodified_states, perturbed_states):
    """Analyze and save the mean water and ground height differences."""
    assert unmodified_states.shape == perturbed_states.shape, \
        f"State dimensions do not match: {unmodified_states.shape} vs {perturbed_states.shape}"
    
    assert unmodified_states[0].shape == unmodified_states.shape[1:], \
        f"State dimensions do not match: {unmodified_states[0].shape} vs {unmodified_states.shape[1:]}"
    
    water_height_unmod = unmodified_states[:, :, :, WATER_HEIGHT]  # Shape: (10, 101, 21)
    water_height_perturbed = perturbed_states[:, :, :, WATER_HEIGHT]  # Shape: (10, 101, 21)

    ground_height_unmod = unmodified_states[:, :, :, GROUND_HEIGHT]  # Shape: (10, 101, 21)
    ground_height_perturbed = perturbed_states[:, :, :, GROUND_HEIGHT]  # Shape: (10, 101, 21)

    water_diff = np.abs(water_height_unmod - water_height_perturbed)
    ground_diff = np.abs(ground_height_unmod - ground_height_perturbed)

    mean_water_diff = np.mean(water_diff, axis=(1, 2))  # Shape: (10,)
    mean_ground_diff = np.mean(ground_diff, axis=(1, 2))  # Shape: (10,)

    np.save('data/mean_water_height_diff.npy', mean_water_diff)
    np.save('data/mean_ground_height_diff.npy', mean_ground_diff)

def main():
    """Main function to run the simulation."""
    # Set random seed for reproducibility
    np.random.seed(42)

    # Define the grid dimensions and initial ground height
    width, height, ground_height = 21, 101, 101 * 0.1

    # Generate the initial terrain
    initial_state = generate_initial_state(width, height, ground_height)

    # Create a 4D grid for the simulation
    num_steps = 10000  # Number of time steps to simulate
    grids = np.zeros((num_steps, height, width, 2), dtype=np.float64)  # 2 channels: GROUND_HEIGHT, WATER_HEIGHT

    # Initialize both states
    grids[0, :, :, GROUND_HEIGHT] = initial_state[:, :, GROUND_HEIGHT]
    grids[0, :, :, WATER_HEIGHT] = 50  # Set water source at the center

    perturbed_initial_state = initial_state.copy()
    perturbed_initial_state[1, width // 2, GROUND_HEIGHT] += 0.1  # Add a small perturbation
    grids[0, :, :, GROUND_HEIGHT] = perturbed_initial_state[:, :, GROUND_HEIGHT]
    grids[0, :, :, WATER_HEIGHT] = 50  # Set water source at the center

    # Plot the initial difference
    plot_initial_difference(initial_state, perturbed_initial_state)

    # Define parameters as a dictionary
    params = {
        "EROSION_K": 0.1,
        "EROSION_C": 0.1,
        "EROSION_n": 2.0,
        "EROSION_m": 1.0
    }

    # Run the simulations
    run_simulation(grids, params)

    # Retrieve saved states
    unmodified_states = grids.copy()  # Assume grids are modified in-place by the C++ function
    perturbed_states = grids.copy()  # Modify as needed to reflect perturbed results

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # Plot for unmodified ground height
    axs[0, 0].imshow(unmodified_states[-1][:, :, GROUND_HEIGHT], cmap='hot', interpolation='nearest')
    axs[0, 0].set_title("Unmodified Ground Height")
    axs[0, 0].axis('off')

    # Plot for perturbed ground height
    axs[0, 1].imshow(perturbed_states[-1][:, :, GROUND_HEIGHT], cmap='hot', interpolation='nearest')
    axs[0, 1].set_title("Perturbed Ground Height")
    axs[0, 1].axis('off')

    # Plot for unmodified water height
    axs[1, 0].imshow(unmodified_states[-1][:, :, WATER_HEIGHT], cmap='Blues', interpolation='nearest')
    axs[1, 0].set_title("Unmodified Water Height")
    axs[1, 0].axis('off')

    # Plot for perturbed water height
    axs[1, 1].imshow(perturbed_states[-1][:, :, WATER_HEIGHT], cmap='Blues', interpolation='nearest')
    axs[1, 1].set_title("Perturbed Water Height")
    axs[1, 1].axis('off')

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    plt.show()

    # Analyze and save results
    analyze_results(unmodified_states, perturbed_states)

if __name__ == "__main__":
    main()
