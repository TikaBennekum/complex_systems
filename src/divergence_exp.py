import numpy as np
import matplotlib.pyplot as plt
from CA import *
from initial_state_generation import generate_initial_slope

def initialize_ca(width, height, initial_state, water_height=50):
    """Initialize a cellular automaton with a given initial state and water source."""
    ca = CA(width, height, initial_state.copy(), neighbor_list=BOTTOM_NEIGHBORS)
    ca.grid[0, width // 2, WATER_HEIGHT] = water_height  # Set water source at the center
    return ca

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

def run_simulation(ca1, ca2, num_epochs=1000):
    """Run the simulations for both cellular automata."""
    ca1.run_simulation(num_epochs=num_epochs, show_live=False, save_nth=1)  # Unperturbed system
    ca2.run_simulation(num_epochs=num_epochs, show_live=False, save_nth=1)  # Perturbed system

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

    # Initialize both cellular automata
    ca1 = initialize_ca(width, height, initial_state)  # Unperturbed system
    perturbed_initial_state = initial_state.copy()
    perturbed_initial_state[1, width // 2, GROUND_HEIGHT] += 0.1  # Add a small perturbation
    ca2 = initialize_ca(width, height, perturbed_initial_state)  # Perturbed system

    # Plot the initial difference
    plot_initial_difference(initial_state, perturbed_initial_state)

    # Run the simulations
    run_simulation(ca1, ca2)

    # Retrieve saved states
    unmodified_states = ca1.saved_grids
    perturbed_states = ca2.saved_grids

    # Analyze and save results
    analyze_results(unmodified_states, perturbed_states)

if __name__ == "__main__":
    main()
