import numpy as np
import matplotlib.pyplot as plt
from cpp_modules import fastCA
from CA import *
from initial_state_generation import generate_initial_slope, add_central_flow
from constants import *

class Experiment:
    def __init__(self, width, height, ground_height, init_water_height, num_steps, perturbation_factor):
        # Set simulation parameters
        self.width = width
        self.height = height
        self.ground_height = ground_height
        self.init_water_height = init_water_height
        self.num_steps = num_steps
        self.perturbation_factor = perturbation_factor

        # Initialize grids
        self.grids_unperturbed = np.zeros([num_steps, height, width, NUM_CELL_FLOATS])
        self.grids_perturbed = np.zeros_like(self.grids_unperturbed)

        # Define erosion parameters
        self.params = {
            "EROSION_K": EROSION_K,
            "EROSION_C": EROSION_C,
            "EROSION_n": N,
            "EROSION_m": EROSION_EXPONENT,
        }

    def initialize_states(self):
        """Generate initial states for the simulation."""
        # Generate initial ground slope with optional noise
        initial_state = generate_initial_slope(
            self.height, self.width, 
            slope_top=self.ground_height, slope_bot=0, 
            noise_amplitude=0.2, noise_type='white'
        )
        # Add central flow for water
        initial_state = add_central_flow(initial_state, flow_amount=self.init_water_height)

        # Set initial states for unperturbed and perturbed grids
        self.grids_unperturbed[0] = initial_state.copy()
        self.grids_perturbed[0] = initial_state.copy()

    def apply_perturbation(self, max_perturbation=0.1):
        """
        Apply a fixed 3x3 perturbation window two rows below the water source.
        The window contains random values between -max_perturbation and max_perturbation.
        """
        # Define the fixed window size
        window_size = 3

        # Generate the perturbation window
        perturbation_window = np.random.uniform(
            -max_perturbation, max_perturbation, (window_size, window_size)
        )

        # Define the center of the window (two rows below the water source at row 0)
        center_row = 2  # Two rows below the water source
        center_col = self.width // 2  # Center column

        # Compute the start and end indices of the window
        start_row = max(center_row - window_size // 2, 0)
        start_col = max(center_col - window_size // 2, 0)
        end_row = start_row + window_size
        end_col = start_col + window_size

        # Ensure the window fits within the grid boundaries
        end_row = min(end_row, self.height)
        end_col = min(end_col, self.width)

        # Apply the perturbation
        self.grids_perturbed[:, start_row:end_row, start_col:end_col, GROUND_HEIGHT] += \
            perturbation_window[:end_row - start_row, :end_col - start_col]

    def plot_initial_states(self):
        """Plot the initial states of the grids."""
        # Plot unperturbed ground height
        plt.imshow(self.grids_unperturbed[0, :, :, GROUND_HEIGHT])
        plt.colorbar()
        plt.title("Initial State - Unperturbed")
        plt.show()

        # Plot perturbed ground height
        plt.imshow(self.grids_perturbed[0, :, :, GROUND_HEIGHT])
        plt.colorbar()
        plt.title("Initial State - Perturbed")
        plt.show()

    def plot_initial_difference(self):
        """Plot the initial difference in ground height between perturbed and unperturbed grids."""
        difference = self.grids_perturbed[0, :, :, GROUND_HEIGHT] - self.grids_unperturbed[0, :, :, GROUND_HEIGHT]
        plt.imshow(difference, cmap="seismic", interpolation="nearest")
        plt.colorbar(label="Height Difference")
        plt.title("Initial Terrain Height Difference (Time 0)")
        plt.show()

    def plot_time_0_difference(self):
        """
        Visualize the difference in ground height at time 0 
        between the unperturbed and perturbed simulations.
        """
        ground_diff = self.grids_perturbed[0, :, :, GROUND_HEIGHT] - self.grids_unperturbed[0, :, :, GROUND_HEIGHT]
        plt.figure(figsize=(8, 6))
        plt.imshow(ground_diff, cmap="seismic", interpolation="nearest")
        plt.colorbar(label="Height Difference")
        plt.title("Ground Height Difference at Time 0")
        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.show()

    def run_simulation(self):
        """Run the simulation for both unperturbed and perturbed grids."""
        fastCA.simulate(self.grids_unperturbed, self.params)  # Unperturbed simulation
        fastCA.simulate(self.grids_perturbed, self.params)  # Perturbed simulation

    def save_results(self, unperturbed_file, perturbed_file):
        """Save the simulation results to .npy files."""
        np.save(unperturbed_file, self.grids_unperturbed)
        np.save(perturbed_file, self.grids_perturbed)

    def run(self):
        """Execute the full workflow."""
        print("Initializing states...")
        self.initialize_states()
        print("Initial states generated.")

        print("Applying perturbation...")
        self.apply_perturbation()
        print("Perturbation applied.")

        print("Plotting initial states...")
        self.plot_initial_states()

        print("Visualizing initial height difference...")
        self.plot_initial_difference()

        print("Visualizing ground height difference at time 0...")
        self.plot_time_0_difference()

        print("Running simulations...")
        self.run_simulation()
        print("Simulations completed.")

        print("Saving results...")
        self.save_results('data/unperturbed_data.npy', 'data/perturbed_data.npy')
        print("Results saved.")


if __name__ == "__main__":
    # Set parameters for the simulation
    exp = Experiment(
        width=21,
        height=101,
        ground_height=101,
        init_water_height=25,
        num_steps=1000,
        perturbation_factor=0.1
    )

    # Run the full simulation workflow
    exp.run()
