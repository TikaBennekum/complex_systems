"""
    Course: Complex systems
    Names: Marvin Frommer, Wessel Beumer, Paul Jungnickel, Tika van Bennekum

    File description:
        This file performs an experiment. 
        It increases the slope linearly while increasing the erosion rate linearly.
"""

from CA import *
from initial_state_generation import generate_initial_slope
import matplotlib.pyplot as plt
import numpy as np 

GROUND_HEIGHT = 0
WATER_HEIGHT = 1

def simulation(seed, erosion_rate, flow_rate, slope=101*.1):
    """ Runs simulation of system for certain initial conditions. """
    np.random.seed(seed)
    width, height = 21, 101
    initial_state = generate_initial_slope(height, width, slope, noise_amplitude = 0.1, noise_type = 'white')
    ca = CA(width, height, initial_state, neighbor_list=BOTTOM_NEIGHBORS)
    grids = ca.run_experiment(100, erosion_rate, flow_rate)

    erosion_per_iteration = []
    previous_ground = grids[0][:, :, 0]
    # Each loop is one iteration of the simulation
    for i in range(1, len(grids)):
        current_ground = grids[i][:, :, 0]
        
        difference = current_ground - previous_ground
        erosion_per_iteration.append(float(np.sum(np.abs(difference[difference < 0]))))  # Cells where ground height is lowered in the iteration

        previous_ground = current_ground
    return erosion_per_iteration

if __name__ == "__main__":
    """
    First plot: Shows total erosion for a simultaneously increasing flow rate and decreasing erosion rate.
    """
    plt.figure(figsize=(10, 6))

    total_erosion = []
    flow_rate = 1
    slope = np.arange(101*0.1, 101*0.15, 0.5)
    k = 1 / slope
    k = np.round(k, 3)
    print(slope)
    print(k)

    for i in range(len(slope)):
        erosion_per_iteration = simulation(42, k[i], flow_rate, slope[i])
        print(i)
        total_erosion.append(erosion_per_iteration[-1])

    # Create combined labels for x-axis
    labels = [f"k: {k_val:.2f}, slope: {np.round(slope_val, 3)}" for k_val, slope_val in zip(k, slope)]

    # Plot data
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(slope)), total_erosion, marker='o')  # Use indices for x-values
    plt.xticks(ticks=range(len(slope)), labels=labels, rotation=45, ha='right')  # Add combined labels
    plt.xlabel('k and slope')
    plt.ylabel('Total Erosion')
    plt.title('Linear slope increase, linear erosion rate increase -> ?')
    plt.grid(True, linestyle='--')
    plt.tight_layout()  # Adjust layout to fit rotated labels
    plt.savefig('../data/exp_incr_slope_incr_eros.png')
    plt.show()




    