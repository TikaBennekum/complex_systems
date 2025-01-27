from CA import *
from initial_state_generation import generate_initial_slope
import matplotlib.pyplot as plt
import numpy as np 

GROUND_HEIGHT = 0
WATER_HEIGHT = 1

def simulation(seed, erosion_rate, flow_rate):
    """ Runs simulation of system for certain initial conditions. """
    np.random.seed(seed)
    width, height, ground_height = 21, 101, 101*.1
    initial_state = generate_initial_slope(height, width, ground_height, noise_amplitude = 0.1, noise_type = 'white')
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
    First plot: Compares the total erosion over time for different intitial flow rates.
    Second plot: Shows total final erosion for linearly different initial flow rates.
    """
    plt.figure(figsize=(10, 6))

    total_erosion = []
    flow_rate = np.arange(2, 9, 1)
    k = 1 / flow_rate
    print(flow_rate)
    print(k)

    for i in range(len(flow_rate)):
        erosion_per_iteration = simulation(42, k[i], flow_rate[i])
        print(i)
        total_erosion.append(erosion_per_iteration[-1])

    # Create combined labels for x-axis
    labels = [f"k: {k_val:.2f}, flow_rate: {flow_rate_val}" for k_val, flow_rate_val in zip(k, flow_rate)]

    # Plot data
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(flow_rate)), total_erosion, marker='o')  # Use indices for x-values
    plt.xticks(ticks=range(len(flow_rate)), labels=labels, rotation=45, ha='right')  # Add combined labels
    plt.xlabel('k and Flow rate')
    plt.ylabel('Total Erosion')
    plt.title('Linear flow increase, linear erosion rate decrease-> erosion stays same ??')
    plt.grid(True, linestyle='--')
    plt.tight_layout()  # Adjust layout to fit rotated labels
    plt.savefig('../data/exp_incr_flow_decr_eros.png')
    plt.show()




    