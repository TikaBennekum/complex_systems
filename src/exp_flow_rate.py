from CA import *
from initial_state_generation import generate_initial_slope
import matplotlib.pyplot as plt
import numpy as np 

GROUND_HEIGHT = 0
WATER_HEIGHT = 1

def simulation(seed, erosion_rate, flow_rate, slope=101*0.1):
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
    First plot: Compares the total erosion over time for different intitial flow rates.
    Second plot: Shows total final erosion for linearly different initial flow rates.
    """
    plt.figure(figsize=(10, 6))

    total_erosion = []
    flow_rate = np.arange(1, 9, 1)
    k = 0.1

    for i in flow_rate:
        erosion_per_iteration = simulation(42, k, i)
        print(i)
        total_erosion.append(erosion_per_iteration[-1])
        plt.plot(range(1, len(erosion_per_iteration) + 1), erosion_per_iteration, label=f'flow rate={i}')

    plt.xlabel('Nr of iterations')
    plt.ylabel('Total Erosion')
    plt.title('Erosion over time vor various flow rates')
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.savefig('../data/exp_flow_rate.png')

    plt.figure(figsize=(10, 6))
    plt.scatter(flow_rate, total_erosion, marker='o')
    plt.xlabel('Flow rate')
    plt.ylabel('Total Erosion')
    plt.title('Linear flow rate increase -> linear total erosion increase')
    plt.grid(True, linestyle='--')
    plt.savefig('../data/exp_flow_rate2.png')
    plt.show()




    