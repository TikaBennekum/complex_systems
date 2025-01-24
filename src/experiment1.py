from CA import *
from initial_state_generation import generate_initial_slope
import matplotlib.pyplot as plt
import numpy as np 

GROUND_HEIGHT = 0
WATER_HEIGHT = 1

def erosion_per_iteration():
    return

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    width, height, ground_height = 21, 101, 101*.1
    
    initial_state = generate_initial_slope(height, width, ground_height, noise_amplitude = 0.1, noise_type = 'white')
    
    ca = CA(width, height, initial_state, neighbor_list=BOTTOM_NEIGHBORS)
    grids = ca.run_experiment1(100)
    grids[0] = initial_state
    # print("initial_grid", initial_state)
    # print("second grid:", grids[0])
    # print(grids[23][:, :, 0] == grids[46][:, :, 0])

    erosion_per_iteration = []
    previous_ground = grids[0][:, :, 0]

    for i in range(1, len(grids)):
        current_ground = grids[i][:, :, 0]
        
        difference = current_ground - previous_ground
        erosion_per_iteration.append(float(np.sum(np.abs(difference[difference < 0]))))  # Cells where ground height is lowered in the iteration

        previous_ground = current_ground

    print(erosion_per_iteration)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(erosion_per_iteration) + 1), erosion_per_iteration, marker='o', linestyle='-', color='b')

    # Add labels and title
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Total Erosion', fontsize=14)
    plt.title('Erosion Per Iteration', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Show the graph
    plt.show()

    # `erosion_per_iteration` now contains the erosion values for each iteration



    