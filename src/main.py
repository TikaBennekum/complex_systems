from CA import *
from initial_state_generation import generate_initial_slope
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Example usage
    width, height, ground_height = 101, 1001, 10
    
    initial_state = generate_initial_slope(height, width, ground_height, noise_amplitude = 10)
    
    output_file = 'videos/water_simulation.mp4'
    ca = CA(width, height, initial_state, neighbor_list=BOTTOM_NEIGHBORS)
    ca.run_simulation(1000)
    plt.imshow(ca.grid[:,:, 0])
    plt.colorbar()
    plt.savefig('data/test.png')




