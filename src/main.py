from CA import *
from initial_state_generation import generate_initial_slope
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    width, height, ground_height = 21, 101, 101*.1
    
    initial_state = generate_initial_slope(height, width, ground_height, noise_amplitude = 0.1, noise_type = 'white')
    
    output_file = 'videos/water_simulation.mp4'
    ca = CA(width, height, initial_state, neighbor_list=BOTTOM_NEIGHBORS)
    ca.run_simulation(1000, show_live=True)
    
    plt.imshow(ca.grid[:,:,GROUND_HEIGHT] - initial_state[:,:,GROUND_HEIGHT])
    plt.colorbar()
    
    plt.savefig('data/test.png')
    
    