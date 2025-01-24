from CA import CA, BOTTOM_NEIGHBORS
from initial_state_generation import generate_initial_slope
from cProfile import Profile
import numpy as np

if __name__ == '__main__':
    profiler = Profile()
    
    np.random.seed(42)
    width, height, ground_height = 21, 101, 101*.1
    
    initial_state = generate_initial_slope(height, width, ground_height, noise_amplitude = 0.1, noise_type = 'white')
    
    ca = CA(width, height, initial_state, neighbor_list=BOTTOM_NEIGHBORS)
    profiler.enable()
    ca.run_simulation(200, show_live=False)
    profiler.disable()
    profiler.dump_stats("data/profile.dat")