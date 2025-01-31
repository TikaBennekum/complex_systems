"""
    Course: Complex systems
    Names: Marvin Frommer, Wessel Beumer, Paul Jungnickel, Tika van Bennekum

    File description:
        File to profile the code.
"""

from CA import CA, BOTTOM_NEIGHBORS
from initial_state_generation import generate_initial_slope
import yappi
import numpy as np
import re

if __name__ == '__main__':
    np.random.seed(42)
    width, height, ground_height = 21, 101, 101*.1
    
    initial_state = generate_initial_slope(height, width, ground_height, noise_amplitude = 0.1, noise_type = 'white')
    ca = CA(width, height, initial_state, neighbor_list=BOTTOM_NEIGHBORS)

    yappi.start(builtins=True)
    ca.run_simulation(200, show_live=False)
    yappi.stop()
    yappi.get_func_stats().save("data/profile.dat", type="pstat")
