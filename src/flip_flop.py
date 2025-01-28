"""
    Course: Complex systems
    Names: Marvin Frommer, Wessel Beumer, Paul Jungnickel, Tika van Bennekum

    File description:
        Shows a system which temporarily forms a single stream, and then 
        becomes chaotic again.
"""

import numpy as np
from cpp_modules import fastCA
from constants import *
from initial_state_generation import add_central_flow, generate_initial_slope
from viewer import BarChartVisualizer

EROSION_K = 0.1 # erosion rate
EROSION_C = 0.3
EROSION_EXPONENT = 2.5
N = 0.5

def view_configuration(width, height, ground_height, num_steps, noise_amplitude, seed = 42):
    np.random.seed(seed)

    # Generate the initial grid state
    initial_state = generate_initial_slope(height, width, ground_height, noise_amplitude=noise_amplitude, noise_type='white')
    add_central_flow(initial_state, 1)

    # Create grids and simulate data
    grids = np.zeros([num_steps, height, width, NUM_CELL_FLOATS])
    grids[0] = initial_state

    params = {
        "EROSION_K": EROSION_K,
        "EROSION_C": EROSION_C,
        "EROSION_n": N,
        "EROSION_m": EROSION_EXPONENT,
    }

    fastCA.simulate(grids, params)

    # Visualize the data
    visualizer = BarChartVisualizer(grids[::1000])
    visualizer.run()


width, height, ground_height, num_steps = 21, 101, 51 * 0.1, 1_000_000
noise_amplitude = 0.2
view_configuration(width, height, ground_height, num_steps, noise_amplitude)