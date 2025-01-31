"""
Course: Complex systems
Names: Marvin Frommer, Wessel Beumer, Paul Jungnickel, Tika van Bennekum

File description:
    Other files can be called from this main (Python) file.
"""

from CA import *
from initial_state_generation import generate_initial_slope, add_central_flow
import matplotlib.pyplot as plt
from visualization2d import *


from cpp_modules import fastCA


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    width, height, ground_height, num_steps = 21, 101, 101 * 0.1, 10000

    initial_state = generate_initial_slope(
        height, width, ground_height, noise_amplitude=0.2, noise_type="white"
    )

    add_central_flow(initial_state, 1)

    grids = np.zeros([num_steps, height, width, NUM_CELL_FLOATS])
    grids[0] = initial_state

    plt.savefig("../data/cpptest0.png")

    params = {
        "EROSION_K": EROSION_K,
        "EROSION_C": EROSION_C,
        "EROSION_n": N,
        "EROSION_m": EROSION_EXPONENT,
    }

    fastCA.simulate(grids, params)

    plt.imshow(grids[-1, :, :, GROUND_HEIGHT] - initial_state[:, :, GROUND_HEIGHT])
    plt.colorbar()
    plt.show()
    
    stream_video(grids)
