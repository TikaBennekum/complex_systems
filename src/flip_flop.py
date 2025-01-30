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
from tqdm import tqdm
import matplotlib.pyplot as plt


EROSION_K = 0.1  # erosion rate
EROSION_C = 0.3
EROSION_EXPONENT = 2.5
N = 0.5
default_params = {
    "EROSION_K": EROSION_K,
    "EROSION_C": EROSION_C,
    "EROSION_n": N,
    "EROSION_m": EROSION_EXPONENT,
    "print_params": 0,
}


def view_configuration(
    width, height, ground_height, num_steps, noise_amplitude, seed=42
):
    """Function to view configuration."""
    np.random.seed(seed)

    # Generate the initial grid state
    initial_state = generate_initial_slope(
        height,
        width,
        ground_height,
        noise_amplitude=noise_amplitude,
        noise_type="white",
    )
    add_central_flow(initial_state, 1)

    grids = run_fastCA(initial_state, num_steps, 1_000)

    # Visualize the data
    visualizer = BarChartVisualizer(grids)
    visualizer.run()


def run_fastCA(
    initial_state, steps, steps_per_gen, params=default_params, update_progress=False
):
    """Function to run the Cellular Automata fast."""

    height, width, cell_dim = initial_state.shape
    num_gens = steps // steps_per_gen
    saved_grids = np.zeros([num_gens, height, width, cell_dim])
    saved_grids[0] = initial_state
    gen_grids = np.zeros([steps_per_gen, height, width, cell_dim])
    for gen in tqdm(range(num_gens)):
        if update_progress:
            print(gen % 10, end="", flush=True)
        gen_grids[0] = saved_grids[gen]
        fastCA.simulate(gen_grids, params)
        if gen < num_gens - 1:
            saved_grids[gen + 1] = gen_grids[-1]

    if update_progress:
        print("")
    return saved_grids


def get_stream_number(grid, threshold):
    """Returns the mean stream number."""
    streams_in_row = []
    for row in grid:
        stream_count = 0
        in_stream = False
        for cell in row:
            if cell[WATER_HEIGHT] > threshold and not in_stream:
                stream_count += 1

            if cell[WATER_HEIGHT] > threshold:
                in_stream = True
            else:
                in_stream = False
        streams_in_row.append(stream_count)
    return np.mean(streams_in_row)


def stream_number_progression(
    width, height, ground_height, num_steps, noise_amplitude, seed
):
    """Shows the stream number progression,=."""
    np.random.seed(seed)

    # Generate the initial grid state
    initial_state = generate_initial_slope(
        height,
        width,
        ground_height,
        noise_amplitude=noise_amplitude,
        noise_type="white",
    )
    add_central_flow(initial_state, 1)

    grids = run_fastCA(initial_state, num_steps, 1_000)

    # for threshold in [1e-6, 1e-4, 1e-2]:
    threshold = 1e-6
    stream_list = []
    for grid in grids:
        stream_list.append(get_stream_number(grid, threshold=threshold))

    plt.plot(np.arange(grids.shape[0]), stream_list, "-", label=f"{threshold = }")

    plt.legend()
    plt.savefig("videos/stream_count.png")
    plt.close()

    derivative = abs(np.gradient(stream_list))
    print(derivative)
    plt.plot(np.arange(len(derivative)), derivative)
    plt.savefig("videos/stream_derivative.png")
    plt.close()

    div_hist, bins = np.histogram(derivative, 20)
    plt.plot(bins[:-1], div_hist, "o")
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("videos/stream_hist.png")
    plt.close()

    visualizer = BarChartVisualizer(grids)
    visualizer.run()


width, height, ground_height, num_steps = 21, 101, 51 * 0.1, 200_000
noise_amplitude = 0.2

stream_number_progression(
    width, height, ground_height, num_steps, noise_amplitude, seed=42
)
