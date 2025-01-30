"""
Course: Complex systems
Names: Marvin Frommer, Wessel Beumer, Paul Jungnickel, Tika van Bennekum

File description:
    Shows a system which temporarily forms a single stream, and then
    becomes chaotic again.
"""

import numpy as np
from numpy.typing import NDArray
from cpp_modules import fastCA
from constants import *
from initial_state_generation import add_central_flow, generate_initial_slope
from viewer import BarChartVisualizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from os.path import isfile, isdir


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


def cached_fastCA(
    width,
    height,
    steps,
    steps_per_gen,
    ground_height,
    flow_amount,
    noise_amplitude,
    seed,
):
    if not isdir(".cache"):
        os.mkdir(".cache")
        with open(".cache/.gitignore", "w") as file:
            file.write("*")
    if not isfile(".cache/index.npy"):
        index: NDArray = np.array([])
    else:
        index: NDArray = np.load(".cache/index.npy")
    state = np.array(
        [
            width,
            height,
            steps,
            steps_per_gen,
            ground_height,
            flow_amount,
            noise_amplitude,
            seed,
        ]
    )

    # print(f"{index = }")
    # print(f"{index[:, :-1] = }")
    # # print(f"{state = }")
    # print(f"{index[:, :-1] == state = }")
    # query = index[(index[:, :-1] == state).all()] if index.shape[0] > 0 else None
    query = None
    # print(f"{query = }")

    assert query is None or len(query) < 2, "Cache corruption detected."
    # print(f"{query = }")
    query = query[0] if query is not None else None
    # print(f"{query = }")

    if query is None or len(query) == 0:
        np.random.seed(seed)
        initial_state = generate_initial_slope(
            height,
            width,
            ground_height,
            noise_amplitude=noise_amplitude,
            noise_type="white",
        )
        add_central_flow(initial_state, flow_amount)

        number = index.shape[0]
        grids = run_fastCA(initial_state, steps, steps_per_gen)
        save_state = np.append(state, [number]).reshape((1, 9))
        print(f"{save_state = }")

        if index.shape[0] == 0:
            index = save_state
        # if index.shape[0] == 1:
        else:
            index = np.append(index, save_state.reshape(1, 9))
        print(f"{index = }")
        np.save(f".cache/grids_{number}", grids)
        np.save(".cache/index", index)
        return grids

    number = round(query[-1])
    
    grids = np.load(f".cache/grids_{number}.npy")
    print("Loaded grids from cache.")
    return grids


def run_fastCA(
    initial_state, steps, steps_per_gen, params=default_params, update_progress=False
):
    height, width, cell_dim = initial_state.shape
    num_gens = steps // steps_per_gen
    saved_grids = np.zeros([num_gens, height, width, cell_dim])
    saved_grids[0] = initial_state
    gen_grids = np.zeros([steps_per_gen, height, width, cell_dim])

    for gen in tqdm(range(num_gens)):
        gen_grids[0] = saved_grids[gen]
        fastCA.simulate(gen_grids, params)
        if gen < num_gens - 1:
            saved_grids[gen + 1] = gen_grids[-1]

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
    width: int,
    height: int,
    steps: int,
    steps_per_gen,
    ground_height,
    flow_amount,
    noise_amplitude,
    seed,
):
    grids = cached_fastCA(
        width,
        height,
        steps,
        steps_per_gen,
        ground_height,
        flow_amount,
        noise_amplitude,
        seed,
    )

    # for threshold in [1e-6, 1e-4, 1e-2]:
    threshold = 1e-6
    stream_list = []
    for grid in grids:
        stream_list.append(get_stream_number(grid, threshold=threshold))

    plt.plot(np.arange(grids.shape[0]), stream_list, "-", label=f"{threshold = }")

    plt.legend()
    plt.savefig(f"videos/stream_count_{seed}.png")
    plt.close()

    derivative = abs(np.gradient(stream_list))
    plt.plot(np.arange(len(derivative)), derivative)
    plt.savefig(f"videos/stream_derivative_{seed}.png")
    plt.close()

    div_hist, bins = np.histogram(
        derivative, len(np.arange(derivative.min(), derivative.max() + 1, step=1.0))
    )
    plt.plot(bins[:-1], div_hist, "o")
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(f"videos/stream_hist_{seed}.png")
    plt.close()

    visualizer = BarChartVisualizer(grids)
    visualizer.run()


width = 21
height = 101
steps = 20_000
steps_per_gen = 100
ground_height = 51 * 0.1
flow_amount = 1.0
noise_amplitude = 0.2
seed = 42

cached_fastCA(
    width,
    height,
    steps,
    steps_per_gen,
    ground_height,
    flow_amount,
    noise_amplitude,
    seed,
)
# cached_fastCA()
