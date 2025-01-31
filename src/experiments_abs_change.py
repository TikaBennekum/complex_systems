"""
Course: Complex systems
Names: Marvin Frommer, Wessel Beumer, Paul Jungnickel, Tika van Bennekum

File description:
    This file contains the code to run the system in C++, resulting in faster simulations.
"""

from CA import *
from initial_state_generation import generate_initial_slope, add_central_flow
import matplotlib.pyplot as plt
from visualization2d import *


from cpp_modules import fastCA

default_params = {
    "EROSION_K": EROSION_K,
    "EROSION_C": EROSION_C,
    "EROSION_n": N,
    "EROSION_m": EROSION_EXPONENT,
    "print_params": 0,
}


def mse(A, B):
    """The mean square error algorithm."""
    return np.mean((A - B) ** 2)


def mean_abs(A, B):
    """returns the mean of two lists of absolute differences."""
    return np.mean(np.abs(A - B))


def max_abs(A, B):
    """returns the max of two lists of absolute differences."""
    return np.max(np.abs(A - B))


def change_every_n_steps(
    initial_state,
    steps,
    steps_per_gen,
    metric,
    skip_initial_gens=0,
    params=default_params,
    update_progress=False,
):
    """
    Run the simulation with the fast cpp version for a number of timesteps
    the cpp simulation is invoked every state_per_gen steps, the grids after each
    cpp run are saved and returned

    Args:
        initial state: grid of ground and water heights at t=0, should contain water source at the top
        steps: number of timesteps to run simulation
        steps_per_gen: number of runs in each individual simulation run - reduce this if grids are too large
        params: additional simulation parameters passed to the cpp simulator

    Returns:
        saved_grids: simulation states saved every steps_per_gen timesteps with shape [num_steps //steps_per_gen x height x width x NUM_CELL_FLOATS]
    """
    height, width, cell_dim = initial_state.shape
    num_gens = steps // steps_per_gen

    diffs = np.zeros(num_gens)

    last_state = initial_state.copy()
    gen_grids = np.zeros([steps_per_gen, height, width, cell_dim])
    for gen in range(num_gens):
        if update_progress:
            print(gen % 10, end="", flush=True)
        gen_grids[0] = last_state
        fastCA.simulate(gen_grids, params)
        
        if gen >= skip_initial_gens:
            diffs[gen] = metric(
                last_state[:, :, WATER_HEIGHT], gen_grids[-1, :, :, WATER_HEIGHT]
            )
        last_state = gen_grids[-1].copy()

    if update_progress:
        print("")
    return diffs


def small_grid_change_experiment(
    output_file,
    slope=0.1,
    flow=1,
    width=21,
    height=100,
    num_steps=10000,
    steps_per_gen=100,
    params=default_params,
    metric=mse,
):
    """Experiment that computes a histogram of the sizes in water height changes over a long duration."""
    np.random.seed(42)
    ground_height = height * slope
    initial_state = generate_initial_slope(
        height, width, ground_height, noise_amplitude=0.2, noise_type="white"
    )
    add_central_flow(initial_state, flow)

    diffs = change_every_n_steps(initial_state, 1000000, 100, metric=metric)

    print(diffs)
    np.save(output_file, diffs)
    return diffs


def grid_change_histogram(diffs):
    """A histogram of the grid change."""
    hist, bins = np.histogram(diffs, bins=np.logspace(-2, 1, 50, base=10))

    hist = hist / len(diffs)
    plt.plot(1 / 2 * (bins[:-1] + bins[1:]), hist, linestyle="", marker="o")
    plt.xscale("log")
    plt.xlabel("size of change")
    plt.ylabel("frequency")

    plt.show()


if __name__ == "__main__":
    file = "data/height_change_hist_w1.npy"
    small_grid_change_experiment(file)
    grid_change_histogram(np.load(file))
