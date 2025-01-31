"""
Course: Complex systems
Names: Marvin Frommer, Wessel Beumer, Paul Jungnickel, Tika van Bennekum

File description:
    Analysis of the divergence between perturbed and unperturbed data.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

perturbed_data = np.load("data/perturbed_data.npy")
unperturbed_data = np.load("data/unperturbed_data.npy")


def show_grid(grid, title=""):
    """
    Show a grid.

    Parameters:
        grid (np.ndarray): The grid to show.
        title (str): The title of the plot.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap="ocean")
    plt.colorbar()
    plt.title(title)
    plt.show()


def plot_initial_states():
    """
    Plot the initial states of the grids.

    This function loads unperturbed and perturbed data, then plots the initial states
    and their differences.
    """
    grids_unperturbed = np.load("data/unperturbed_data.npy")
    grids_perturbed = np.load("data/perturbed_data.npy")

    # Extract the initial ground height for unperturbed data
    unperturbed_ground_height = grids_unperturbed[
        0, :, :, 0
    ]  # Assuming GROUND_HEIGHT index is 0

    # Extract the initial ground height for the first simulation of the perturbed data
    perturbed_ground_height = grids_perturbed[
        0, 0, :, :, 0
    ]  # 0 for the first simulation

    # Create a figure for the plots
    plt.figure(figsize=(18, 6))  # Increased width for three plots

    # Plot unperturbed ground height
    ax1 = plt.subplot(1, 3, 1)
    plt.imshow(unperturbed_ground_height, cmap="copper")
    plt.colorbar()
    plt.title("Initial State - Unperturbed")
    ax1.set_xticks([])  # Remove x-axis ticks
    ax1.set_yticks([])  # Remove y-axis ticks

    # Plot perturbed ground height
    ax2 = plt.subplot(1, 3, 2)
    plt.imshow(perturbed_ground_height, cmap="copper")
    plt.colorbar()
    plt.title("Initial State - Perturbed")
    ax2.set_xticks([])  # Remove x-axis ticks
    ax2.set_yticks([])  # Remove y-axis ticks

    # Plot the difference between perturbed and unperturbed ground heights
    ax3 = plt.subplot(1, 3, 3)
    difference = perturbed_ground_height - unperturbed_ground_height
    plt.imshow(difference, cmap="terrain")
    plt.colorbar()
    plt.title("Initial State - Difference")
    ax3.set_xticks([])  # Remove x-axis ticks
    ax3.set_yticks([])  # Remove y-axis ticks

    plt.tight_layout()
    plt.savefig("data/2initial_states.png")
    plt.show()


def plot_divergence(perturbed_file, unperturbed_file):
    """
    Load perturbed and unperturbed data, compute the mean absolute divergence,
    and plot the results for ground and water height divergences.

    Parameters:
        perturbed_file (str): Path to the perturbed data file.
        unperturbed_file (str): Path to the unperturbed data file.
    """
    # Load perturbed and unperturbed data
    perturbed_data = np.load(perturbed_file)
    unperturbed_data = np.load(unperturbed_file)

    print(unperturbed_data.shape)
    print(perturbed_data.shape)

    # Compute absolute differences
    diffs = np.abs(perturbed_data - unperturbed_data)  # Shape: (10, 1000, 101, 21, 2)

    # Average over spatial dimensions (width=101, height=21)
    spatial_avg_diffs = np.mean(diffs, axis=(2, 3))  # Shape: (10, 1000, 2)

    # Average over all perturbed simulations
    mean_divergence = np.mean(spatial_avg_diffs, axis=0)  # Shape: (1000, 2)

    # Extract ground and water divergence
    ground_divergence = mean_divergence[:, 0]  # Shape: (1000,)
    water_divergence = mean_divergence[:, 1]  # Shape: (1000,)

    # Time axis
    timesteps = np.arange(len(ground_divergence))

    plt.figure(figsize=(8, 4))

    # Plot ground divergence
    plt.plot(timesteps, ground_divergence, label="Ground Height", color="blue")
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Mean Absolute Difference", fontsize=12)
    plt.title("Ground Height Difference Over Time", fontsize=14)
    plt.legend()
    plt.yscale("log")
    plt.grid()
    plt.savefig("data/ground_divergence.png")
    plt.show()

    plt.figure(figsize=(8, 4))
    # Plot water divergence
    plt.plot(timesteps, water_divergence, label="Water Height", color="green")
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Mean Absolute Divergence", fontsize=12)
    plt.title("Water Height Difference Over Time", fontsize=14)
    plt.yscale("log")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig("data/water_divergence.png")
    plt.show()


def plot_hamming(perturbed_file, unperturbed_file, threshold=0.002):
    """
    Load perturbed and unperturbed data, binarize them, compute the Hamming distance,
    and plot the results for average changes over time.

    Parameters:
        perturbed_file (str): Path to the perturbed data file.
        unperturbed_file (str): Path to the unperturbed data file.
        threshold (float): Threshold for binarizing the data.
    """
    # Load perturbed and unperturbed data
    perturbed_data = np.load(
        perturbed_file
    )  # Shape: (num_simulations, num_timesteps, height, width, NUM_CELL_FLOATS)
    unperturbed_data = np.load(
        unperturbed_file
    )  # Shape: (num_timesteps, height, width, NUM_CELL_FLOATS)

    print(unperturbed_data.shape)  # e.g., (1000, 101, 21, 2)
    print(perturbed_data.shape)  # e.g., (10, 1000, 101, 21, 2)

    # Binarize data (set values < threshold to 0, others to 1)
    perturbed_bin = (perturbed_data >= threshold).astype(int)
    unperturbed_bin = (unperturbed_data >= threshold).astype(int)

    # Expand unperturbed data to match the perturbed shape
    unperturbed_expanded = np.expand_dims(
        unperturbed_bin, axis=0
    )  # Shape: (1, 1000, 101, 21, 2)
    unperturbed_repeated = np.repeat(
        unperturbed_expanded, perturbed_data.shape[0], axis=0
    )  # Match simulations

    # Compute Hamming distance
    hamming = np.sum(
        perturbed_bin != unperturbed_repeated, axis=-1
    )  # Shape: (num_simulations, 1000, 101, 21)

    # Average over spatial dimensions (width=21, height=101)
    spatial_avg_hamming = np.mean(
        hamming, axis=(2, 3)
    )  # Shape: (num_simulations, 1000)

    # Average over all perturbed simulations to get the grand average
    mean_hamming = np.mean(spatial_avg_hamming, axis=0)  # Shape: (1000,)

    # Plotting the grand average Hamming distance
    timesteps = np.arange(len(mean_hamming))

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(timesteps, mean_hamming, label="Average Hamming Distance", color="purple")
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Average Hamming Distance", fontsize=12)
    plt.title("Average Hamming Distance Over Time (Binarized Water Data)", fontsize=14)
    plt.legend()
    plt.grid()
    plt.yscale("log")
    plt.tight_layout()
    plt.show()

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(timesteps, mean_hamming, label="Average Hamming Distance", color="purple")
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Average Hamming Distance", fontsize=12)
    plt.title("Average Hamming Distance Over Time (Water)", fontsize=14)
    plt.legend()
    plt.grid()
    plt.yscale("log")
    plt.tight_layout()
    plt.show()


def create_threshold_colormap(threshold):
    """
    Create a colormap with only two colors:
    - Light beige-brown for land
    - Blue for water

    Parameters:
        threshold (float): The water height threshold for coloring.

    Returns:
        cmap, norm: The colormap and normalization to use in imshow.
    """
    colors = ["#D2B48C", "#0000FF"]  # Light brown (beige) and blue
    boundaries = [0, threshold, 1000000]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)

    return cmap, norm


def show_masked_grid(grid, threshold, title=""):
    """
    Show a grid while masking NaN and Inf values to avoid errors in plotting.

    Parameters:
        grid (np.ndarray): The grid to show.
        title (str): The title of the plot.
    """
    masked_grid = np.ma.masked_invalid(grid, threshold)  # Mask NaN and Inf values
    cmap, norm = create_threshold_colormap(threshold)
    plt.figure(figsize=(6, 6))
    plt.imshow(masked_grid, cmap=cmap, norm=norm)
    plt.colorbar()
    plt.title(title)
    plt.show()


# Try plotting with masked values
show_masked_grid(perturbed_data[0, -1, :, :, 1], 0.002, title="Perturbed Water Height")
show_masked_grid(unperturbed_data[-1, :, :, 1], 0.002, title="Unperturbed Water Height")


plot_divergence("data/perturbed_data.npy", "data/unperturbed_data.npy")
plot_hamming("data/perturbed_data.npy", "data/unperturbed_data.npy")
plot_initial_states()
