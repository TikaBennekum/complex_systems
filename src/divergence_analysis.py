import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_initial_states():
    """
    Plot the initial states of the grids.

    This function loads unperturbed and perturbed data, then plots the initial states
    and their differences.
    """
    grids_unperturbed = np.load("data/unperturbed_data.npy")
    grids_perturbed = np.load("data/perturbed_data.npy")

    # Extract the initial ground height for unperturbed data
    unperturbed_ground_height = grids_unperturbed[0, :, :, 0]  # Assuming GROUND_HEIGHT index is 0

    # Extract the initial ground height for the first simulation of the perturbed data
    perturbed_ground_height = grids_perturbed[0, 0, :, :, 0]  # 0 for the first simulation

    # Create a figure for the plots
    plt.figure(figsize=(18, 6))  # Increased width for three plots

    # Plot unperturbed ground height
    ax1 = plt.subplot(1, 3, 1)
    plt.imshow(unperturbed_ground_height, cmap='copper')
    plt.colorbar()
    plt.title("Initial State - Unperturbed")
    ax1.set_xticks([])  # Remove x-axis ticks
    ax1.set_yticks([])  # Remove y-axis ticks

    # Plot perturbed ground height
    ax2 = plt.subplot(1, 3, 2)
    plt.imshow(perturbed_ground_height, cmap='copper')
    plt.colorbar()
    plt.title("Initial State - Perturbed")
    ax2.set_xticks([])  # Remove x-axis ticks
    ax2.set_yticks([])  # Remove y-axis ticks

    # Plot the difference between perturbed and unperturbed ground heights
    ax3 = plt.subplot(1, 3, 3)
    difference = perturbed_ground_height - unperturbed_ground_height
    plt.imshow(difference, cmap='terrain')
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
    water_divergence = mean_divergence[:, 1]   # Shape: (1000,)

    # Time axis
    timesteps = np.arange(len(ground_divergence))

    plt.figure(figsize=(8, 4))

    # Plot ground divergence
    plt.plot(timesteps, ground_divergence, label="Ground Height", color="blue")
    plt.xlabel("Time Step")
    plt.ylabel("Mean Absolute Difference")
    plt.title("Ground Height Difference Over Time")
    plt.legend()
    #plt.yscale("log")
    plt.grid()
    #plt.savefig("data/ground_divergence.png")
    plt.show()

    plt.figure(figsize=(8, 4))
    # Plot water divergence
    plt.plot(timesteps, water_divergence, label="Water Height", color="green")
    plt.xlabel("Time Step")
    plt.ylabel("Mean Absolute Divergence")
    plt.title("Water Height Difference Over Time")
    #plt.yscale("log")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    #plt.savefig("data/water_divergence.png")
    plt.show()

def plot_hamming(perturbed_file, unperturbed_file):
    """
    Load perturbed and unperturbed data, compute the Hamming distance,
    and plot the results for average changes over time.

    Parameters:
        perturbed_file (str): Path to the perturbed data file.
        unperturbed_file (str): Path to the unperturbed data file.
    """
    # Load perturbed and unperturbed data
    perturbed_data = np.load(perturbed_file)  # Shape: (num_simulations, num_timesteps, height, width, NUM_CELL_FLOATS)
    unperturbed_data = np.load(unperturbed_file)  # Shape: (num_timesteps, height, width, NUM_CELL_FLOATS)

    print(unperturbed_data.shape)  # e.g., (1000, 101, 21, 2)
    print(perturbed_data.shape)     # e.g., (10, 1000, 101, 21, 2)

    # Ensure unperturbed data is expanded to match the shape of perturbed data
    unperturbed_expanded = np.expand_dims(unperturbed_data, axis=0)  # Shape: (1, 1000, 101, 21, 2)
    unperturbed_repeated = np.repeat(unperturbed_expanded, perturbed_data.shape[0], axis=0)  # Shape: (num_simulations, 1000, 101, 21, 2)

    # Compute Hamming distance
    hamming = np.sum(perturbed_data != unperturbed_repeated, axis=-1)  # Shape: (num_simulations, 1000, 101, 21)

    # Average over spatial dimensions (width=21, height=101)
    spatial_avg_hamming = np.mean(hamming, axis=(2, 3))  # Shape: (num_simulations, 1000)

    # Average over all perturbed simulations to get the grand average
    mean_hamming = np.mean(spatial_avg_hamming, axis=0)  # Shape: (1000,)

    # Plotting the grand average Hamming distance
    timesteps = np.arange(len(mean_hamming))

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(timesteps, mean_hamming, label="Average Hamming Distance", color="purple")
    plt.xlabel("Time Step")
    plt.ylabel("Average Hamming Distance")
    plt.title("Average Hamming Distance Over Time (Water)")
    plt.legend()
    plt.grid()
    plt.yscale("log")
    plt.tight_layout()
    #plt.savefig("data/hamming.png")
    plt.show()

# Example of calling the function
plot_divergence("data/perturbed_data.npy", "data/unperturbed_data.npy")
#plot_hamming("data/perturbed_data.npy", "data/unperturbed_data.npy")
#plot_initial_states()