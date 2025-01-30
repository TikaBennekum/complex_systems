"""
    Course: Complex systems
    Names: Marvin Frommer, Wessel Beumer, Paul Jungnickel, Tika van Bennekum

    File description:
        This file contains experiments, where the effect of parameter values on the erosion is researched.
"""

from cpp_modules import fastCA
from CA import *
from initial_state_generation import generate_initial_slope
import matplotlib.pyplot as plt
import numpy as np 

GROUND_HEIGHT = 0
WATER_HEIGHT = 1

def simulation(seed, erosion_rate, flow_rate, slope=101*0.1):
    """ Runs simulation of system for certain initial conditions. """
    np.random.seed(seed)
    width, height = 21, 101
    initial_state = generate_initial_slope(height, width, slope, noise_amplitude = 0.1, noise_type = 'white')
    ca = CA(width, height, initial_state, neighbor_list=BOTTOM_NEIGHBORS)
    nr_of_iterations = 100
    grids = ca.run_experiment(nr_of_iterations, erosion_rate, flow_rate)

    erosion_per_iteration = []
    previous_ground = grids[0][:, :, 0]
    # Each loop is one iteration of the simulation
    for i in range(1, len(grids)):
        current_ground = grids[i][:, :, 0]
        
        difference = current_ground - previous_ground
        erosion_per_iteration.append(float(np.sum(np.abs(difference[difference < 0]))))  # Cells where ground height is lowered in the iteration

        previous_ground = current_ground
    return erosion_per_iteration

def exp_erosion_rate():
    """
    First plot: Compares the total erosion over time for different intitial erosion rates.
    Second plot: Shows total final erosion for linearly different initial erosion rates.
    """
    plt.figure(figsize=(10, 6))

    total_erosion = []
    k = np.arange(0.1, 0.5, 0.05)
    flow_rate = 1

    for i in k:
        erosion_per_iteration = simulation(42, np.round(i, 2), flow_rate)
        print(np.round(i, 2))
        total_erosion.append(erosion_per_iteration[-1])
        plt.plot(range(1, len(erosion_per_iteration) + 1), erosion_per_iteration, label=f'Erosion rate={np.round(i, 2)}')

    plt.xlabel('Nr of iterations')
    plt.ylabel('Total Erosion')
    plt.title('Erosion over time for various erosion rates')
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.savefig('../data/exp_erosion_rate.png')


    plt.figure(figsize=(10, 6))
    plt.scatter(k, total_erosion, marker='o')
    plt.xlabel('Erosion rate', fontsize=16)
    plt.ylabel('Total Erosion', fontsize=16)
    plt.title('Total erosion for linearly increasing erosion rates', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--')
    plt.savefig('../data/exp_erosion_rate2.png')

def exp_flow_rate():
    """
    First plot: Compares the total erosion over time for different intitial flow rates.
    Second plot: Shows total final erosion for linearly different initial flow rates.
    """
    plt.figure(figsize=(10, 6))

    total_erosion = []
    flow_rate = np.arange(1, 9, 1)
    k = 0.1

    for i in flow_rate:
        erosion_per_iteration = simulation(42, k, i)
        print(i)
        total_erosion.append(erosion_per_iteration[-1])
        plt.plot(range(1, len(erosion_per_iteration) + 1), erosion_per_iteration, label=f'flow rate={i}')

    plt.xlabel('Nr of iterations')
    plt.ylabel('Total Erosion')
    plt.title('Erosion over time vor various flow rates')
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.savefig('../data/exp_flow_rate.png')

    plt.figure(figsize=(10, 6))
    plt.scatter(flow_rate, total_erosion, marker='o')
    plt.xlabel('Flow rate', fontsize=16)
    plt.ylabel('Total Erosion', fontsize=16)
    plt.title('Total erosion for linearly increasing flow rates', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--')
    plt.savefig('../data/exp_flow_rate2.png')

def exp_slope():
    """
    First plot: Compares the total erosion over time for different intitial slopes.
    Second plot: Shows total final erosion for linearly different initial slopes.
    """
    plt.figure(figsize=(10, 6))

    total_erosion = []
    flow_rate = 1
    k = 0.1
    slope = np.arange(101*0.1, 101*0.15, 0.5)
    print(slope)

    for i in slope:
        i = np.round(i, 1)
        erosion_per_iteration = simulation(42, k, flow_rate, i)
        print(i)
        total_erosion.append(erosion_per_iteration[-1])
        plt.plot(range(1, len(erosion_per_iteration) + 1), erosion_per_iteration, label=f'flow rate={i}')

    plt.xlabel('Nr of iterations')
    plt.ylabel('Total Erosion')
    plt.title('Erosion over time vor various slopes')
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.savefig('../data/exp_slope.png')

    plt.figure(figsize=(10, 6))
    plt.scatter(slope, total_erosion, marker='o')
    plt.xlabel('Slope', fontsize=16)
    plt.ylabel('Total Erosion', fontsize=16)
    plt.title('Total erosion for linearly increasing slopes', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--')
    plt.savefig('../data/exp_slope2.png')

def exp_incr_flow_decr_eros():
    """
    First plot: Shows total erosion for a simultaneously increasing flow rate and decreasing erosion rate.
    """
    plt.figure(figsize=(10, 6))

    total_erosion = []
    flow_rate = np.arange(10, 18, 1)
    k = 1 / flow_rate
    k = k
    print(flow_rate)
    print(k)

    for i in range(len(flow_rate)):
        erosion_per_iteration = simulation(42, k[i], flow_rate[i])
        print(i)
        total_erosion.append(erosion_per_iteration[-1])

    # Create combined labels for x-axis
    labels = [f"k: {k_val:.2f}, flow_rate: {flow_rate_val}" for k_val, flow_rate_val in zip(k, flow_rate)]

    # Plot data
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(flow_rate)), total_erosion, marker='o')  # Use indices for x-values
    plt.xticks(ticks=range(len(flow_rate)), labels=labels, rotation=45, ha='right')  # Add combined labels
    plt.xlabel('Slope', fontsize=16)
    plt.ylabel('Total Erosion', fontsize=16)
    plt.title('Total erosion for linearly increasing slopes', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--')
    plt.tight_layout()  # Adjust layout to fit rotated labels
    plt.savefig('../data/exp_incr_flow_decr_eros.png')

def exp_incr_slope_incr_eros():
    """
    First plot: Shows total erosion for a simultaneously increasing flow rate and decreasing erosion rate.
    """
    plt.figure(figsize=(10, 6))

    total_erosion = []
    flow_rate = 1
    slope = np.arange(101*0.1, 101*0.15, 0.5)
    k = 1 / slope
    k = np.round(k, 3)
    print(slope)
    print(k)

    for i in range(len(slope)):
        erosion_per_iteration = simulation(42, k[i], flow_rate, slope[i])
        print(i)
        total_erosion.append(erosion_per_iteration[-1])

    # Create combined labels for x-axis
    labels = [f"k: {k_val:.2f}, slope: {np.round(slope_val, 3)}" for k_val, slope_val in zip(k, slope)]

    # Plot data
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(slope)), total_erosion, marker='o')  # Use indices for x-values
    plt.xticks(ticks=range(len(slope)), labels=labels, rotation=45, ha='right')  # Add combined labels
    plt.xlabel('k and slope')
    plt.ylabel('Total Erosion')
    plt.title('Linear slope increase, linear erosion rate increase -> ?')
    plt.grid(True, linestyle='--')
    plt.tight_layout()  # Adjust layout to fit rotated labels
    plt.savefig('../data/exp_incr_slope_incr_eros.png')

if __name__ == "__main__":
    """
    Calls the different experiments.
    """

    # exp_erosion_rate()
    # exp_flow_rate()
    # exp_slope()
    exp_incr_flow_decr_eros()
    # exp_incr_slope_incr_eros()





    