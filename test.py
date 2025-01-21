import matplotlib.pyplot as plt
import numpy as np

class CA:
    """
    Cellular Automaton for simulating water flow over a sloped surface.

    Attributes:
        width (int): Width of the grid.
        height (int): Height of the grid.
        water (float): Initial water level in the blob.
        sed (float): Base sediment height at the top of the slope.
        grid (np.ndarray): 3D array storing water and sediment layers.
        total (np.ndarray): Combined height of water and sediment layers.
    """

    def __init__(self, width, height, water, sed):
        """
        Initialize the CA simulation with the given dimensions and parameters.
        """
        self.width = width
        self.height = height
        self.water = water
        self.sed = sed
        self.grid = np.zeros((width, height, 2))  # Third dimension: [water, sediment]
        self.total = self.grid[:, :, 0] + self.grid[:, :, 1]  # Combined height for total elevation

    def grid_settings(self):
        """
        Configure the initial sediment slope and water blob.
        """
        # Set sediment height with a downward slope
        for j in range(self.height):
            self.grid[:, j, 1] = self.sed - j

        # Initialize a square blob of water at the top center
        blob_size = self.height // 10  # Size of the water blob
        start_col = (self.width - blob_size) // 2  # Center the blob horizontally
        for j in range(blob_size):
            for i in range(start_col, start_col + blob_size):
                if i < self.width and j < self.height:  # Ensure within bounds
                    self.grid[i, j, 0] = self.water

    def apply_rules(self, i, j):
        """
        Calculate water redistribution for a given cell.

        Args:
            i (int): Row index of the cell.
            j (int): Column index of the cell.

        Returns:
            tuple: Updated water and sediment levels for the cell.
        """
        rows, cols, _ = self.grid.shape
        current_water = self.grid[i, j, 0]
        current_sediment = self.grid[i, j, 1]
        current_total = current_water + current_sediment

        # Gather neighbors' total heights and slopes
        neighbors, slopes = [], []
        for di, dj in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                neighbor_total = self.total[ni, nj]
                slope = current_total - neighbor_total
                if abs(di) + abs(dj) == 2:  # Adjust for diagonal neighbors
                    slope /= 2**0.5
                neighbors.append((ni, nj))
                slopes.append(slope)

        # Redistribute water based on slopes
        positive_slopes = [slope for slope in slopes if slope > 0]
        if positive_slopes:
            total_positive_slope = sum(positive_slopes)
            for index, slope in enumerate(slopes):
                if slope > 0:
                    discharge = (slope / total_positive_slope) * current_water
                    ni, nj = neighbors[index]
                    current_water -= discharge
                    self.grid[ni, nj, 0] += discharge
        else:
            # Evenly distribute water to neighbors with zero or negative slopes
            total_weight = sum(abs(slope) ** -0.5 for slope in slopes if slope < 0)
            if total_weight > 0:
                for index, slope in enumerate(slopes):
                    if slope < 0:
                        ni, nj = neighbors[index]
                        discharge = (abs(slope) ** -0.5 / total_weight) * current_water
                        self.grid[ni, nj, 0] += discharge
                        current_water -= discharge

        return max(0, current_water), current_sediment

    def update_grid(self):
        """
        Update the entire grid based on the rules for water redistribution.
        """
        next_grid = np.copy(self.grid)
        rows, cols, _ = self.grid.shape
        for i in range(rows):
            for j in range(cols):
                new_water, new_sediment = self.apply_rules(i, j)
                next_grid[i, j, 0] = new_water
                next_grid[i, j, 1] = new_sediment

        self.total = next_grid[:, :, 0] + next_grid[:, :, 1]  # Update total heights
        return next_grid

    def run_simulation(self, num_epochs):
        """
        Run the CA simulation and visualize water flow over time.

        Args:
            num_epochs (int): Number of simulation steps to run.
        """
        self.grid_settings()  # Initialize the grid
        plt.figure(figsize=(10, 10))  # Larger figure size for clarity
        
        for generation in range(num_epochs):
            self.grid = self.update_grid()  # Update grid state
            water_mask = self.grid[:, :, 0] > 0  # Binary mask for water cells

            # Plot the current state
            plt.clf()
            plt.imshow(water_mask, cmap='Blues', interpolation='nearest')
            plt.title(f'Water Flow Simulation - Step {generation + 1}')
            plt.pause(0.1)

        plt.show()


# Run the simulation with example parameters
width, height, water, sed = 100, 100, 100, 10
ca = CA(width, height, water, sed)
ca.run_simulation(50)
