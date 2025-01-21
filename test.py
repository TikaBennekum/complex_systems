import matplotlib.pyplot as plt
import numpy as np

class CA(object):

    def __init__(self, width, height, water, sed):
        self.width = width
        self.height = height
        self.water = water
        self.sed = sed
        self.grid = np.zeros((width, height, 2))  # Third dimension: [water, sediment]
        self.total = self.grid[:, :, 0] + self.grid[:, :, 1]  # Combined height for total elevation

    def grid_settings(self):
        # Create a downward slope for sediment heights
        for i in range(self.width):
            for j in range(self.height):
                self.grid[i, j, 1] = self.sed - j  # Regular downward slope for sediment

        # Define the size of the water blob
        water_blob_size = 5  # Size of the square water blob
        start_row = 0
        start_col = (self.width - water_blob_size) // 2  # Center the blob horizontally

        # Fill the top middle with water (square blob)
        for j in range(start_row, start_row + water_blob_size):
            for i in range(start_col, start_col + water_blob_size):
                if i < self.width and j < self.height:  # Ensure we don't go out of bounds
                    self.grid[i, j, 0] = self.water  # Fill with water
                
    def apply_rules(self, i, j):
        rows, cols, _ = self.grid.shape
        current_water = self.grid[i, j, 0]
        current_sediment = self.grid[i, j, 1]
        current_total = current_water + current_sediment

        # Collect neighbors' total heights
        neighbors = []
        for di, dj in [(0, -1), (0, 0), (0, 1)]:  # Left, center, right neighbors
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                neighbors.append(self.total[ni, nj])  # Store total height of neighbors

        # Ensure neighbors have been collected correctly
        while len(neighbors) < 3:  # Ensure we have exactly 3 neighbors
            neighbors.append(0)  # Append a height of 0 for missing neighbors

        slopes = [(current_total - neighbors[0]),  # Left
                (current_total - neighbors[1]),  # Center
                (current_total - neighbors[2])]  # Right

        positive_slopes = [slope for slope in slopes if slope > 0]
        zero_slopes = [slope for slope in slopes if slope == 0]

        if positive_slopes:
            # Route water according to positive slopes
            total_positive_slope = sum(positive_slopes)
            if total_positive_slope > 0:  # Ensure we don't divide by zero
                Q0 = current_water  # Total discharge
                for index, slope in enumerate(slopes):
                    if slope > 0:  # Only route water to neighbors with positive slopes
                        discharge = (slope / total_positive_slope) * Q0
                        # Transfer water to the neighbor cell
                        if index == 0:  # Left neighbor
                            if j > 0:  # Ensure we don't go out of bounds
                                current_water -= discharge
                                self.grid[i, j - 1, 0] += discharge
                        elif index == 1:  # Center neighbor
                            # Optionally handle sediment transfer logic here if needed
                            pass
                        elif index == 2:  # Right neighbor
                            if j < cols - 1:  # Ensure we don't go out of bounds
                                current_water -= discharge
                                self.grid[i, j + 1, 0] += discharge
        elif zero_slopes:
            # Distribute evenly to neighbors with zero slopes
            num_zero_neighbors = len(zero_slopes)
            if num_zero_neighbors > 0:
                Q0 = current_water / num_zero_neighbors
                for index, slope in enumerate(slopes):
                    if slope == 0:
                        if index == 0:  # Left neighbor
                            if j > 0:  # Ensure we don't go out of bounds
                                self.grid[i, j - 1, 0] += Q0
                        elif index == 1:  # Center neighbor
                            # Handle sediment logic here if needed
                            pass
                        elif index == 2:  # Right neighbor
                            if j < cols - 1:  # Ensure we don't go out of bounds
                                self.grid[i, j + 1, 0] += Q0
        else:
            # All slopes are negative, distribute according to the formula
            for index, slope in enumerate(slopes):
                if index == 0:  # Left neighbor
                    if slope < 0 and j > 0:
                        self.grid[i, j - 1, 0] += current_water * (1 / (1 - slope ** 0.5))  # Example logic
                elif index == 1:  # Center neighbor
                    # You can decide how to handle center neighbor
                    pass
                elif index == 2:  # Right neighbor
                    if slope < 0 and j < cols - 1:
                        self.grid[i, j + 1, 0] += current_water * (1 / (1 - slope ** 0.5))  # Example logic

        # Return the new state of water and sediment
        return max(0, current_water), current_sediment  # Ensure no negative values


    def update_grid(self):
        next_grid = np.copy(self.grid)  # Create a copy for updates
        rows, cols, _ = self.grid.shape

        for i in range(rows):
            for j in range(cols):
                new_water, new_sediment = self.apply_rules(i, j)
                next_grid[i, j, 0] = new_water
                next_grid[i, j, 1] = new_sediment

        # Update the combined total height
        self.total = next_grid[:, :, 0] + next_grid[:, :, 1]
        return next_grid

    def run_simulation(self, num_epochs):
        self.grid_settings()  # Initialize the grid
        fig, ax = plt.subplots()
        
        for generation in range(num_epochs):
            self.grid = self.update_grid()  # Update the grid
            
            # Visualize total height (water + sediment)
            img = ax.imshow(self.grid[:, :, 0] > 0, cmap='Blues', interpolation='nearest')  # Cells with water are blue
            plt.title('Cellular Automaton Simulation')
            ax.set_title(f'Generation: {generation}')
            plt.pause(0.1)

        plt.show()


# Example usage with increased resolution and larger water blob
width, height, water, sed = 100, 100, 50, 10  # Increased grid size and water amount
ca = CA(width, height, water, sed)
ca.run_simulation(10)  # Increased the number of generations for better visualization