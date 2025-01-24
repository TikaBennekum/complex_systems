import matplotlib.pyplot as plt
import numpy as np


class CellularAutomaton:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = np.random.choice([0, 1], size=(width, height))  # Randomly initialize grid with 0s and 1s

    def count_alive_neighbors(self, i, j):
        """
        Count the number of alive neighbors for the cell at position (i, j).
        """
        rows, cols = self.grid.shape
        alive_neighbors = 0

        for di, dj in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                alive_neighbors += self.grid[ni, nj]

        return alive_neighbors

    def apply_conway_rules(self, cell_state, alive_neighbors):
        """
        Apply Conway's Game of Life rules to determine the next state of a cell.
        """
        if cell_state == 1:  # Cell is alive
            if alive_neighbors < 2 or alive_neighbors > 3:
                return 0  # Dies
            else:
                return 1  # Lives
        else:  # Cell is dead
            if alive_neighbors == 3:
                return 1  # Becomes alive
            else:
                return 0  # Stays dead

    def update_grid(self):
        """
        Update the entire grid based on the rules of Conway's Game of Life.
        """
        next_grid = np.zeros_like(self.grid)
        rows, cols = self.grid.shape

        for i in range(rows):
            for j in range(cols):
                cell_state = self.grid[i, j]
                alive_neighbors = self.count_alive_neighbors(i, j)
                next_grid[i, j] = self.apply_conway_rules(cell_state, alive_neighbors)

        self.grid = next_grid

    def run_simulation(self, num_epochs):
        """
        Run the simulation for a given number of epochs.
        """
        fig, ax = plt.subplots()
        img = ax.imshow(self.grid, interpolation='nearest', cmap='binary')
        plt.title('Conway\'s Game of Life')

        for generation in range(num_epochs):
            self.update_grid()  # Update the grid
            img.set_data(self.grid)  # Update visualization
            plt.pause(0.5)
            ax.set_title(f'Generation: {generation}')
            fig.canvas.draw()

        plt.show()


# Example usage
width, height = 50, 50
ca = CellularAutomaton(width, height)
ca.run_simulation(50)
