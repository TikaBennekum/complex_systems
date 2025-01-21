import numpy as np
import cv2

class Cell:
    def __init__(self, water=0, ground_height=0):
        self.ground_height = ground_height  # Ground height in the cell
        self.water = water  # 0: no water, 1: water


class CA:
    def __init__(self, width, height, ground_height):
        self.width = width
        self.height = height
        self.grid = [[Cell(ground_height=ground_height - (i * 0.1)) for _ in range(width)] for i in range(height)]
        self.grid[0][width // 2].water = 1  # Fill the center cell at the top with water

    def apply_rules(self, i, j, previous_grid):
        """Apply rules based on the states in the previous grid."""
        current_cell = self.grid[i][j]
        previous_cell = previous_grid[i][j]
        neighbors = []
        indices = []

        for di, dj in [(0, -1), (0, 1), (1, 0), (-1, 0)]:  # Left, right, down, up
            ni, nj = i + di, j + dj
            if 0 <= ni < self.height and 0 <= nj < self.width:
                neighbors.append(previous_grid[ni][nj])
                indices.append((ni, nj))

        # Rule: Water spreads to current cell if a neighbor has water and elevation is valid
        if previous_cell.water == 0:  # Only apply rule if there's no water currently
            for cell in neighbors:
                if cell.water == 1 and cell.ground_height > current_cell.ground_height:
                    current_cell.water = 1
                    break

    def update_grid(self):
        # Create a copy of the grid to represent the previous state
        previous_grid = [[Cell(water=cell.water, ground_height=cell.ground_height) for cell in row] for row in self.grid]

        # Update the current grid based on the previous state
        for i in range(self.height):
            for j in range(self.width):
                self.apply_rules(i, j, previous_grid)

    def run_simulation(self, num_epochs, output_file):
        frames = []  # Store frames for video

        for generation in range(num_epochs):
            self.update_grid()

            # Create a frame for the video
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            for i in range(self.height):
                for j in range(self.width):
                    if self.grid[i][j].water == 1:
                        frame[i, j] = [255, 0, 0]  # Blue for water
                    else:
                        frame[i, j] = [255, 255, 255]  # Brown for no water

            # Convert the frame to the required format and append
            frames.append(frame)

        # Save all frames as a video
        if frames:
            height, width, _ = frames[0].shape
            out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
            for frame in frames:
                out.write(frame)
            out.release()
            print(f"Simulation saved to {output_file}")
        else:
            print("No frames to save!")



# Example usage
width, height, ground_height = 100, 100, 50
output_file = 'videos/water_simulation.mp4'
ca = CA(width, height, ground_height)
ca.run_simulation(50, output_file)

