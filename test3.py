import numpy as np
import cv2

class Cell:
    def __init__(self, water_present=0, water_height=0, ground_height=0):
        self.ground_height = ground_height  # Ground height in the cell
        self.water_present = water_present  # 1: water present, 0: no water
        self.water_height = water_height  # Height of the water in the cell


class CA:
    def __init__(self, width, height, ground_height):
        self.width = width
        self.height = height
        # Initialize the grid with cells, each having a ground height
        self.grid = [[Cell(ground_height=ground_height - (i * 0.1)) for _ in range(width)] for i in range(height)]
        # Set the center cell at the top row with some water
        self.grid[0][width // 2].water_present = 1
        self.grid[0][width // 2].water_height = 50  # Arbitrary water height for the top-center cell

    def apply_rules(self, i, j, previous_grid):
        """Apply the water flow rules based on the previous grid state."""
        current_cell = self.grid[i][j]
        previous_cell = previous_grid[i][j]
        neighbors = []
        indices = []
        slopes = []
        
        # Collect neighbors and calculate slopes
        for di, dj in [(0, -1), (0, 1), (1, 0), (-1, 0)]:  # Left, right, down, up
            ni, nj = i + di, j + dj
            if 0 <= ni < self.height and 0 <= nj < self.width:
                neighbors.append(previous_grid[ni][nj])
                indices.append((ni, nj))
                # Calculate the slope to each neighbor (elevation difference)
                if di == 0 and dj == 0:
                    slopes.append(0)  # No slope to itself
                else:
                    slope = (previous_cell.ground_height - previous_grid[ni][nj].ground_height) / np.sqrt(di**2 + dj**2)
                    slopes.append(slope)

        # Rule 1: Water flows downhill into neighbors with positive slope
        positive_slopes = [s for s in slopes if s > 0]
        if positive_slopes:
            # Route water to neighbors with positive slopes based on the magnitude of the slope
            total_positive_slope = sum(s**0.5 for s in positive_slopes)  # n = 0.5 as default
            for idx, slope in enumerate(slopes):
                if slope > 0:
                    proportion = (slope**0.5) / total_positive_slope
                    discharge = previous_cell.water_height * proportion
                    ni, nj = indices[idx]
                    # Only add water if discharge > 0 (no water is added if there's no flow)
                    if discharge > 0:
                        self.grid[ni][nj].water_present = 1
                        self.grid[ni][nj].water_height += discharge
                        current_cell.water_height -= discharge  # Decrease water from the current cell
        # Rule 2: If none of the slopes are positive but at least one is zero, distribute water evenly
        elif 0 in slopes:
            zero_slopes_indices = [idx for idx, slope in enumerate(slopes) if slope == 0]
            if zero_slopes_indices:
                # Evenly distribute the water to neighbors with zero slope
                num_zero_slopes = len(zero_slopes_indices)
                if num_zero_slopes > 0:
                    discharge = previous_cell.water_height / num_zero_slopes
                    for idx in zero_slopes_indices:
                        ni, nj = indices[idx]
                        # Only add water if discharge > 0 (no water is added if there's no flow)
                        if discharge > 0:
                            self.grid[ni][nj].water_present = 1
                            self.grid[ni][nj].water_height += discharge
                    current_cell.water_height -= discharge * num_zero_slopes  # Decrease water from the current cell
        # Rule 3: If all slopes are negative, distribute water proportionally to the slopes
        else:
            total_negative_slope = sum(abs(s)**-0.5 for s in slopes if s < 0)  # n = -0.5 for negative slopes
            for idx, slope in enumerate(slopes):
                if slope < 0:
                    proportion = (abs(slope)**-0.5) / total_negative_slope
                    discharge = previous_cell.water_height * proportion
                    ni, nj = indices[idx]
                    # Only add water if discharge > 0 (no water is added if there's no flow)
                    if discharge > 0:
                        self.grid[ni][nj].water_present = 1
                        self.grid[ni][nj].water_height += discharge
                    current_cell.water_height -= discharge  # Decrease water from the current cell

    def update_grid(self):
        """Update the grid based on the previous state."""
        # Create a copy of the grid to represent the previous state
        previous_grid = [[Cell(water_present=cell.water_present, water_height=cell.water_height, ground_height=cell.ground_height)
                          for cell in row] for row in self.grid]

        # Apply the rules to update the current grid based on the previous state
        for i in range(self.height):
            for j in range(self.width):
                self.apply_rules(i, j, previous_grid)

    def run_simulation(self, num_epochs, output_file):
        """Run the simulation for a number of epochs and save the results as a video."""
        frames = []  # Store frames for video

        for generation in range(num_epochs):
            self.update_grid()

            # Create a frame for the video showing water presence and height
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            for i in range(self.height):
                for j in range(self.width):
                    # Display water as blue and non-water as brown
                    if self.grid[i][j].water_present == 1:
                        frame[i, j] = [255, 0, 0]  # Blue for water
                    else:
                        frame[i, j] = [255, 255, 255]  # Brown for no water

            # Append the frame for the video
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
