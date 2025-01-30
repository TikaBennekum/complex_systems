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
        self.grid = [
            [Cell(ground_height=ground_height - (i * 0.1)) for _ in range(width)]
            for i in range(height)
        ]
        # Set the center cell at the top row with some water
        self.grid[0][width // 2].water_present = 1
        self.grid[0][
            width // 2
        ].water_height = 50  # Arbitrary water height for the top-center cell

    def apply_rules(self, i, j, previous_grid):
        """Apply the water flow rules based on the previous grid state."""
        current_cell = self.grid[i][j]
        previous_cell = previous_grid[i][j]

        if previous_cell.water_height == 0:
            return  # Skip cells without water

        neighbors = []
        indices = []
        slopes = []

        # Collect neighbors and calculate slopes
        for di, dj in [
            (0, -1),
            (0, 1),
            (1, 0),
            (-1, 0),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.height and 0 <= nj < self.width:
                neighbor = previous_grid[ni][nj]
                neighbors.append(neighbor)
                indices.append((ni, nj))
                # Calculate slope to the neighbor
                distance = np.sqrt(di**2 + dj**2)
                slope = (
                    previous_cell.ground_height
                    + previous_cell.water_height
                    - (neighbor.ground_height + neighbor.water_height)
                ) / distance
                slopes.append(slope)

        # Distribute water based on slopes
        total_positive_slope = sum(s for s in slopes if s > 0)
        if total_positive_slope > 0:
            for idx, slope in enumerate(slopes):
                if slope > 0:
                    proportion = slope / total_positive_slope
                    discharge = previous_cell.water_height * proportion
                    ni, nj = indices[idx]
                    self.grid[ni][nj].water_present = 1
                    self.grid[ni][nj].water_height += discharge
                    current_cell.water_height -= discharge

        # If no positive slopes, distribute water evenly to zero-slope neighbors
        elif 0 in slopes:
            zero_slope_indices = [idx for idx, slope in enumerate(slopes) if slope == 0]
            if zero_slope_indices:
                discharge = previous_cell.water_height / len(zero_slope_indices)
                for idx in zero_slope_indices:
                    ni, nj = indices[idx]
                    self.grid[ni][nj].water_present = 1
                    self.grid[ni][nj].water_height += discharge
                current_cell.water_height -= discharge * len(zero_slope_indices)

        # If all slopes are negative, distribute water proportionally to their magnitudes
        else:
            total_negative_slope = sum(abs(s) for s in slopes if s < 0)
            if total_negative_slope > 0:
                for idx, slope in enumerate(slopes):
                    if slope < 0:
                        proportion = abs(slope) / total_negative_slope
                        discharge = previous_cell.water_height * proportion
                        ni, nj = indices[idx]
                        self.grid[ni][nj].water_present = 1
                        self.grid[ni][nj].water_height += discharge
                        current_cell.water_height -= discharge

    def update_grid(self):
        """Update the grid based on the previous state."""
        # Create a copy of the grid to represent the previous state
        previous_grid = [
            [
                Cell(
                    water_present=cell.water_present,
                    water_height=cell.water_height,
                    ground_height=cell.ground_height,
                )
                for cell in row
            ]
            for row in self.grid
        ]

        # Apply the rules to update the current grid based on the previous state
        for i in range(self.height):
            for j in range(self.width):
                self.apply_rules(i, j, previous_grid)

    def run_simulation(self, num_epochs, output_file):
        """Run the simulation for a number of epochs and save the results as a video."""
        frames = []  # Store frames for video

        for generation in range(num_epochs):
            self.update_grid()

            # Create a frame for the video showing water height
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            max_water_height = max(
                cell.water_height for row in self.grid for cell in row
            )
            if max_water_height == 0:
                max_water_height = 1  # Avoid division by zero

            for i in range(self.height):
                for j in range(self.width):
                    cell = self.grid[i][j]
                    if cell.water_present == 1:
                        # Lighter blue intensity proportional to water height
                        intensity = min(
                            255, max(0, int(self.grid[i][j].water_height * 5))
                        )  # Scale and clamp intensity
                        frame[i, j] = [
                            intensity,
                            255 - intensity,
                            255,
                        ]  # Light blue gradient

                    else:
                        frame[i, j] = [255, 255, 255]  # Light brown for no water

            # Append the frame for the video
            frames.append(frame)

        # Save all frames as a video
        if frames:
            height, width, _ = frames[0].shape
            out = cv2.VideoWriter(
                output_file, cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height)
            )
            for frame in frames:
                out.write(frame)
            out.release()
            print(f"Simulation saved to {output_file}")
        else:
            print("No frames to save!")


# Example usage
width, height, ground_height = 100, 100, 50
output_file = "videos/water_simulation.mp4"
ca = CA(width, height, ground_height)
ca.run_simulation(60, output_file)
