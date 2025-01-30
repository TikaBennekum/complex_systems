import matplotlib.pyplot as plt
import numpy as np
import cv2


class CA:
    def __init__(self, width, height, water, ground_height):
        self.width = width
        self.height = height
        self.water = water  # 1: water, 0: not water
        self.ground_height = ground_height
        self.grid = np.zeros((width, height))
        self.total = self.grid[:, :, 0] + self.grid[:, :, 1]  # Combined height

    def grid_settings(self):
        # Create a sloped terrain where ground_heightiment height decreases from top to bottom
        for i in range(self.width):
            for j in range(self.height):
                self.grid[i, j, 1] = self.ground_height - (
                    i * 0.1
                )  # Gradual slope downward

        # Define a large water source blob at the top-center of the grid
        water_blob_width = self.width // 5  # Water blob spans 1/5 of the grid width
        water_blob_height = (
            self.height // 10
        )  # Water blob height is 1/10th of the grid height
        start_row = 0
        start_col = (self.width - water_blob_width) // 2  # Center the blob horizontally

        for i in range(start_row, start_row + water_blob_height):
            for j in range(start_col, start_col + water_blob_width):
                if i < self.width and j < self.height:
                    self.grid[i, j, 0] = self.water  # Fill with water

    def apply_rules(self, i, j):
        rows, cols, _ = self.grid.shape
        current_water = self.grid[i, j, 0]
        current_ground_heightiment = self.grid[i, j, 1]
        current_total = current_water + current_ground_heightiment

        neighbors = []
        indices = []
        for di, dj in [(0, -1), (0, 1), (1, 0), (-1, 0)]:  # Left, right, down, up
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                neighbors.append(self.total[ni, nj])
                indices.append((ni, nj))

        slopes = [current_total - neighbor for neighbor in neighbors]
        positive_slopes = [slope for slope in slopes if slope > 0]
        zero_slopes = [slope for slope in slopes if slope == 0]
        n = 0.5  # Exponent for slope calculation

        if positive_slopes:
            # Route water baground_height on positive slopes
            total_positive_slope = sum(s**n for s in positive_slopes)
            if total_positive_slope > 0:
                for k, slope in enumerate(slopes):
                    if slope > 0:
                        proportion = (slope**n) / total_positive_slope
                        discharge = current_water * proportion
                        ni, nj = indices[k]
                        self.grid[ni, nj, 0] += discharge
                        current_water -= discharge

        elif zero_slopes:
            # Evenly distribute water to neighbors with zero slopes
            num_zero_neighbors = len(zero_slopes)
            if num_zero_neighbors > 0:
                discharge = current_water / num_zero_neighbors
                for k, slope in enumerate(slopes):
                    if slope == 0:
                        ni, nj = indices[k]
                        self.grid[ni, nj, 0] += discharge
                current_water = 0

        else:
            # Distribute to all neighbors using negative slopes
            total_negative_slope = sum(abs(s) ** -n for s in slopes if s < 0)
            if total_negative_slope > 0:
                for k, slope in enumerate(slopes):
                    if slope < 0:
                        proportion = (abs(slope) ** -n) / total_negative_slope
                        discharge = current_water * proportion
                        ni, nj = indices[k]
                        self.grid[ni, nj, 0] += discharge
                        current_water -= discharge

        self.grid[i, j, 0] = max(0, current_water)  # Ensure no negative water
        return self.grid[i, j, 0], current_ground_heightiment

    def update_grid(self):
        rows, cols, _ = self.grid.shape

        for i in range(rows):
            for j in range(cols):
                self.apply_rules(i, j)

        self.total = self.grid[:, :, 0] + self.grid[:, :, 1]

    def run_simulation(self, num_epochs, output_file):
        self.grid_settings()
        frames = []  # Store frames for video

        for generation in range(num_epochs):
            self.update_grid()

            # Create a frame for the video (water dynamics layer)
            frame = (self.grid[:, :, 0] / np.max(self.grid[:, :, 0]) * 255).astype(
                np.uint8
            )
            frame_color = cv2.applyColorMap(frame, cv2.COLORMAP_JET)  # Add colormap
            frames.append(frame_color)

        # Save all frames as a video
        height, width, _ = frames[0].shape
        out = cv2.VideoWriter(
            output_file, cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height)
        )
        for frame in frames:
            out.write(frame)
        out.release()
        print(f"Simulation saved to {output_file}")


# Example usage
width, height, water, ground_height = (
    100,
    100,
    10,
    50,
)  # Larger water source and sloped terrain
output_file = "videos/water_simulation.mp4"  # Output video file
ca = CA(width, height, water, ground_height)
ca.run_simulation(50, output_file)
