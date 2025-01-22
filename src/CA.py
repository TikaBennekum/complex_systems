import numpy as np
import cv2
ALL_NEIGHBORS = [(0, -1), (0, 1), (1, 0), (-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
class Cell:
    def __init__(self, water_height=0, ground_height=0):
        self.ground_height = ground_height  # Ground height in the cell
        self.water_height = water_height  # Height of the water in the cell


class CA:
    
    def __init__(self, width: int, height: int, ground_height: int):
        self.width = width
        self.height = height
        # Initialize the grid with cells, each having a ground height
        self.grid = [[Cell(ground_height=ground_height - (i * 0.1)) for _ in range(width)] for i in range(height)]
        # Set the center cell at the top row with some water
        self.grid[0][width // 2].water_height = 50  # Arbitrary water height for the top-center cell
            
    def apply_rules(self, i: int, j: int, previous_grid: list[list[Cell]]):
        """Apply the water flow rules based on the previous grid state."""
        current_cell = self.grid[i][j]
        previous_cell = previous_grid[i][j]

        if previous_cell.water_height == 0:
            return  # Skip cells without water

        neighbors = []
        indices = []
        slopes = []

        # Collect neighbors and calculate slopes
        for di, dj in ALL_NEIGHBORS:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.height and 0 <= nj < self.width:
                neighbor = previous_grid[ni][nj]
                neighbors.append(neighbor)
                indices.append((ni, nj))
                # Calculate slope to the neighbor
                distance = np.sqrt(di**2 + dj**2)
                slope = (previous_cell.ground_height + previous_cell.water_height -
                        (neighbor.ground_height + neighbor.water_height)) / distance
                slopes.append(slope)

        # Distribute water based on slopes
        total_positive_slope = sum(s for s in slopes if s > 0)
        if total_positive_slope > 0:
            for idx, slope in enumerate(slopes):
                if slope > 0:
                    proportion = slope / total_positive_slope
                    discharge = previous_cell.water_height * proportion
                    ni, nj = indices[idx]
                    self.grid[ni][nj].water_height += discharge
                    current_cell.water_height -= discharge
                    

        # If no positive slopes, distribute water evenly to zero-slope neighbors
        elif 0 in slopes:
            zero_slope_indices = [idx for idx, slope in enumerate(slopes) if slope == 0]
            if zero_slope_indices:
                discharge = previous_cell.water_height / len(zero_slope_indices)
                for idx in zero_slope_indices:
                    ni, nj = indices[idx]
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
                        self.grid[ni][nj].water_height += discharge
                        current_cell.water_height -= discharge

    def update_grid(self):
        """Update the grid based on the previous state."""
        # Create a copy of the grid to represent the previous state
        previous_grid = [[Cell(water_height=cell.water_height, ground_height=cell.ground_height)
                          for cell in row] for row in self.grid]

        # Apply the rules to update the current grid based on the previous state
        for i in range(self.height):
            for j in range(self.width):
                self.apply_rules(i, j, previous_grid)

    def run_simulation(self, num_epochs: int, output_file: None|str =None, show_live: bool=True, window_scale: int=5):
        """
        Run the simulation for a number of epochs, display it live, 
        and optionally save the results as a video.
        
        Args:
            num_epochs (int): Number of epochs to run the simulation.
            output_file (str): File path to save the video (optional).
            show_live (bool): Whether to display the simulation live.
            window_scale (float): Scale factor for the display window size (e.g., 2.0 for 2x size).
        """
        frames = []  # Store frames for video

        for generation in range(num_epochs):
            self.update_grid()

            # Create a frame for the video showing water presence and height
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            for i in range(self.height):
                for j in range(self.width):
                    # Display water as blue and non-water as brown
                    if self.grid[i][j].water_height > 0:
                        frame[i, j] = [255, 0, 0]  # Blue for water
                    else:
                        frame[i, j] = [255, 255, 255]  # Brown for no water

            # Optionally resize the frame for a larger display window
            if show_live and window_scale != 1:
                scaled_size = (int(frame.shape[1] * window_scale), int(frame.shape[0] * window_scale))
                display_frame = cv2.resize(frame, scaled_size, interpolation=cv2.INTER_LINEAR)
            else:
                display_frame = frame

            # Optionally show the simulation live
            if show_live:
                cv2.imshow("Simulation", display_frame)
                # Wait for a short period to control the frame rate
                # and allow the user to exit the simulation by pressing 'q'
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    print("Simulation interrupted by user.")
                    break

            # Append the original frame for the video
            frames.append(frame)

        # Close the display window
        if show_live:
            cv2.destroyAllWindows()

        # Save all frames as a video if output_file is provided
        if output_file and frames:
            height, width, _ = frames[0].shape
            out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
            for frame in frames:
                out.write(frame)
            out.release()
            print(f"Simulation saved to {output_file}")
        elif not frames:
            print("No frames to save!")