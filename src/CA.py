from typing import Any
import numpy as np
from numpy.typing import NDArray
import cv2
ALL_NEIGHBORS = [(0, -1), (0, 1), (1, 0), (-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]

EROSION_CONSTANT = 0.01


NUM_CELL_FLOATS = 2
GROUND_HEIGHT = 0
WATER_HEIGHT = 1

class CA:
    
    def __init__(self, width: int, height: int, ground_height: int):
        self.width = width
        self.height = height
        # Initialize the grid with cells, each having a ground height
        self.grid = np.zeros([height, width, NUM_CELL_FLOATS])
        height_gradient = np.linspace(ground_height, 0, height)
        for i in range(height):
            self.grid[i, :, GROUND_HEIGHT] = height_gradient[i]
            
        self.enforce_boundary
        
    def enforce_boundary(self):        
        # Set the center cell at the top row with some water
        self.grid[0, self.width // 2, WATER_HEIGHT] = 50  # Arbitrary water height for the top-center cell
        
            
    def apply_rules(self, i: int, j: int, previous_grid: NDArray):
        """Apply the water flow rules based on the previous grid state."""
        current_cell = self.grid[i][j]
        previous_cell = previous_grid[i][j]

        if previous_cell[WATER_HEIGHT] == 0:
            return  # Skip cells without water

        indices, slopes = self.create_indices_slopes(i, j, previous_grid, previous_cell)

        # Distribute water based on slopes
        total_positive_slope = sum(s for s in slopes if s > 0)
        if total_positive_slope > 0:
            for idx, slope in enumerate(slopes):
                if slope > 0:
                    proportion = slope / total_positive_slope
                    discharge = previous_cell[WATER_HEIGHT] * proportion
                    ni, nj = indices[idx]
                    self.grid[ni][nj][WATER_HEIGHT] += discharge
                    current_cell[WATER_HEIGHT] -= discharge
                    current_cell[GROUND_HEIGHT] -=  EROSION_CONSTANT * discharge
                    

        # If no positive slopes, distribute water evenly to zero-slope neighbors
        elif 0 in slopes:
            zero_slope_indices = [idx for idx, slope in enumerate(slopes) if slope == 0]
            if zero_slope_indices:
                discharge = previous_cell[WATER_HEIGHT] / len(zero_slope_indices)
                for idx in zero_slope_indices:
                    ni, nj = indices[idx]
                    self.grid[ni][nj][WATER_HEIGHT] += discharge
                    current_cell[GROUND_HEIGHT] -=  EROSION_CONSTANT * discharge
                current_cell[WATER_HEIGHT] -= discharge * len(zero_slope_indices)

        # If all slopes are negative, distribute water proportionally to their magnitudes
        else:
            total_negative_slope = sum(abs(s) for s in slopes if s < 0)
            if total_negative_slope > 0:
                for idx, slope in enumerate(slopes):
                    if slope < 0:
                        proportion = abs(slope) / total_negative_slope
                        discharge = previous_cell[WATER_HEIGHT] * proportion
                        ni, nj = indices[idx]
                        self.grid[ni][nj][WATER_HEIGHT] += discharge
                        current_cell[WATER_HEIGHT] -= discharge
                        current_cell[GROUND_HEIGHT] -=  EROSION_CONSTANT * discharge

    def create_indices_slopes(self, i: int, j: int, previous_grid: NDArray, previous_cell: NDArray) -> tuple[list[tuple[int, int]], list[float]]:
        neighbors: list[NDArray] = []
        indices: list[tuple[int, int]] = []
        slopes: list[float] = []

        # Collect neighbors and calculate slopes
        for di, dj in ALL_NEIGHBORS:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.height and 0 <= nj < self.width:
                neighbor = previous_grid[ni][nj]
                neighbors.append(neighbor)
                indices.append((ni, nj))
                # Calculate slope to the neighbor
                distance = np.sqrt(di**2 + dj**2)
                slope = (previous_cell[GROUND_HEIGHT] + previous_cell[WATER_HEIGHT] -
                        (neighbor[GROUND_HEIGHT] + neighbor[WATER_HEIGHT])) / distance
                slopes.append(slope)
        return indices,slopes

    def create_indices_slopes(self, i: int, j: int, previous_grid: NDArray, previous_cell: NDArray) -> tuple[list[tuple[int, int]], list[float]]:
        neighbors: list[tuple[float,float]] = []
        indices: list[tuple[int, int]] = []
        slopes: list[float] = []

        # Collect neighbors and calculate slopes
        for di, dj in ALL_NEIGHBORS:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.height and 0 <= nj < self.width:
                neighbor = previous_grid[ni][nj]
                neighbors.append(neighbor)
                indices.append((ni, nj))
                # Calculate slope to the neighbor
                distance = np.sqrt(di**2 + dj**2)
                slope = (previous_cell[GROUND_HEIGHT] + previous_cell[WATER_HEIGHT] -
                        (neighbor[GROUND_HEIGHT] + neighbor[WATER_HEIGHT])) / distance
                slopes.append(slope)
        return indices,slopes

    def update_grid(self):
        """Update the grid based on the previous state."""
        # Create a copy of the grid to represent the previous state
        self.enforce_boundary()
        previous_grid = self.grid.copy()

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
        frames: list[NDArray[Any]] = []  # Store frames for video

        for _ in range(num_epochs):
            self.update_grid()

            # Create a frame for the video showing water presence and height
            frame = np.zeros((self.height, 2*self.width, 3), dtype=np.uint8)

            for i in range(self.height):
                for j in range(self.width):
                    # Display water as blue and non-water as brown
                    h_range = np.max(self.grid[:,:,GROUND_HEIGHT]) - np.min(self.grid[:,:,GROUND_HEIGHT])
                    frame[i, self.width + j] = [0,0, int(255/h_range*(self.grid[i][j][GROUND_HEIGHT] - np.min(self.grid[:,:,GROUND_HEIGHT])))] 
                    if self.grid[i][j][WATER_HEIGHT] > 0:
                        frame[i, j] = [255, 0, 0]  # Blue for water
                        

                    else:
                        frame[i, j] = [0,0,0]  # Brown for no water

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
            out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))  # type: ignore
            for frame in frames:
                out.write(frame)
            out.release()
            print(f"Simulation saved to {output_file}")
        elif not frames:
            print("No frames to save!")