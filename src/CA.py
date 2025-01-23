from typing import Any
import numpy as np
from numpy.typing import NDArray
import cv2
from dataclasses import dataclass

ALL_NEIGHBORS = [(0, -1), (0, 1), (1, 0), (-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
BOTTOM_NEIGHBORS = [(1, 0), (1, -1), (1, 1)]
BOTTOM_SIDE_NEIGHBORS = [(1, 0), (1, -1), (1, 1),(0, -1), (0, 1)]

EROSION_K = 0.1
EROSION_C = 0.3
EROSION_EXPONENT = 2.5
N = 0.5

def visualise_height(grid):
    """ Prints out a grid of heights (numbers) to visualise the initial terrain. """
    for row in grid:
        heights = [np.round(cell.ground_height, 2).item() for cell in row]  # Extract ground heights for each cell in the row
        print(heights)


NUM_CELL_FLOATS = 2
GROUND_HEIGHT = 0
WATER_HEIGHT = 1

class CA:
    def __init__(self, width: int, height: int, initial_state: NDArray = None, neighbor_list = BOTTOM_NEIGHBORS):
        self.width = width
        self.height = height
        # Initialize the grid with cells, each having a ground height
        if initial_state is not None:
            self.grid = initial_state.copy()	
        else:
            self.grid = np.zeros([height, width, NUM_CELL_FLOATS])                
        self.enforce_boundary()
        
        self.neighbor_list = neighbor_list
        
    def enforce_boundary(self):        
        # Set the center cell at the top row with some water
        self.grid[0, self.width // 2, WATER_HEIGHT] = 50  # Arbitrary water height for the top-center cell
        self.grid[-1, :, WATER_HEIGHT] = 0  # Arbitrary water height for the top-center cell
        self.grid[0, :, GROUND_HEIGHT] = self.grid[0,-1, GROUND_HEIGHT]  # Arbitrary water height for the top-center cell
        self.grid[-1, :, GROUND_HEIGHT] = 0  # Arbitrary water height for the top-center cell
        
            
    def apply_rules(self, i: int, j: int, previous_grid: NDArray):
        """Apply the water flow rules based on the previous grid state."""
        current_cell = self.grid[i][j]
        previous_cell = previous_grid[i][j]

        if previous_cell[WATER_HEIGHT] == 0:
            return  # Skip cells without water

        indices, slopes = self.create_indices_slopes(i, j, previous_grid, previous_cell)

        # Distribute water based on slopes
        total_positive_slope = sum(s**N for s in slopes if s > 0)
        if total_positive_slope > 0:
            for idx, slope in enumerate(slopes):
                if slope > 0:
                    proportion = slope**N / total_positive_slope
                    discharge = previous_cell[WATER_HEIGHT] * proportion
                    ni, nj = indices[idx]
                    self.grid[ni][nj][WATER_HEIGHT] += discharge
                    current_cell[WATER_HEIGHT] -= discharge
                    current_cell[GROUND_HEIGHT] -= self.erosion_rule(EROSION_K, discharge, slope)
                    self.grid[ni,nj,GROUND_HEIGHT] += self.erosion_rule(EROSION_K, discharge, slope)
                    

        # If no positive slopes, distribute water evenly to zero-slope neighbors
        elif 0 in slopes:
            zero_slope_indices = [idx for idx, slope in enumerate(slopes) if slope == 0]
            if zero_slope_indices:
                discharge = previous_cell[WATER_HEIGHT] / len(zero_slope_indices)
                for idx in zero_slope_indices:
                    ni, nj = indices[idx]
                    self.grid[ni][nj][WATER_HEIGHT] += discharge
                    current_cell[GROUND_HEIGHT] -=  self.erosion_rule(EROSION_K, discharge, slope)
                    self.grid[ni,nj,GROUND_HEIGHT] += self.erosion_rule(EROSION_K, discharge, slope)
                current_cell[WATER_HEIGHT] -= discharge * len(zero_slope_indices)

        # If all slopes are negative, distribute water proportionally to their magnitudes
        else:
            total_negative_slope = sum(abs(s)**-N for s in slopes if s < 0)
            if total_negative_slope > 0:
                for idx, slope in enumerate(slopes):
                    if slope < 0:
                        proportion = abs(slope)**-N / total_negative_slope
                        discharge = previous_cell[WATER_HEIGHT] * proportion
                        ni, nj = indices[idx]
                        self.grid[ni][nj][WATER_HEIGHT] += discharge
                        current_cell[WATER_HEIGHT] -= discharge
                        current_cell[GROUND_HEIGHT] -= self.erosion_rule(EROSION_K, discharge, slope)
                        self.grid[ni,nj,GROUND_HEIGHT] += self.erosion_rule(EROSION_K, discharge, slope)
                        
           
    def erosion_rule(self, K, Q, S=0, C=EROSION_C):
        if Q <= 0:
            return 0
        return K* np.sign(S) * np.minimum(C, np.power(Q*(np.abs(S) + C), EROSION_EXPONENT))

    def create_indices_slopes(self, i: int, j: int, previous_grid: NDArray, previous_cell: NDArray) -> tuple[list[tuple[int, int]], list[float]]:
        neighbors: list[NDArray] = []
        indices: list[tuple[int, int]] = []
        slopes: list[float] = []

        # Collect neighbors and calculate slopes
        for di, dj in self.neighbor_list:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.height and 0 <= nj < self.width:
                neighbor = previous_grid[ni][nj]
                neighbors.append(neighbor)
                indices.append((ni, nj))
                # Calculate slope to the neighbor
                distance = np.sqrt(di**2 + dj**2)
                slope = (previous_cell[GROUND_HEIGHT]  -
                        (neighbor[GROUND_HEIGHT] )) / distance
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
                
        # diff = self.grid[:,:,GROUND_HEIGHT] - previous_grid[:,:,GROUND_HEIGHT]
        # print( np.max(diff), np.min(diff), np.mean(diff))
    

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

            if(output_file or show_live):
                # Create a frame for the video showing water presence and height
                frame = np.zeros((self.height, 2*self.width, 3), dtype=np.uint8)

                h_range = np.max(self.grid[:,:,GROUND_HEIGHT]) - np.min(self.grid[:,:,GROUND_HEIGHT])
                frame[:, self.width:, 2] = np.floor(255/h_range*(self.grid[:,:,GROUND_HEIGHT] - np.min(self.grid[:,:,GROUND_HEIGHT])))
                
                # log_height = np.log10(np.maximum(1e-6, self.grid[:,:,WATER_HEIGHT]))
                log_height = np.sqrt(np.maximum(1e-6, self.grid[:,:,WATER_HEIGHT]))
                h_range = np.max(log_height) - np.min(log_height)
                frame[:, :self.width, 0] = np.floor(155/h_range*(log_height - np.min(log_height)))
                
                
                frame[:, :self.width, 0] += (50*(self.grid[:,:,WATER_HEIGHT] > 0)).astype(np.uint8)
                frame[:, :self.width, 0] += (50*(self.grid[:,:,WATER_HEIGHT] > 1e-3)).astype(np.uint8)


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