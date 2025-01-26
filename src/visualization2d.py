import numpy as np
from CA import *
from constants import *


def compute_frame(grid):
    # Create a frame for the video showing water presence and height
    height = grid.shape[0]
    width = grid.shape[1]
    frame = np.zeros((height, 2*width, 3), dtype=np.uint8)

    h_range = np.max(grid[:,:,GROUND_HEIGHT]) - np.min(grid[:,:,GROUND_HEIGHT])
    frame[:, width:, 2] = np.floor(255/h_range*(grid[:,:,GROUND_HEIGHT] - np.min(grid[:,:,GROUND_HEIGHT])))

    # log_height = np.log10(np.maximum(1e-6, grid[:,:,WATER_HEIGHT]))
    log_height = np.sqrt(np.maximum(1e-6, grid[:,:,WATER_HEIGHT]))
    h_range = np.max(log_height) - np.min(log_height)
    frame[:, :width, 0] = np.floor(155/h_range*(log_height - np.min(log_height)))


    frame[:, :width, 0] += (50*(grid[:,:,WATER_HEIGHT] > 0)).astype(np.uint8)
    frame[:, :width, 0] += (50*(grid[:,:,WATER_HEIGHT] > 1e-3)).astype(np.uint8)
    
    return frame
    
    
def save_video(saved_grids, output_file):
        height, width, _ = compute_frame(saved_grids[0]).shape
        out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 60, (width, height))  # type: ignore
        for grid in saved_grids:
            out.write(compute_frame(grid))
        out.release()
        print(f"Simulation saved to {output_file}")
    