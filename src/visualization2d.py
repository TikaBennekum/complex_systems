"""
Course: Complex systems
Names: Marvin Frommer, Wessel Beumer, Paul Jungnickel, Tika van Bennekum

File description:
    This file contains the code to visualis our system in a 2D video.
"""

import numpy as np
from CA import *
from constants import *


def compute_frame(grid):
    # Create a frame for the video showing water presence and height
    height = grid.shape[0]
    width = grid.shape[1]
    frame = np.zeros((height, 2 * width, 3), dtype=np.uint8)

    h_range = np.max(grid[:, :, GROUND_HEIGHT]) - np.min(grid[:, :, GROUND_HEIGHT])
    frame[:, width:, 2] = np.floor(
        255 / h_range * (grid[:, :, GROUND_HEIGHT] - np.min(grid[:, :, GROUND_HEIGHT]))
    )

    # log_height = np.log10(np.maximum(1e-6, grid[:,:,WATER_HEIGHT]))
    log_height = np.sqrt(np.maximum(1e-6, grid[:, :, WATER_HEIGHT]))
    h_range = np.max(log_height) - np.min(log_height)
    frame[:, :width, 0] = np.floor(155 / h_range * (log_height - np.min(log_height)))

    frame[:, :width, 0] += (50 * (grid[:, :, WATER_HEIGHT] > 0)).astype(np.uint8)
    frame[:, :width, 0] += (50 * (grid[:, :, WATER_HEIGHT] > 1e-3)).astype(np.uint8)

    return frame


def save_video(saved_grids, output_file):
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"mp4v"), 60, (saved_grids.shape[2], 2 * saved_grids.shape[1]))  # type: ignore
    for grid in saved_grids:
        frame = compute_frame(grid)
        out.write(frame)
    out.release()
    print(f"Simulation saved to {output_file}")


def stream_video(saved_grids, scale=10, fps=100):

    for grid in saved_grids:
        frame = compute_frame(grid)
        if scale != 1:
            scaled_size = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
            frame = cv2.resize(frame, scaled_size, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Simulation", frame)
        if cv2.waitKey(1000 // fps) & 0xFF == ord("q"):
            print("Simulation interrupted by user.")
            break
