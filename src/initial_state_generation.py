"""
Course: Complex systems
Names: Marvin Frommer, Wessel Beumer, Paul Jungnickel, Tika van Bennekum

File description:
    The initial state of the terrain of the system is generated in this file.
"""

import numpy as np
from noise import pnoise2

NUM_CELL_FLOATS = 2
GROUND_HEIGHT = 0
WATER_HEIGHT = 1


def generate_initial_slope(
    height: int,
    width: int,
    slope_top: float,
    slope_bot: float = 0,
    noise_amplitude: float = 0,
    noise_type="perlin",
):
    """Function that generates the initial slopes the terrain of the
    system starts out with."""

    grid = np.zeros([height, width, NUM_CELL_FLOATS])
    height_gradient = np.linspace(slope_top, slope_bot, height)
    for i in range(height):
        grid[i, :, GROUND_HEIGHT] = height_gradient[i]

    if noise_type == "white":
        grid[:, :, GROUND_HEIGHT] += np.random.normal(
            0, noise_amplitude, size=[height, width]
        )
    elif noise_type == "perlin":
        grid[:, :, GROUND_HEIGHT] += perlin_noise(height, width, noise_amplitude)

    return grid


def add_central_flow(grid, flow_amount):
    """Adds central flow to the grid."""
    grid[0, grid.shape[1] // 2, WATER_HEIGHT] = flow_amount

    return grid


def perlin_noise(height: int, width: int, max_amplitude: float):
    """
    Generates a grid of Perlin noise to be added to the grid of ground heights.

    Scale: i/x or j/x, where the x controls the scale of the noise,
            a higher x creates a smoother noise pattern, and a smaller x creates a more detailed pattern.
    Octaves: The number of layers of noise combined to create the final texture.
            A higher value makes the noise more complex.
    Persistence: Controls how much influence higher-frequency octaves have.
            Essentially, this determines the amplitude (height) of the noise at smaller scales.
    Lacunarity: Determines the frequency increase between octaves.
            Higher values give more intricate patterns.
    Repeatx/repeaty: The periodicity of the noise.
            Setting it to the width,height means no repeating patterns.
    Base: This is like a random seed, keeping the results consistent.

    The max amplitude decides the maximum amount of noise added to the ground eh
    """
    noise_map = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            noise_map[i][j] = pnoise2(
                i / 10,
                j / 10,
                octaves=3,
                persistence=0.4,
                lacunarity=2.2,
                repeatx=width,
                repeaty=height,
                base=25,
            )
    normalized_noise_map = (noise_map + 1) / 2
    scaled_map = normalized_noise_map * max_amplitude
    return scaled_map
