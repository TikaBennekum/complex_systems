import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from dataclasses import dataclass
from numba import jit
from inspect import getfullargspec
from copy import copy
from typing import Callable
from time import thread_time

import matplotlib
import imageio_ffmpeg

matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


### CONSTANTS
L = 10
FLOW_FRACTION = 0.010  # maximum fraction of water which can flow out of cell


### LAYERS
water_level = np.zeros([L, L])
water_level[L // 2, L // 2] = 10.0


### FUNCTIONS
@jit
def raise_water(water_level: NDArray) -> NDArray:
    return water_level + 1


@jit
def spread_water_simple(water_level: NDArray) -> NDArray:
    change_in_water_level = np.zeros_like(water_level)
    flow = np.zeros_like(water_level)

    for i in range(water_level.size):
        x, y = i % water_level.shape[0], i // water_level.shape[1]
        local_delta = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        neighbour_amount = 0
        for other_x, other_y in donut_mask(x, y):
            if not (0 <= other_x < water_level.shape[0]) or not (
                0 <= other_y < water_level.shape[1]
            ):
                continue
            neighbour_amount += 1
            local_delta[other_x - x, other_y - y] = (
                water_level[other_x, other_y] - water_level[x, y]
            )
        total_delta = local_delta.sum()
        flow[x, y] = abs(total_delta * FLOW_FRACTION)
        change_in_water_level[x, y] = total_delta * FLOW_FRACTION

    return water_level + change_in_water_level


@jit
def donut_mask(x, y):
    mask = [
        (x - 1, y - 1),
        (x - 1, y),
        (x - 1, y + 1),
        (x, y - 1),
        (x, y + 1),
        (x + 1, y - 1),
        (x + 1, y),
        (x + 1, y + 1),
    ]
    return mask


@jit
def run_simulation(water_level: NDArray, steps):
    history = [water_level]
    for i in range(steps):
        water_level = spread_water_simple(water_level)
        history.append(water_level)
    return history


### PLOTTING
def plot_in_3d(history):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(
        np.arange(history[0].shape[0]), np.arange(history[0].shape[1]), indexing="ij"
    )
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0
    dx = dy = np.ones_like(zpos)
    dz = history[0].ravel()
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz)  # type: ignore
    plt.savefig("videos/image.png")
    plt.show()


def animate_in_3d(history):
    # matplotlib 3d boilerplate
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Construct arrays for the anchor positions of the bars.
    xpos, ypos = np.meshgrid(
        np.arange(history[0].shape[0]), np.arange(history[0].shape[1]), indexing="ij"
    )
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    dx = dy = 1
    zpos = 0
    dz = history[0].ravel()
    mask = dz == 0
    bars = [ax.bar3d(xpos[~mask], ypos[~mask], zpos, dx, dy, dz[~mask], color="deepskyblue")]  # type: ignore
    ax.set(
        xlim3d=[0, history[0].shape[0]], ylim3d=[0, history[0].shape[1]], zlim3d=[0, 10]
    )

    whatever = None

    def update(frame, bars):
        """Update the display by removing the current graph and making a new one. This is the only way
        to do this unfortunately. Very inefficient."""
        bars[0].remove()
        dz = history[frame].ravel()
        mask = dz == 0
        bars[0] = ax.bar3d(xpos[~mask], ypos[~mask], zpos, dx, dy, dz[~mask], color="deepskyblue")  # type: ignore
        return bars

    animation = anim.FuncAnimation(
        fig=fig, func=update, frames=len(history), fargs=(bars,), interval=1 / 60 * 1000
    )
    animation.save("videos/wessels_water.mp4")
    plt.close()


if __name__ == '__main__':
    history = run_simulation(water_level, 120)
    # plot_in_3d(history)
    animate_in_3d(history)
