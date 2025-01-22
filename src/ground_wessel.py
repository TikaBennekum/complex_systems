import numpy as np
from numpy.typing import NDArray
from numpy.dtypes import StringDType
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from numba import jit
from random import random

import matplotlib
import imageio_ffmpeg

matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


### CONSTANTS
L = 20
FLOW_FRACTION = 1/8.0  # maximum fraction of water which can flow out of cell
FRAMES = 100


### LAYERS
water_level = np.zeros([L, L])
water_level[L // 2, L // 2] = 10.0
ground_level = np.ones_like(water_level)
for i in range(ground_level.shape[0]):
    ground_level[i, :] *= ground_level.shape[0] - i


### FUNCTIONS
@jit
def spread_water(
    water_level: NDArray, ground_level: NDArray
) -> tuple[NDArray, NDArray]:
    change_in_water_level = np.zeros_like(water_level)
    change_in_ground_level = np.zeros_like(water_level)
    # use the absolute value for flow to indicate the flux of the water, 
    # even when level stays constant
    flow = np.zeros_like(water_level)
    total_height = water_level + ground_level
    # print(water_level)
    # print(ground_level)
    # print()

    for i in range(water_level.size):
        x, y = i % water_level.shape[0], i // water_level.shape[1]
        # height_delta = np.zeros((3, 3))
        # neighbour_amount = 0
        for other_x, other_y in donut_mask(x, y):
            if not (0 <= other_x < water_level.shape[0]) or not (
                0 <= other_y < water_level.shape[1]
            ):
                continue
            # neighbour_amount += 1
            # height_delta[other_x - x + 1, other_y - y + 1] = (
            #     water_level[other_x, other_y] - water_level[x, y]
            #     # total_height[other_x, other_y] - total_height[x, y]
            # )
            change_in_water_level[other_x, other_y] += determine_water_flow(water_level[x, y], ground_level[x, y], water_level[other_x, other_y], ground_level[other_x, other_y], constant=FLOW_FRACTION)
        # print("For "+str(x)+", "+str(y)+": height_delta = ")
        # print(height_delta)
        # print()
        # total_delta = height_delta.sum()
        # use the absolute value for flow to indicate the flux of the water, 
        # even when level stays constant
        # flow[x, y] = abs(total_delta * FLOW_FRACTION)
        # change_in_water_level[x, y] = total_delta * FLOW_FRACTION
    # print(change_in_water_level)

    return water_level + change_in_water_level, ground_level + change_in_ground_level


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
def cap(value, at):
    if value > at:
        return at 
    return value 


@jit 
def abs_cap(value, at):
    if abs(value) > abs(at):
        return at 
    return value 


@jit 
def lower_cap(value, at):
    if value < at:
        return at 
    return value 


@jit
def run_simulation(water_level: NDArray, ground_level: NDArray, steps):
    history = [(water_level, ground_level)]
    for i in range(steps):
        # run functions
        water_level, ground_level = spread_water(water_level, ground_level)
        # save results
        history.append((water_level, ground_level))
    return history


@jit 
def determine_water_flow(water_self: float, ground_self: float, water_other: float, ground_other: float, constant: float) -> float:
    """Outflow if negative, inflow when positive."""
    return cap(
        lower_cap(
            (water_self + ground_self) - (water_other + ground_other), -water_other), 
            water_self
        ) * constant



@jit 
def test_water_flow(iterations: int):
    for _ in range(iterations):
        scaling_factor = 10.0
        water_self   = random() * scaling_factor
        water_other  = random() * scaling_factor
        ground_self  = random() * scaling_factor
        ground_other = random() * scaling_factor

        water_before = water_self + water_other
        ground_before = ground_self + ground_other

        self_to_other = determine_water_flow(water_self, ground_self, water_other, ground_other, 1.0)
        other_to_self = determine_water_flow(water_other, ground_other, water_self, ground_self, 1.0)

        def debug_print():
            print("water_self:    ", water_self)
            print("ground_self:   ", ground_self)
            print("height_self:   ", water_self + ground_self)
            print("water_other:   ", water_other)
            print("ground_other:  ", ground_other)
            print("height_other:  ", water_other + ground_other)
            print("water_before:  ", water_before)
            print("ground_before: ", ground_before)
            print("self_to_other: ", self_to_other)
            print("other_to_self: ", other_to_self)

        assert self_to_other == - other_to_self, debug_print()
        assert not self_to_other > water_self, debug_print()
        assert not other_to_self > water_other, debug_print()
    print("Water flow test passed.")



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
    # print(history[-1][0])
    # matplotlib 3d boilerplate
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    shape = history[0][0].shape

    # Construct arrays for the anchor positions of the bars.
    base_x, base_y = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), indexing="ij"
    )
    base_x = base_x.ravel()
    base_y = base_y.ravel()
    base_dx = base_dy = 1
    base = np.zeros_like(base_x)

    # ground
    ground_x = base_x
    ground_y = base_y
    ground_z = np.zeros_like(base)
    # ground_dx = 1
    # ground_dy = 1
    ground_dz = history[0][1].ravel()
    ground_color = np.full(len(base), "peru", dtype=StringDType())

    # water
    mask = history[0][0].ravel() < 1e-6
    water_x = base_x[~mask]
    water_y = base_y[~mask]
    water_z = history[0][1].ravel()[~mask]
    # water_dx = 1
    # water_dy = 1
    water_dz = history[0][0].ravel()[~mask]
    water_color = np.full(len(base), "deepskyblue", dtype=StringDType())[~mask]

    bars = [
        ax.bar3d(  # type: ignore
            np.append(ground_x, water_x),
            np.append(ground_y, water_y),
            np.append(ground_z, water_z),
            base_dx,
            base_dy,
            np.append(ground_dz, water_dz),
            color=np.append(ground_color, water_color),
            zsort="max",
        ),
    ]
    ax.set(xlim3d=[0, shape[0]], ylim3d=[0, shape[1]], zlim3d=[0, 10])

    def update(frame, bars):
        """
        Update the display by removing the current graph and making a new one. 
        This is the only way to do this unfortunately. Very inefficient.
        """
        bars[0].remove()

        # ground
        ground_x = base_x
        ground_y = base_y
        ground_z = np.zeros_like(base)
        # ground_dx = 1
        # ground_dy = 1
        ground_dz = history[frame][1].ravel()
        ground_color = np.full(len(base), "peru", dtype=StringDType())

        # water
        mask = history[frame][0].ravel() < 1e-6
        water_x = base_x[~mask]
        water_y = base_y[~mask]
        water_z = history[frame][1].ravel()[~mask]
        # water_dx = 1
        # water_dy = 1
        water_dz = history[frame][0].ravel()[~mask]
        water_color = np.full(len(base), "deepskyblue", dtype=StringDType())[~mask]

        bars[0] = ax.bar3d(  # type: ignore
            np.append(ground_x, water_x),
            np.append(ground_y, water_y),
            np.append(ground_z, water_z),
            base_dx,
            base_dy,
            np.append(ground_dz, water_dz),
            color=np.append(ground_color, water_color),
            zsort="max",
        )
        return bars

    animation = anim.FuncAnimation(
        fig=fig, func=update, frames=len(history), fargs=(bars,), interval=100#1 / 60 * 1000
    )
    animation.save("videos/wessels_ground.mp4")
    plt.close()


if __name__ == "__main__":
    # history = run_simple(water_level, 120)
    test_water_flow(100000)
    history = run_simulation(water_level, ground_level, FRAMES)
    animate_in_3d(history)
