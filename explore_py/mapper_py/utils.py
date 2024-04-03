"""Mapping visualization utility functions for 16-761: Mobile Robot Algorithms Laboratory

Author(s): Kshitij Goel, Andrew Jong, Rebecca Martin, Wennie Tabib
"""

from os.path import exists
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.ticker import MultipleLocator


def json_to_grid3d(grid, filepath):
    """
    Convert from the JSON voxel format specified from https://nimadez.github.io/voxel-builder/

    Args:
        grid : (Grid3D) 3D grid that will be changed during this function
        filepath : (str) Path to the json file used to generate cell values

    Returns:
        grid : (Grid3D) The updated grid object
    """
    data = json.load(open(filepath, "r"))
    voxel_str = data["data"]["voxels"]
    cell_strs = voxel_str.split(";")
    for row in range(grid.depth):
        for col in range(grid.width):
            for layer in range(grid.height):
                grid.set_row_col_layer(row, col, layer, grid.min_clamp)

    for cell_str in cell_strs:
        if cell_str:
            cell = cell_str.split(",")
            col = int(cell[0])
            layer = int(cell[1])
            row = int(cell[2])
            if (col >= 0 and row >= 0 and layer >= 0) and (col < grid.width and row < grid.depth and layer < grid.height):
                grid.set_row_col_layer(row, col, layer, grid.max_clamp)

    return grid


def visualize3d(grid, ax, val_fn=lambda x, v: x.probability(v)):
    """Visualize the grid on canvas.
    Each voxel alpha is set according to the probability value of the grid at that point.

    Args:
        grid : (Grid3D) 3D grid to visualize
        ax :

    Returns:
        plot : Plotted matplotlib object
    """

    xs, ys, zs, colors = [], [], [], []
    res = grid.resolution

    cmap = matplotlib.cm.get_cmap("rainbow")

    # convert grid to image and display
    for row in range(grid.depth):
        for col in range(grid.width):
            for layer in range(grid.height):
                logprob = grid.get_row_col_layer(row, col, layer)
                if logprob > grid.occ_thres:
                    xs.append(col * res)
                    ys.append(row * res)
                    zs.append(layer * res)
                    alpha = grid.probability(logprob)
                    rgba = cmap(layer / grid.height)[:3] + (alpha,)
                    colors.append(rgba)

    positions = np.column_stack([xs, ys, zs])
    colors = np.array(colors)
    # edgecolors = colors.copy()
    # edgecolors = np.clip(2 * edgecolors - 0.5, 0, 1)
    if len(positions) > 0:
        pc = plot_cubes_at(positions, colors=colors, edgecolor="k", sizes=res)
        ax.add_collection3d(pc)

    # plot the boundaries of the grid
    corners = np.array(
        [
            [0, 0, 0],
            [res * grid.width, 0, 0],
            [res * grid.width, res * grid.depth, 0],
            [0, res * grid.depth, 0],
            [0, 0, res * grid.height],
            [res * grid.width, 0, res * grid.height],
            [res * grid.width, res * grid.depth, res * grid.height],
            [0, res * grid.depth, res * grid.height],
        ]
    )
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]
    for edge in edges:
        ax.plot3D(*zip(*corners[edge]), color="k")

    # set drawing
    ax.set_xlim([0.0, res * grid.width])
    ax.set_ylim([0.0, res * grid.depth])
    ax.set_zlim([0.0, res * grid.height])

    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.zaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(grid.resolution))
    ax.yaxis.set_minor_locator(MultipleLocator(grid.resolution))
    ax.zaxis.set_minor_locator(MultipleLocator(grid.resolution))

    # Labels for clarity on the cell space and point space
    ax.set_xlabel("Cell Space: Cols, Point Space: X (meters)")
    ax.set_ylabel("Cell Space: Rows, Point Space: Y (meters)")
    ax.set_zlabel("Cell Space: Layers, Point Space: Z (meters)")

    ax.set_title(f"Grid3D, Cell Resolution={grid.resolution}m^3")

    ax.set_aspect("equal")
    # ax.set_box_aspect([1,1,1])

    return ax


# https://stackoverflow.com/a/42611693
def cuboid_data(origin, size=(1, 1, 1)):
    X = [
        [[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
        [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
        [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
        [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
        [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
        [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]],
    ]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:, :, i] *= size[i]
    X += np.array(origin)
    return X


def plot_cubes_at(positions, sizes=None, colors=None, **kwargs):
    if not isinstance(colors, (list, np.ndarray)):
        colors = ["C0"] * len(positions)
    if isinstance(sizes, (float, int)):
        sizes = [(sizes, sizes, sizes)] * len(positions)
    if not isinstance(sizes, (list, np.ndarray)):
        sizes = [(1, 1, 1)] * len(positions)
    g = []
    for p, s in zip(positions, sizes):
        g.append(cuboid_data(p, size=s))
    return Poly3DCollection(
        np.concatenate(g), facecolors=np.repeat(colors, 6, axis=0), **kwargs
    )


# Helper function to plot the traced cells on the grid
def plot_traced_cells(traced_cells, ax, resolution=1.0, c="red", alpha=0.1):
    xs, ys, zs = [], [], []
    for t in traced_cells:
        xs.append(resolution * t.col)
        ys.append(resolution * t.row)
        zs.append(resolution * t.layer)

    positions = np.column_stack([xs, ys, zs])
    color = matplotlib.colors.to_rgba(c)
    colors = np.array([color for _ in range(len(xs))])
    colors[:, 3] = alpha
    # make edge colors brighter but clamp at 1
    edgecolors = colors.copy()
    edgecolors = np.clip(edgecolors + 0.5, 0, 1)
    pc = plot_cubes_at(positions, colors=colors, edgecolor=edgecolors, sizes=resolution)
    ax.add_collection3d(pc)
