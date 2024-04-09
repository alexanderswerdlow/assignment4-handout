import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mapper_py.data_structures.grid import Grid3D, Point

class Visualizer3D:
    def __init__(self, grid: Grid3D) -> None:
        self.fig = plt.figure(figsize=(10, 7.5))
        self.ax = self.fig.add_subplot(projection='3d')

        self.grid = grid
        self.grid_voxels = None

        # set drawing
        self.ax.set_xlim([0.0, grid.resolution * grid.width])
        self.ax.set_ylim([0.0, grid.resolution * grid.depth])
        self.ax.set_zlim([0.0, grid.resolution * grid.height])

        self.ax.xaxis.set_major_locator(MultipleLocator(1.0))
        self.ax.yaxis.set_major_locator(MultipleLocator(1.0))
        self.ax.zaxis.set_major_locator(MultipleLocator(1.0))

        # Labels for clarity on the cell space and point space
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        self.ax.set_title(f"Robot Exploration")

        self.ax.set_aspect("equal")
        # self.ax.set_box_aspect([1,1,1])

        self.rob_history_plot = None
        self.robot_drawing = None
        self.reset()

    def reset(self):
        self.prim_lib_vis = []
        self.prim_lib_objs = LineCollection(
            self.prim_lib_vis, color='red', alpha=0.5, linewidth=2.0)
        
        self.ax.add_collection3d(self.prim_lib_objs)

        self.rob_history = []

        self.draw_robot(Point(0.0, 0.0, 0.0))

        if self.rob_history_plot is not None:
            for l in self.rob_history_plot:
                l.remove()


    def draw_robot(self, pos: Point):
        if self.robot_drawing is not None:
            self.robot_drawing.remove()
        # Make data
        r = 1 * self.grid.resolution 
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = r * np.outer(np.cos(u), np.sin(v)) + pos.x
        y = r * np.outer(np.sin(u), np.sin(v)) + pos.y
        z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + pos.z

        # Plot the surface
        self.robot_drawing = self.ax.plot_surface(x, y, z, color='red', alpha=1.0)

        self.rob_history.append(np.array([pos.x, pos.y, pos.z]))

    def draw_prim_lib(self, prim_lib_gen):
        segs = []
        for l in prim_lib_gen.library:
            segs.append(
                np.array([[l.start.p.x, l.start.p.y, l.start.p.z], [l.end(3).p.x, l.end(3).p.y, l.end(3).p.z]]))
        self.prim_lib_objs.set_segments(segs)
    
    def draw_path(self):
        self.rob_history = np.array(self.rob_history)
        self.rob_history_plot = self.ax.plot(
            self.rob_history[:, 0], 
            self.rob_history[:, 1], 
            self.rob_history[:, 2], 
            'o-', color='r')
    

    def draw_grid(self, grid):
        xs, ys, zs, colors = [], [], [], []
        res = grid.resolution
        ax = self.ax

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
            if self.grid_voxels is not None:
                self.grid_voxels.remove()
            self.grid_voxels = plot_cubes_at(positions, colors=colors, edgecolor="k", sizes=res)
            ax.add_collection3d(self.grid_voxels)

        ax.set_aspect("equal")


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
    # unfortunatley it seems individual alpha values are not supported
    return Poly3DCollection(
        np.concatenate(g), alpha=0.8, facecolors=np.repeat(colors, 6, axis=0), **kwargs
    )

