import argparse
from cprint import cprint
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from data_structures.grid import Grid3D, Point
from utils import json_to_grid3d, visualize3d, plot_traced_cells

def test_data_structure(map_name, grid_visible=True):
    # Path to the png file corresponding to the environment
    # Black regions are occupied, white regions are free
    # In this test, a grid map should be created for this map
    png_map_path = f'test_data/{map_name}.json'

    # Grid map at resolution 0.1 of size 60 cells x 80 cells
    # Minimum probability (i.e. highest confidence about free space) is 0.001
    width = 22 if map_name == 'pineapple' else 50
    height = 50 if map_name == 'pineapple' else 10
    # Maximum probability (i.e. highest confidence about occupied space) is 0.999
    grid = Grid3D(1.0, width, width, height, 0.001, 0.999)
    # grid = Grid3D(0.5, 100, 100, 20, 0.001, 0.999)

    # Update the grid using the png image
    grid = json_to_grid3d(grid, png_map_path)

    # Get a numpy array corresponding to the grid
    grid_numpy = grid.to_numpy()
    # np.savez(f'test_data/{map_name}.npz', grid_numpy=grid_numpy)  # this line is to generate the answers. leave this commented out
    # Load the correct answer
    b = np.load(f'test_data/{map_name}.npz')
    grid_numpy_correct = b['grid_numpy']

    # # Check if all the values are close
    # If you get "test_data_structure failed", check your grid implementation
    if (np.abs(grid_numpy_correct - grid_numpy) < 1e-6).all():
        cprint.info('test_data_structure successful.')
    else:
        cprint.err('test_data_structure failed.', interrupt=False)

    # Visualize for extra clarity
    grid_fig = plt.figure(figsize=(10, 7.5))
    grid_ax = grid_fig.add_subplot(projection='3d')
    visualize3d(grid, ax=grid_ax)

    plt.show()

def test_traversal(grid, grid_ax, start=Point(1.2, 1.2, 1.2), end=Point(2.2, 1.5, 2.2), test_file='traced_cells_1',
                   c='navy', grid_visible=True):

    # Visualize the empty grid
    visualize3d(grid, ax=grid_ax)

    # Plot the ray from start to end
    grid_ax.plot([start.x, end.x], [start.y, end.y], [start.z, end.z], color=c)

    # Get the traced cells
    success, traced_cells = grid.traverse(start, end)

    # First, the traverse function should succeed
    # If your code fails this check, check your traverse function implementation
    if success:
        cprint.info(f"traverse function succeeded, number of traced cells: {len(traced_cells)}")
    else:
        cprint.err("traverse function failed.", interrupt=True)

    # Then, check if the traced cells are correct
    # If your code fails this check, check your traverse function implementation
    traced_cells_np = np.array([a.to_numpy() for a in traced_cells])
    # np.savez(f'test_data/{test_file}.npz', traced_cells=traced_cells_np)  # this line is to generate the answers. leave this commented out
    traced_cells_np_correct = np.load(f'test_data/{test_file}.npz')['traced_cells']

    if (np.abs(traced_cells_np_correct - traced_cells_np) < 1e-6).all():
        cprint.info(f"test_traversal {test_file} successful.")
    else:
        cprint.err(f"test_traversal {test_file} failed.", interrupt=False)

    plot_traced_cells(traced_cells, grid_ax, c=c, resolution=grid.resolution)

def test_traversals(grid):
    trav_fig = plt.figure(figsize=(10, 7.5))
    trav_ax = trav_fig.add_subplot(projection='3d')

    # # Test slopped rays
    # these should be the same. drawing start to end should be the same as drawing end to start
    test_traversal(grid, trav_ax, start=Point(1.0, 1.0, 1.0), end=Point(5.0, 5.0, 3.0), c="red", test_file=f'traced_cells_01_{grid.resolution}m')
    test_traversal(grid, trav_ax, start=Point(1.0, 1.0, 5.0), end=Point(5.0, 5.0, 7.0), c="orange", test_file=f'traced_cells_02_{grid.resolution}m')
    # criss cross
    test_traversal(grid, trav_ax, start=Point(11.0, 5.0, 3.0), end=Point(6.0, 1.0, 1.0), c="yellow", test_file=f'traced_cells_03_{grid.resolution}m')
    test_traversal(grid, trav_ax, start=Point(6.0, 5.0, 5.0), end=Point(11.0, 1.0, 7.0), c="green", test_file=f'traced_cells_04_{grid.resolution}m')

    test_traversal(grid, trav_ax, start=Point(13.0, 5.0, 1.0), end=Point(19.0, 1.0, 9.0), c="springgreen", test_file=f'traced_cells_05_{grid.resolution}m')

    # # Test edge rays (horizontal and vertical)
    # vertical z
    test_traversal(grid, trav_ax, start=Point(1.0, 10.0, 1.0), end=Point(1.0, 10.0, 7.0), c='cyan', test_file=f'traced_cells_06_{grid.resolution}m')
    # horizontal x
    test_traversal(grid, trav_ax, start=Point(7.0, 10.0, 5.0), end=Point(12.0, 10.0, 5.0), c='blue', test_file=f'traced_cells_07_{grid.resolution}m')
    # horizontal y
    test_traversal(grid, trav_ax, start=Point(18.0, 8.0, 5.0), end=Point(18.0, 12.0, 5.0), c='blueviolet', test_file=f'traced_cells_08_{grid.resolution}m')

    # # Test finer rays (horizontal and vertical)
    # vertical z
    test_traversal(grid, trav_ax, start=Point(0.54, 15.0, 1.0), end=Point(1.89, 15.5, 7.0), c='magenta', test_file=f'traced_cells_09_{grid.resolution}m')
    # horizontal x
    test_traversal(grid, trav_ax, start=Point(7.0, 14.5, 5.0), end=Point(12.0, 15.0, 5.0), c='orchid', test_file=f'traced_cells_10_{grid.resolution}m')

    # # Test rays going outside of the map bounds
    test_traversal(grid, trav_ax, start=Point(15.0, 14.5, 3.0), end=Point(25.0, 30.0, 11.0), c='hotpink', test_file=f'traced_cells_11_{grid.resolution}m')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default='i_love_mr')

    args = parser.parse_args()

    test_data_structure(args.map)

    # Initialize an empty grid
    # these should look roughly the same
    grid_unit_res = Grid3D(1.0, 20, 20, 10, 0.001, 0.999)
    test_traversals(grid_unit_res)

    grid_half_res = Grid3D(0.5, 40, 40, 20, 0.001, 0.999)
    test_traversals(grid_half_res)

    plt.show()
