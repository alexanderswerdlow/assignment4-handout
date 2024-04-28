import numpy as np
from copy import copy
from visualizer import Visualizer3D
from cprint import cprint
import matplotlib.pyplot as plt

from mapper_py.utils import json_to_grid3d
from robot import PointRobotState
from mapper_py.data_structures.grid import Grid3D, Cell
from mapper_py.data_structures.sensor import Sensor
from mapper_py.data_structures.observer import Observer
from mapper_py.mapper import Mapper

from exploration import MIPlanner

def compute_all_mi(mi_grid: np.ndarray, prob_grid: Grid3D, planner_obj: MIPlanner):
    for row in range(prob_grid.depth):
        for col in range(prob_grid.width):
            for slc in range(prob_grid.height):
                cell = Cell(row, col, slc)
                pos = prob_grid.cell_to_point(cell)
                mi_grid[row][col][slc], _ = planner_obj.compute_mi(pos)

def score_mi_reward():
    gt_grid = Grid3D(0.5, 50, 50, 10, 0.001, 0.999)
    gt_grid = json_to_grid3d(gt_grid, 'test_data/simple_box.json')

    exp_grid = Grid3D(0.5, 50, 50, 10, 0.001, 0.999)
    grid_min = exp_grid.cell_to_point_row_col_layer(0, 0, 0)
    grid_max = exp_grid.cell_to_point_row_col_layer(exp_grid.depth, exp_grid.width, exp_grid.height)
    mi_grid = -10000 * np.ones((exp_grid.depth, exp_grid.width, exp_grid.height))

    mi_fig, mi_ax = plt.subplots()

    observer_obj = Observer(gt_grid)
    sensor_obj = Sensor(max_range=2.0, max_height=2.0, num_rays=50)
    mapper_obj = Mapper(exp_grid, sensor_obj, observer_obj)

    planner_obj = MIPlanner(exp_grid, PointRobotState.from_cell(Cell(10, 10, 1), exp_grid.resolution))
    mapper_obj.add_obs(planner_obj.state.p)
    mapper_obj.add_obs(planner_obj.state.p)
    mapper_obj.add_obs(planner_obj.state.p)
    planner_obj.update_map(mapper_obj.grid)

    score1 = 0.0
    compute_all_mi(mi_grid, planner_obj.map, planner_obj)
    soln1 = np.load('test_data/mi_grid1.npz')['arr_0']
    overlap = np.mean(soln1 == mi_grid)
    if overlap >= 0.95:
        score1 = 1.0
    else:
        score1 = copy(overlap)

    plot_obj = mi_ax.imshow(mi_grid[:, :, 5], origin='lower', extent=(grid_min.x, grid_max.x,
                                                        grid_min.y, grid_max.y,
                                                        grid_min.z, grid_max.z) )
    hl = mi_fig.colorbar(plot_obj, ax=mi_ax)
    plt.draw()
    plt.pause(0.5)

    planner_obj.update_state(PointRobotState.from_cell(Cell(25, 10, 1), exp_grid.resolution))
    mapper_obj.add_obs(planner_obj.state.p)
    mapper_obj.add_obs(planner_obj.state.p)
    mapper_obj.add_obs(planner_obj.state.p)
    planner_obj.update_map(mapper_obj.grid)

    score2 = 0.0
    compute_all_mi(mi_grid, planner_obj.map, planner_obj)
    soln2 = np.load('test_data/mi_grid2.npz')['arr_0']
    overlap = np.mean(soln2 == mi_grid)
    if overlap >= 0.95:
        score2 = 1.0
    else:
        score2 = copy(overlap)
    plot_obj.set_data(mi_grid[:, :, 5])
    plt.draw()
    plt.pause(2.0)

    return score1, score2

if __name__ == "__main__":
    score1, score2 = score_mi_reward()
    final_score = (score1 + score2) / 2.0
    if final_score == 1.0:
        cprint.ok("[Task 4.1]: Full Credit.")
    else:
        cprint.info(f"[Task 4.1] Score: {final_score}")

    plt.show()
