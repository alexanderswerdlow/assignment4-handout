import argparse
import code
import numpy as np
import matplotlib.pyplot as plt
from cprint import cprint
from time import time

from visualizer import Visualizer3D
from matplotlib.ticker import MultipleLocator
from robot import PointRobotState
from mapper_py.data_structures.grid import Grid3D, Cell
from mapper_py.data_structures.sensor import Sensor
from mapper_py.data_structures.observer import Observer
from mapper_py.mapper import Mapper


from exploration import ExplorationPlanner, FrontierPlanner, MIPlanner
from mapper_py.utils import json_to_grid3d


class ExploreTest:
    def __init__(self, max_time: int = 200, grid_size = (50, 50, 10), vis_on: bool = True):

        self.grid_res = 1.0
        self.grid_size = grid_size
        self.grid_clamps = (0.001, 0.999)
        self.Tmax = max_time
        self.gt_grid = Grid3D(self.grid_res,
                              self.grid_size[0],
                              self.grid_size[1],
                              self.grid_size[2],
                              self.grid_clamps[0],
                              self.grid_clamps[1])

        # Initialize the map to be explored
        # Initially all probabilities are 0.5
        self.exp_grid = Grid3D(self.grid_res,
                               self.grid_size[0],
                               self.grid_size[1],
                               self.grid_size[2],
                               self.grid_clamps[0],
                               self.grid_clamps[1])

        self.vis_on = vis_on
        if self.vis_on:
            self.vis = Visualizer3D(self.exp_grid)

    def reset(self):
        if self.vis_on:
            self.vis.reset()

        self.gt_grid = Grid3D(self.grid_res,
                              self.grid_size[0],
                              self.grid_size[1],
                              self.grid_size[2],
                              self.grid_clamps[0],
                              self.grid_clamps[1])
        self.exp_grid = Grid3D(self.grid_res,
                               self.grid_size[0],
                               self.grid_size[1],
                               self.grid_size[2],
                               self.grid_clamps[0],
                               self.grid_clamps[1])

    def check_collision(self, state):
        if self.gt_grid.is_cell_free(state.c):
            return False
        else:
            return True

    def run(self,
            scenario='simple_box',
            planner=ExplorationPlanner,
            initial_cell=Cell(10, 10, 5)):
        # Initialize the ground truth map
        json_map_path = f'test_data/{scenario}.json'
        self.gt_grid = json_to_grid3d(self.gt_grid, f'{json_map_path}')

        # Objects developed in Assignment 2 for sensing and mapping
        observer_obj = Observer(self.gt_grid)
        sensor_obj = Sensor(max_range=10.0, max_height=7.0, num_rays=5000)
        mapper_obj = Mapper(self.exp_grid, sensor_obj, observer_obj)

        # Initial robot state
        robot_state = PointRobotState.from_cell(
            initial_cell, self.exp_grid.resolution)

        # Planner for exploration
        planner_obj = planner(self.exp_grid, robot_state)

        entropies = np.zeros(self.Tmax)
        for i in range(self.Tmax):
            mapper_obj.add_obs(planner_obj.state.p)
            planner_obj.update_map(mapper_obj.grid)

            if self.vis_on:
                self.vis.draw_grid(self.exp_grid)
                self.vis.draw_prim_lib(planner_obj.mpl)
                self.vis.draw_robot(planner_obj.state.p)
                # self.vis.save_frame(f'output/{i:06}.png')
                plt.draw()
                plt.pause(0.001)

            planner_obj.take_action()

            # The robot must not collide after taking an action
            if self.check_collision(planner_obj.state):
                cprint.err("The robot has collided. Please check your collision avoidance implementation.",
                           interrupt=False)
                # code.interact(local=locals())
                raise RuntimeError

            entropies[i] = planner_obj.map.map_entropy()


        if self.vis_on:
            self.vis.draw_path()
            # self.vis.fig.colorbar(self.vis.grid_viz, ax=self.vis.ax)
            plt.draw()

        return entropies


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--planner-type", help="type of exploration planner to run", type=str, default="random", choices=['random', 'frontier', 'mi'])
    parser.add_argument("--env", help="environment",
                        type=str, default="simple_box", choices=['simple_box', 'i_love_mr', 'pineapple'])
    parser.add_argument(
        "--max-time", help="maximum duration of exploration", type=int, default=2000)
    parser.add_argument("--vis-on",
                        help="turn visualization on",
                        action='store_true')
    parser.add_argument("--vis-off", help="turn visualization off",
                        dest='vis_on', action='store_false')
    parser.set_defaults(vis_on=True)
    args = parser.parse_args()

    E = np.zeros(args.max_time)

    planner = ExplorationPlanner
    if args.planner_type == 'frontier':
        planner = FrontierPlanner
    if args.planner_type == 'mi':
        planner = MIPlanner

    if args.env == "simple_box":
        initial_cell = Cell(10, 10, 2)
        grid_size = (50, 50, 10)
    if args.env == "i_love_mr":
        initial_cell = Cell(10, 10, 7)
        grid_size = (50, 50, 10)
    if args.env == 'pineapple':
        initial_cell = Cell(0, 0, 5)
        grid_size = (30, 30, 50)

    test_obj = ExploreTest(args.max_time, grid_size, args.vis_on)
    E = test_obj.run(scenario=args.env,
                     planner=planner, initial_cell=initial_cell)

    ent_fig, ent_ax = plt.subplots()
    ent_ax.set_title(f"Entropy Over Time")
    ent_ax.set_xlabel(f"Time step")
    ent_ax.set_ylabel(f"Entropy")
    ent_ax.xaxis.set_major_locator(MultipleLocator(1.0))

    ent_ax.plot(np.arange(args.max_time), E)

    if test_obj.vis_on:
        plt.show()
