#from visualizer import Visualizer3D
from robot import PointRobotState
from exploration import ExplorationPlanner, SimplePrimitive
from mapper_py.utils import json_to_grid3d
from mapper_py.data_structures.grid import Grid3D, Cell
import matplotlib.pyplot as plt
from cprint import cprint

grid = Grid3D(0.5, 50, 50, 10, 0.001, 0.999)
grid = json_to_grid3d(grid, 'test_data/simple_box.json')

initial_cell=Cell(25, 23, 6)
robot_state = PointRobotState.from_cell(initial_cell, grid.resolution)

planner_obj = ExplorationPlanner(grid, robot_state)

result = planner_obj.is_feasible(SimplePrimitive(robot_state, Cell(0, -1, 0)))
assert(result == True)

result = planner_obj.is_feasible(SimplePrimitive(robot_state, Cell(1, -1, 0)))
assert(result == True)

result = planner_obj.is_feasible(SimplePrimitive(robot_state, Cell(0, 1, 0)))
assert(result == False)

result = planner_obj.is_feasible(SimplePrimitive(robot_state, Cell(-1, -1, -1)))
assert(result == True)

# If you want visualization during debugging, uncomment the following:
# vis = Visualizer2D()
# vis.initialize_grid(grid)
# vis.draw_robot(robot_state.p)
# plt.show()

cprint.ok("[Task 2.1]: Full Credit.")