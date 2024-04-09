from cprint import cprint

from mapper_py.utils import json_to_grid3d
from robot import PointRobotState
from mapper_py.data_structures.grid import Grid3D, Cell
from mapper_py.data_structures.sensor import Sensor
from mapper_py.data_structures.observer import Observer
from mapper_py.mapper import Mapper

def score_entropy():
    gt_grid = Grid3D(0.5, 50, 50, 10, 0.001, 0.999)
    gt_grid = json_to_grid3d(gt_grid, 'test_data/simple_box.json')
    ent1 = gt_grid.map_entropy()
    print(ent1)
    assert(abs(ent1 - 285.19) < 1e-2)

    exp_grid = Grid3D(0.5, 50, 50, 10, 0.001, 0.999)
    ent2 = exp_grid.map_entropy()
    print(ent2)
    assert(abs(ent2 - 25000) < 1e-2)

    observer_obj = Observer(gt_grid)
    sensor_obj = Sensor(max_range=2.0, max_height=2.0, num_rays=50)
    mapper_obj = Mapper(exp_grid, sensor_obj, observer_obj)

    robot_state = PointRobotState.from_cell(Cell(10, 10, 1), exp_grid.resolution)
    mapper_obj.add_obs(robot_state.p)
    ent3 = mapper_obj.grid.map_entropy()
    print(ent3)
    assert(ent3 < ent2)
    
    mapper_obj.add_obs(robot_state.p)
    ent4 = mapper_obj.grid.map_entropy()
    print(ent4)
    assert(ent4 < ent2 and ent4 < ent3)

    cprint.ok("[Task 0.2]: Full Credit.")

if __name__ == "__main__":
    score_entropy()