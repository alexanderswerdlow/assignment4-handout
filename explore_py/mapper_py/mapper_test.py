import numpy as np
from cprint import cprint

from data_structures.grid import Grid3D, Point
from mapper import Mapper
from data_structures.sensor import Sensor
from data_structures.observer import Observer

from utils import visualize3d, json_to_grid3d

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from skspatial.objects import Sphere

def test_qualitative(positions, map_name='simple_box'):
    scene_fig = plt.figure()
    scene_ax = []
    scene_ax.append(scene_fig.add_subplot(2, 1, 1, projection='3d')) 
    scene_ax.append(scene_fig.add_subplot(2, 1, 2, projection='3d'))

    json_map_path = f'test_data/{map_name}.json'

    gt_grid = Grid3D(1.0, 22, 22, 52, 0.001, 0.999) if map_name == 'pineapple' else Grid3D(0.5, 50, 50, 10, 0.001, 0.999)
    gt_grid = json_to_grid3d(gt_grid, json_map_path)

    observer_obj = Observer(gt_grid)

    grid = Grid3D(1.0, 22, 22, 52, 0.001, 0.999) if map_name == 'pineapple' else Grid3D(0.5, 50, 50, 10, 0.001, 0.999)
    sensor_obj = Sensor(max_range=5.0, max_height=10.0, num_rays=210)

    mapper_obj = Mapper(grid, sensor_obj, observer_obj)

    scene_ax[0].xaxis.set_major_locator(MultipleLocator(1.0))
    scene_ax[0].yaxis.set_major_locator(MultipleLocator(1.0))
    scene_ax[0].xaxis.set_minor_locator(MultipleLocator(gt_grid.resolution))
    scene_ax[0].yaxis.set_minor_locator(MultipleLocator(gt_grid.resolution))
    scene_ax[0].grid(which='major', axis='both', linestyle='-')
    scene_ax[0].grid(which='minor', axis='both', linestyle='-')
    scene_ax[0].set_xlabel('Cell Space: Cols, Point Space: X (meters)')
    scene_ax[0].set_ylabel('Cell Space: Rows, Point Space: Y (meters)')
    # scene_ax[0].set_aspect('equal')
    scene_ax[0].set_title('Occupancy Grid Map')

    scene_ax[1].xaxis.set_major_locator(MultipleLocator(1.0))
    scene_ax[1].yaxis.set_major_locator(MultipleLocator(1.0))
    scene_ax[1].xaxis.set_minor_locator(MultipleLocator(gt_grid.resolution))
    scene_ax[1].yaxis.set_minor_locator(MultipleLocator(gt_grid.resolution))
    scene_ax[1].grid(which='major', axis='both', linestyle='-')
    scene_ax[1].grid(which='minor', axis='both', linestyle='-')
    scene_ax[1].set_title('Range sensor getting data in real world')
    scene_ax[1].set_xlabel('Cell Space: Cols, Point Space: X (meters)')
    scene_ax[1].set_ylabel('Cell Space: Rows, Point Space: Y (meters)')
    # scene_ax[1].set_aspect('equal')
    visualize3d(gt_grid, ax=scene_ax[1])

    rays_vis = []
    rays_collection = Line3DCollection(rays_vis, color='lime', alpha=0.5)
    scene_ax[1].add_collection(rays_collection)
    for pos in positions:
        endpoints = mapper_obj.add_obs(pos)
        segs = []
        colors = []
        for i in range(len(endpoints)):
            segs.append(
                np.array([[pos.x, pos.y, pos.z], [endpoints[i].x, endpoints[i].y, endpoints[i].z]]))

        rays_collection.set_segments(segs)

        sphere = Sphere([pos.x, pos.y, pos.z], grid.resolution / 2)
        sphere.plot_3d(scene_ax[1], alpha=0.5, color='r')
    vis_obj = visualize3d(grid, ax=scene_ax[0])

    plt.draw()
    plt.show()


def test_quantitative(positions, map_name='simple_box'):
    json_map_path = f'test_data/{map_name}.json'
    gt_grid = Grid3D(1.0, 22, 22, 52, 0.001, 0.999) if map_name == 'pineapple' else Grid3D(0.5, 50, 50, 10, 0.001, 0.999)
    gt_grid = json_to_grid3d(gt_grid, json_map_path)

    observer_obj = Observer(gt_grid)

    grid = Grid3D(1.0, 22, 22, 52, 0.001, 0.999) if map_name == 'pineapple' else Grid3D(0.5, 50, 50, 10, 0.001, 0.999)
    sensor_obj = Sensor(max_range=2.0, max_height=2.0, num_rays=50)

    mapper_obj = Mapper(grid, sensor_obj, observer_obj)


    interval = 50
    scores_arr = []
    for i, pos in enumerate(positions):
        if i % interval == 0 or i == len(positions) - 1:
            mapper_obj.add_obs(pos)
            npz_data_file = f'test_data/{map_name}_mapper_test_{str(i)}.npz'
            grid_numpy = grid.to_numpy()
            # np.savez(npz_data_file, grid_numpy=grid_numpy)
            grid_numpy_correct = np.load(npz_data_file)['grid_numpy']
            corrects = np.abs(grid_numpy_correct - grid_numpy) < 1e-3
            avg = np.sum(corrects) / grid.width / grid.height / grid.depth
            if avg >= 0.95:
                scores_arr.append(1.0)
            else:
                scores_arr.append(avg)
    
    return np.sum(np.array(scores_arr)) / len(scores_arr)


if __name__ == "__main__":
    simple_obs_positions = [Point(float(x), float(y), float(z)) 
                            for x in range(1, 25, 2) 
                            for y in range(1, 25, 2) 
                            for z in range(1, 5)]


    cprint.info('Running qualitative test on simple_box map')
    test_qualitative(simple_obs_positions, 'simple_box')
    cprint.info('Running quantitative test on simple_box map')
    score = test_quantitative(simple_obs_positions, 'simple_box')
    cprint.ok('Quantitative test score for simple_box map %f' % (score))

    i_love_mr_positions = [Point(float(x), float(y), float(z)) 
                            for x in range(1, 25, 2) 
                            for y in range(1, 25, 2) 
                            for z in range(1, 5)]

    cprint.info('Running qualitative test on i_love_mr map')
    # test_qualitative(i_love_mr_positions, 'i_love_mr')
    cprint.info('Running quantitative test on i_love_mr map')
    score = test_quantitative(i_love_mr_positions, 'i_love_mr')
    cprint.ok('Quantitative test score for i_love_mr map %f' % (score))

    plt.show()