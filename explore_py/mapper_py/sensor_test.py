import numpy as np
from cprint import cprint

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from data_structures.grid import Grid3D, Point
from data_structures.sensor import Sensor

from utils import visualize3d

def plot_rays(ax, pos, rays, max_range, max_height):
    for r in rays:
        max_dist = min((max_height / np.abs(r.d.z)), (max_range / np.sqrt(r.d.x ** 2 + r.d.y ** 2))) if r.d.z != 0 else (max_range / np.sqrt(r.d.x ** 2 + r.d.y ** 2))
        ep = r.point_at_dist(max_dist)
        ax.plot([pos.x, ep.x], [pos.y, ep.y], [pos.z, ep.z], color='lime')

# Scene
scene_ax = plt.subplot(projection='3d')

# Initialize an empty grid along with visualization updates
grid = Grid3D(0.1, 40, 40, 40, 0.001, 0.999)
scene_ax.grid(which='major', axis='both', linestyle='-')
scene_ax.grid(which='minor', axis='both', linestyle='-')

grid_hl = visualize3d(grid, ax=scene_ax)

sensor = Sensor()

pos = Point(1.23, 3.2, 2.4)
rays = sensor.rays(pos)

rays_np = np.zeros((len(rays), 6))
for i, r in enumerate(rays):
    rays_np[i, :] = r.to_numpy()

# np.savez('test_data/3d_sensor_test.npz', rays_np=rays_np)
rays_np_correct = np.load('test_data/3d_sensor_test.npz')['rays_np']
if (np.abs(rays_np_correct - rays_np) < 1e-6).all():
    cprint.info('sensor_test successful.')
else:
    cprint.err('sensor_test failed.', interrupt=False)

plot_rays(scene_ax, pos, rays, sensor.max_range, sensor.max_height)

plt.show()