"""Observer class for 16-761: Mobile Robot Algorithms Laboratory

Author(s): Kshitij Goel, Andrew Jong, Rebecca Martin, Wennie Tabib
"""

import numpy as np

from .grid import Point

class Observer:
    def __init__(self, gt_grid):
        self.grid = gt_grid

    def observe_along_ray(self, ray, max_range, max_height):
        max_dist = min((max_height / np.abs(ray.d.z)), (max_range / np.sqrt(ray.d.x ** 2 + ray.d.y ** 2))) if ray.d.z != 0 else (max_range / np.sqrt(ray.d.x ** 2 + ray.d.y ** 2))
        success, cells = self.grid.traverse(ray.o, ray.point_at_dist(max_dist))

        if success:
            for c in cells:
                found_occ = False
                if self.grid.is_cell_occupied(c):
                    found_occ = True
                    break

            if found_occ:
                return self.grid.cell_to_point(c) + Point(self.grid.resolution / 2, self.grid.resolution / 2, self.grid.resolution / 2)
            else:
                return ray.point_at_dist(max_dist)
        else:
            return None
