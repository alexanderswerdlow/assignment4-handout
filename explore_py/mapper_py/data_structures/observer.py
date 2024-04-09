"""Observer class for 16-761: Mobile Robot Algorithms Laboratory

Author(s): Kshitij Goel, Andrew Jong, Rebecca Martin, Wennie Tabib
"""

import numpy as np

from typing import Tuple, List
from .grid import Point, Cell

class Observer:
    def __init__(self, gt_grid):
        self.grid = gt_grid

    def observe_along_ray(self, ray, max_range, max_height) -> Tuple[Point, List[Cell], bool]:
        """_summary_

        Args:
            ray (_type_): _description_
            max_range (_type_): _description_
            max_height (_type_): _description_

        Returns:
            Tuple[Point, List[Cell], bool]: end point, list of traced cells, true or false if last cell in traced cells is occupied
        """
        max_dist = min((max_height / np.abs(ray.d.z)), (max_range / np.sqrt(ray.d.x ** 2 + ray.d.y ** 2))) if ray.d.z != 0 else (max_range / np.sqrt(ray.d.x ** 2 + ray.d.y ** 2))
        success, cells = self.grid.traverse(ray.o, ray.point_at_dist(max_dist))

        traced_cells = []
        is_last_cell_occupied = False

        if success:
            for c in cells:
                traced_cells.append(c)
                if (self.grid.is_cell_occupied(c)):
                    is_last_cell_occupied = True
                    # must use this to preserve the ray angle
                    cutoff_dist = abs(self.grid.cell_to_point(c) + Point(self.grid.resolution / 2, self.grid.resolution / 2, self.grid.resolution / 2) - ray.o)
                    end_point = ray.point_at_dist(cutoff_dist)
                    break
            else:
                end_point = ray.point_at_dist(max_dist)
            return end_point, traced_cells, is_last_cell_occupied
        else:
            return None, [], False
