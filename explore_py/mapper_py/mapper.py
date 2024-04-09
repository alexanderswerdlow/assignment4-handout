"""Mapper class for 16-761: Mobile Robot Algorithms Laboratory

Author(s): Kshitij Goel, Andrew Jong, Rebecca Martin, Wennie Tabib
"""

class Mapper:
    """Occupancy grid mapper that uses the sensor to update the grid.

    Attributes:
        grid: (data_structures.grid3d.Grid3D) The grid being updated by this mapper
        sensor: (data_structures.sensor.Sensor) The sensor model being used for mapping
        observer: (data_structures.observer.Observer) The observer is looking at the real world and providing
            distance measurements (i.e., the first obstacle that is hit by a ray).
    """

    def __init__(self, grid, sensor, observer, prob_hit=0.99, prob_miss=0.33):
        self.grid = grid
        self.sensor = sensor
        self.observer = observer

        self.log_odds_hit = self.grid.logodds(prob_hit)
        self.log_odds_miss = self.grid.logodds(prob_miss)

    def update_logodds(self, cell, update):
        """Update the logodds value in the input cell.

        Args:
            cell: (Cell) Cell in self.grid for which the update has to be applied.
            update: (float) Logodds update value. This needs to be added to the existing value for the cell.
        """
        # TODO: Assignment 2, Problem 1.3
        current_val = self.grid.get_cell(cell)
        val = max(self.grid.min_clamp, min(
            self.grid.max_clamp, current_val + update))
        self.grid.set_cell(cell, val)

    def update_miss(self, cell):
        """Update the logodds value for the cell where the ray passed through ("miss" case)."""
        # TODO: Assignment 2, Problem 1.3
        raise NotImplementedError

    def update_hit(self, cell):
        """Update the logodds value for the cell where the ray terminated ("hit" case)."""
        # TODO: Assignment 2, Problem 1.3
        raise NotImplementedError

    def add_ray(self, ray, max_range, max_height):
        """Add the input ray to the grid while accounting for the sensor's max range.

        Args:
            ray: (Ray) The ray to be added to the grid.
            max_range: (float) Max range of the sensor
            max_height: (float) Maximum reliable range above and below sensor

        Returns:
            success, end: (bool, Point) The first element indicates whether the addition process
                            was successful. The second element returns the end
                            point of the ray (for visualization purposes)
        """
        raise NotImplementedError

    def add_obs(self, pos):
        """Add the observation at the input position to the map."""
        rays = self.sensor.rays(pos)
        es = []
        for r in rays:
            success, e = self.add_ray(r, self.sensor.max_range, self.sensor.max_height)
            if success:
                es.append(e)

        return es
