"""Ray and Sensor classes for 16-761: Mobile Robot Algorithms Laboratory

Author(s): Kshitij Goel, Andrew Jong, Rebecca Martin, Wennie Tabib
"""

import numpy as np

from .grid import Point


class Ray:
    """A ray in 3D space.

    Attributes:
        o: (Point) Origin of the ray
        d: (Point) Direction (unit) vector for the ray
    """

    def __init__(self, origin=Point(0.0, 0.0, 0.0), direction=Point(0.0, 0.0, 0.0)):
        """Initially the origin and direction are zero."""
        self.o = origin
        self.d = direction

    def point_at_dist(self, t):
        """Get the point at the input distance along this ray.

        Args:
            t: (float) Distance along the ray.

        Returns:
            p: (Point) Point along this ray at distance `t`.
        """
        # TODO: Assignment 1, Problem 1.2
        return self.o + (self.d * t)

    def to_numpy(self):
        return np.hstack((self.o.to_numpy(), self.d.to_numpy()))


class Sensor:
    """A range sensor in 3D space.

    Attributes:
        num_rays: (int) Total number of rays.
        max_range: (float) Maximum reliable range around the sensor.
        max_height: (float) Maximum reliable range above and below sensor
    """

    def __init__(self, max_range=1.0, max_height=3.0, num_rays=70):
        """Default max range is 1.0 meters and the number of rays are 70."""
        self.num_rays = num_rays
        self.max_range = max_range
        self.max_height = max_height

    def rays(self, pos):
        """Generate rays at the input position.

        Rays around the given position `pos` at equal angular intervals within
        `[0, 2.0 * np.pi]` (i.e., 0 to 360 degrees).  Do not double count `0.0`
        and `2.0 * np.pi`

        Args:
            pos: (Point) Position of the sensor in 3D point space.

        Returns:
            rays: (list of Ray) List of `num_ray` amount of ray objects.
        """
        # TODO: Assignment 2, Problem 1.2
        # Hint 1: Utilize the `endpoint` option in the `np.linspace` function.
        # Hint 2: Should not require using `self.max_range`.
        rays = []
        dir_vecs = [Point(np.cos(a), np.sin(a), float(h)) \
                    for a in np.linspace(0.0, 2.0 * np.pi, int(self.num_rays/(2*self.max_height+1)), endpoint=False) \
                    for h in range(-int(self.max_height), int(self.max_height+1), 1)]
        for d in dir_vecs:
            ray = Ray(pos, d / abs(d))
            rays.append(ray)

        return rays
