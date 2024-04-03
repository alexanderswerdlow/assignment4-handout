"""Cell, Point, and Grid classes for 16-761: Mobile Robot Algorithms Laboratory

Author(s): Kshitij Goel, Andrew Jong, Rebecca Martin, Wennie Tabib
"""

from math import floor

import numpy as np

from copy import copy


class Cell:
    """A single cell in the occupancy grid map.

    Attributes:
        row: Row number of the cell. Corresponds to Y-axis in 3D space.
        col: Col number of the cell. Corresponds to X-axis in 3D space.
        layer: Layer number of the cell. Corresponds to the Z-axis in the 3D space.
    """

    def __init__(self, row=0, col=0, layer=0):
        """Initializes the row and col for this cell to be 0."""
        self.row = row
        self.col = col
        self.layer = layer

    def __str__(self):
        return f'Cell(row: {self.row}, col: {self.col}, layer: {self.layer})'

    def to_numpy(self):
        """Return a numpy array with the cell row and col."""
        return np.array([self.row, self.col, self.layer], dtype=int)

    def __eq__(self, second):
        if isinstance(second, Cell):
            return ((self.row == second.row) and (self.col == second.col) and (self.layer == second.layer))
        else:
            raise TypeError('Argument type must be Cell.')


class Point:
    """A point in the 3D space.

    Attributes:
        x: A floating point value for the x coordinate of the 3D point.
        y: A floating point value for the y coordinate of the 3D point.
        z: A floating point value for the z coordinate of the 3D point.
    """

    def __init__(self, x=0.0, y=0.0, z=0.0):
        """Initializes the x, y, and z for this point to be 0.0"""
        self.x = x
        self.y = y
        self.z = z

    def __abs__(self):
        return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5

    def __str__(self):
        return f'Point(x: {self.x}, y: {self.y}, z: {self.z})'

    def __eq__(self, second):
        if isinstance(second, Point):
            return ((self.x == second.x) and (self.y == second.y) and (self.z == second.z))
        else:
            raise TypeError('Argument type must be Point.')

    def __ne__(self, second):
        if isinstance(second, Point):
            return ((self.x != second.x) or (self.y != second.y) or (self.z != second.z))
        else:
            raise TypeError('Argument type must be Point.')

    def __add__(self, second):
        if isinstance(second, Point):
            return Point(self.x + second.x, self.y + second.y, self.z + second.z)
        elif isinstance(second, float):
            return Point(self.x + second, self.y + second, self.z + second)
        else:
            raise TypeError('Argument type must be either float or Point.')

    def __sub__(self, second):
        if isinstance(second, Point):
            return Point(self.x - second.x, self.y - second.y, self.z - second.z)
        elif isinstance(second, float):
            # when subtracting with a float, always post-subtract
            # YES: Point(1.2, 3.2) - 5.0
            # NO: 5.0 - Point(1.2, 3.2)
            return Point(self.x - second, self.y - second, self.z - second)
        else:
            raise TypeError('Argument type must be either float or Point.')

    def __mul__(self, second):
        if isinstance(second, Point):
            return (self.x * second.x + self.y * second.y, self.z * second.z)
        elif isinstance(second, float):
            # when multiplying with a float, always post-multiply
            # YES: Point(1.2, 3.2) * 5.0
            # NO: 5.0 * Point(1.2, 3.2)
            return Point(self.x * second, self.y * second, self.z * second)
        else:
            raise TypeError('Argument type must be either float or Point.')

    def __truediv__(self, second):
        if isinstance(second, float):
            # when dividing by a float, always post-divide
            # YES: Point(1.2, 3.2) / 5.0
            # NO: 5.0 / Point(1.2, 3.2)
            if np.abs(second - 0.0) < 1e-12:
                raise ValueError(
                    'Divide by zero error. Second argument is too close to zero.')
            else:
                return Point(self.x / second, self.y / second, self.z / second)
        else:
            raise TypeError('Argument type must be float.')

    def to_numpy(self):
        """Return a numpy array with the x and y coordinates."""
        return np.array([self.x, self.y, self.z], dtype=float)


class Grid3D:
    """Occupancy grid data structure.

    Attributes:
        resolution: (float) The size of each cell in meters.
        width: (int) Maximum number of columns in the grid.
        depth: (int) Maximum number of rows in the grid.
        height: (int) Maximum number of layers in the grid.
        min_clamp: (float) Logodds corresponding to minimum possible probability
        (to ensure numerical stability).
        max_clamp: (float) Logodds corresponding to maximum possible probability
        (to ensure numerical stability).
        free_thres: (float) Logodds below which a cell is considered free
        occ_thres: (float) Logodds above which a cell is considered occupied
        N: (int) Total number of cells in the grid
        data: Linear array of containing the logodds of this occupancy grid
    """

    def __init__(self, res, W, D, H, min_clamp, max_clamp, free_thres=0.13, occ_thres=0.7):
        """Initialize the grid data structure.

        Note that min_clamp, max_clamp, free_thres, and occ_thres inputs to this constructor
        are probabilities. You have to convert them to logodds internally for numerical stability.
        """
        self.resolution = res

        self.width = int(W)
        self.depth = int(D)
        self.height = int(H)

        self.min_clamp = self.logodds(min_clamp)
        self.max_clamp = self.logodds(max_clamp)
        self.free_thres = self.logodds(free_thres)
        self.occ_thres = self.logodds(occ_thres)

        self.N = self.depth * self.width * self.height 

        # Initially all the logodds values are zero. A logodds value of zero corresponds to an occupancy probability of 0.5.
        # Why do we not simply use a numpy N-D array? Because in real life you'll probably implement this in 
        # C++ (mapping is computationally expensive), so it's good to learn how to manipulate 
        # the data without numpy training wheels
        self.data = [0.0] * self.N

    def to_numpy(self):
        """Export the grid in the form of a 3D numpy matrix.

        Each entry in this matrix is the probability of occupancy for the cell.
        """
        g = np.zeros((self.depth, self.width, self.height))
        for row in range(self.depth):
            for col in range(self.width):
                for layer in range(self.height):
                    v = self.get_row_col_layer(row, col, layer)
                    g[row][col][layer] = self.probability(v)

        return g

    def to_index(self, cell):
        """Return the index into the data array (self.data) for the input cell.

        Args:
            cell: (Cell) The input cell for which the index in data array is requested.

        Returns:
            idx: (int) Index in the data array for the cell
        """
        # TODO: Assignment 2, Problem 1.1 (test_data_structure)
        return floor(cell.row * self.width * self.height + cell.col * self.height + cell.layer)

    def from_index(self, idx):
        """Return the cell in grid for the input index.

        Args:
            idx: (int) Index in the data array for which the cell is requested.

        Returns:
            cell: (Cell) Cell corresponding to the index.
        """
        # TODO: Assignment 2, Problem 1.1 (test_data_structure)
        row = floor(idx / (self.width * self.height))
        col = floor((idx - row * self.width * self.height) / self.height)
        layer = idx - row * self.width * self.height - col * self.height

        return Cell(row, col, layer)

    def get(self, idx):
        """Return the cell value corresponding to the input index.

        Args:
            idx: (int) Index in the data array for which the data is requested.

        Returns:
            val: (float) Value in the data array for idx
        """
        # TODO: Assignment 2, Problem 1.1 (test_data_structure)
        return self.data[idx]

    def get_cell(self, cell):
        """Return the cell value corresponding to the input index."""
        # TODO: Assignment 2, Problem 1.1 (test_data_structure)
        # Hint: Use the `to_index` and `get` methods.
        idx = self.to_index(cell)
        return self.get(idx)

    def get_row_col_layer(self, row, col, layer):
        """Return the cell value corresponding to the row and col."""
        # TODO: Assignment 2, Problem 1.1 (test_data_structure)
        # Hint: Use the `get_cell` method and the `Cell` constructor.
        return self.get_cell(Cell(row, col, layer))

    def set(self, idx, value):
        """Set the cell to value corresponding to the input idx."""
        # TODO: Assignment 2, Problem 1.1 (test_data_structure)
        self.data[idx] = value

    def set_cell(self, cell, value):
        """Set the cell to value corresponding to the input cell."""
        # TODO: Assignment 2, Problem 1.1 (test_data_structure)
        # Hint: Use `to_index` and `set` methods.
        if cell.row < 0 or cell.row >= self.depth or cell.col < 0 or cell.col >= self.width or cell.layer < 0 or cell.layer >= self.height:
            return False
        idx = self.to_index(cell)
        self.set(idx, value)
        return True

    def set_row_col_layer(self, row, col, layer, value):
        """Set the cell to value corresponding to the input row and col."""
        # TODO: Assignment 2, Problem 1.1 (test_data_structure)
        # Hint: Use the `set_cell` method and the `Cell` constructor.
        return self.set_cell(Cell(row, col, layer), value)

    def probability(self, logodds):
        """Convert input logodds to probability.

        Args:
            logodds: (float) Logodds representation of occupancy.

        Returns:
            prob: (float) Probability representation of occupancy.
        """
        # TODO: Assignment 2, Problem 1.1 (test_data_structure)
        return 1.0 - (1.0 / (1.0 + np.exp(logodds)))

    def logodds(self, probability):
        """Convert input probability to logodds.

        Args:
            logodds: (float) Logodds representation of occupancy.

        Returns:
            prob: (float) Probability representation of occupancy.
        """
        # TODO: Assignment 2, Problem 1.1 (test_data_structure)
        return np.log(probability / (1.0 - probability))

    def cell_to_point(self, cell):
        """Get the cell's lower-left corner in 2D point space.

        Args:
            cell: (Cell) Input cell.

        Returns:
            point: (Point) Lower-left corner in 2D space.
        """
        # TODO: Assignment 2, Problem 1.1 (test_data_structure)
        x = float(cell.col) * self.resolution
        y = float(cell.row) * self.resolution
        z = float(cell.layer) * self.resolution
        return Point(x, y, z)

    def cell_to_point_row_col_layer(self, row, col, layer):
        """Get the point for the lower-left corner of the cell represented by input row and col."""
        # TODO: Assignment 2, Problem 1.1 (test_data_structure)
        # Hint: Use the `cell_to_point` function and the `Cell` constructor.
        return self.cell_to_point(Cell(row, col, layer))

    def point_to_cell(self, point):
        """Get the cell position (i.e., bottom left hand corner) given the point.

        Args:
            point: (Point) Query point

        Returns:
            cell: (Cell) Cell in the grid corresponding to the query point.
        """
        # TODO: Assignment 2, Problem 1.1 (test_traversal)
        row = floor(point.y / self.resolution)
        col = floor(point.x / self.resolution)
        layer = floor(point.z / self.resolution)
        return Cell(row, col, layer)

    def is_in_grid(self, cell):
        """Is the cell inside this grid? Return True if yes, False otherwise."""
        # TODO: Assignment 2, Problem 1.1 (test_traversal)
        return ((cell.col < self.width) and (cell.row < self.depth) and (cell.layer < self.height)
                and (cell.col >= 0) and (cell.row >= 0) and (cell.layer >= 0))

    # AJ TODO: need to convert this from 2D to 3D. this is still the 2D version
    def traverse(self, start, end):
        """Figure out the cells that the ray from start to end traverses.

        Corner cases that must be accounted for:
        - Check that the start point is inside the grid. Return (False, None) otherwise.
        - If start and end points coincide, return (True, [start cell]).
        - End point can be outside the grid. The ray casting must stop at the edges of the grid.
        - Perfectly horizontal and vertical rays.
        - Ray starts and ends in the same cell.

        Args:
            start: (Point) Start point of the ray
            end: (Point) End point of the ray

        Returns:
            success, raycells: (bool, list of Cell) If the traversal was successful, success is True
                                and raycells is the list of traversed cells (including the starting
                                cell). Otherwise, success is False and raycells is None.
        """
        # TODO: Assignment 2, Problem 1.1 (test_traversal)

        # Array of cells to return
        raycells = []

        start_cell = self.point_to_cell(start)
        end_cell = self.point_to_cell(end)

        curr_cell = Cell()
        curr_cell = start_cell

        # Make sure the start point is in the map
        if not self.is_in_grid(start_cell):
            print('Start point is not in the grid.')
            return False, None
        # Don't check the end point, we will terminate once the ray leaves the map.


        # Corner case: Ray is in only one cell
        # Solution: Return the starting cell
        if start_cell == end_cell:  # shouldn't be necesary due to the while loop condition
            raycells.append(start_cell)
            return True, raycells

        magnitude = abs(end - start) if start_cell != end_cell else 1.0  # avoid div by 0 error
        direction = (end - start) / magnitude

        cell_boundary = self.cell_to_point(curr_cell)

        step_col = -1
        if direction.x > 0.0:
            # sets to the upper right corner instead of lower left corner
            step_col = 1
            cell_boundary.x += self.resolution

        step_row = -1
        if direction.y > 0.0:
            step_row = 1
            cell_boundary.y += self.resolution

        step_layer = -1
        if direction.z > 0.0:
            step_layer = 1
            cell_boundary.z += self.resolution

        tmax = Point()
        tdelta = Point()

        # Corner cases: Horizontal and vertical rays
        # Solution: set to inf, s.t. will never increment
        tmax.x = (cell_boundary.x - start.x) * (1.0 / direction.x) if direction.x != 0.0 else float('inf')
        tdelta.x = self.resolution * float(step_col) * (1.0 / direction.x) if direction.x != 0.0 else None # if None is ever incremented, will complain. shouldn't happen

        tmax.y = (cell_boundary.y - start.y) * (1.0 / direction.y) if direction.y != 0.0 else float('inf')
        tdelta.y = self.resolution * float(step_row) * (1.0 / direction.y) if direction.y != 0.0 else None
        
        tmax.z = (cell_boundary.z - start.z) * (1.0 / direction.z) if direction.z != 0.0 else float('inf')
        tdelta.z = self.resolution * float(step_layer) * (1.0 / direction.z) if direction.z != 0.0 else None

        while True:
            # Invalid cell reached; exit with the currently traversed cells.
            if not self.is_in_grid(curr_cell):
                return True, raycells

            # Add curr_cell to the traversed cells
            raycells.append(copy(curr_cell))

            # Escape after inserting the end cell once
            if curr_cell == end_cell:
                break

            if tmax.x < tmax.y:
                if tmax.x  < tmax.z:
                    curr_cell.col += step_col
                    tmax.x += tdelta.x
                else:
                    curr_cell.layer += step_layer
                    tmax.z += tdelta.z
            else:
                if tmax.y < tmax.z:
                    curr_cell.row += step_row
                    tmax.y += tdelta.y
                else:
                    curr_cell.layer += step_layer
                    tmax.z += tdelta.z

        return True, raycells

    def is_cell_free(self, cell):
        """Is the cell free? Return True if yes, False otherwise."""
        # TODO: Assignment 2, Problem 1.3
        # Hint: Use `get_cell` and `free_thres`
        val = self.get_cell(cell)
        return val <= self.free_thres

    def is_cell_occupied(self, cell):
        """Is the cell occupied? Return True if yes, False otherwise."""
        # TODO: Assignment 2, Problem 1.3
        # Hint: Use `get_cell` and `occ_thres`
        val = self.get_cell(cell)
        return val >= self.occ_thres

    def is_cell_unknown(self, cell):
        """Is the cell unknown? Return True if yes, False otherwise."""
        # TODO: Assignment 2, Problem 1.3
        # Hint: Use `get_cell`, `occ_thres`, and `free_thres`
        val = self.get_cell(cell)
        return ((val < self.occ_thres) and (val > self.free_thres))
