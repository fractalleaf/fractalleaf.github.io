Title: Solving Scott's pentomino problem with DLX
Date: 2018-07-12 11:00
Category: Puzzles
Tags: dlx, exact cover, pentomino, python
Slug: scott-pentomino-problem
related_posts: dlx
Authors: Søren
Summary: Solving a famous pentomino puzzle using Knuth's DLX algorithm.
Image: /Puzzles/images/scott_distinct_solutions1.svg

[Link to GitHub repository](https://github.com/fractalleaf/exact-cover)

In my previous [post]({filename}/Algorithms/dlx.md), I introduced exact cover problems, and described Knuth's DLX algorithm for solving such problems. In this post, I use the DLX algorithm to solve a famous exact cover problem: [Scott's Pentomino problem]({filename}/pdf/ScottCombiPuz.pdf).

## Problem statement
Scott's pentomino problem can be stated as follows:

> Consider a 8x8 square with a 2x2 hole in the centre. Cover each tile with one of the 12 free [pentominoes](https://en.wikipedia.org/wiki/Pentomino) in such a way that no pentominoes overlap, and each pentomino is used only once.

One of the many solutions to the problem is shown below

<figure align="middle">
  <img src="{filename}/Puzzles/images/dlx_scott_pentomino.svg" title="pentomino solution">
  <figcaption>Scott's pentomino problem". A 8x8 grid with a 2x2 hole in centre, with each pentomino being used once.</figcaption>
</figure>

## Background
Scott's pentomino problem is named after [Dana Scott](https://en.wikipedia.org/wiki/Dana_Scott) and is famous, not because Scott was the first to solve the problem, but because it was one of the first experiments with solving such a problem algorithmically. The problem was chosen because it was known to have a lot of solutions, but relatively few possibilities of placing the pieces on the grid, thus making the problem computationally feasible.

The experiment was run in 1958 on the [MANIAC I](https://en.wikipedia.org/wiki/MANIAC_I) computer at Princeton, where Scott was a graduate student. In the [technical report]({filename}/pdf/ScottCombiPuz.pdf), where Scott published the results, he remarked that he was very pleased that

> The whole problem required only about three and one-half hours.

## Pentominoes
A pentomino is a polygon constructed by five squares connected edge-to-edge. The pentominoes are a subset of the polyminoes: An Nth order polymino are polygons constructed by connecting N squares edge-to-edge, making pentominoes are 5th order polyminoes. Other well known polyminoes are tetrominoes (tetris blocks), which are 4th order polyminoes, and dominoes which are 2nd order polyminoes.

<figure align="middle">
  <img src="{filename}/Puzzles/images/pentomino_types.svg" title="free pentominoes">
  <figcaption>The 12 free pentominoes named according to the convention given by Golomb.</figcaption>
</figure>

Ignoring reflections and rotations, there are 12 "free" pentominoes, which are shown above along with the customary name of each pentomino. It is customary to name the pentominoes after the letter in the alphabeth they resemble the most.

## Placing pentominoes on the grid
To solve the pentomino problem using the DLX algorithm we need to record all valid pentomino positions on the grid. To help with this task, I have created two python classes:

* The `Polymino` class takes the polymino name and coordinates as input, and defines methods for manipulating the pentomino position and orientation.
* The `Grid` class defines a grid (possibly with holes) on which Polyminoes can be placed. The Grid class checks if each polymino placement in valid.

The first step is to create 12 instanes of the `Polymino` class. One for each of the free pentominoes. I have made a convenience function, `generate_polyminoes`, which takes ascii drawings of the polyminoes and generates the `Polymino` instances from those.


```python
from collections import defaultdict
from polymino import Polymino

def generate_polyminoes(ascii_drawing):
    """Generate polymino coordinates from ascii drawing

    Args:
        ascii_drawing (str): An ascii drawing of the polyminoes to be
        generated. Polyminoes must be drawn using ascii characters, where the
        polymino gets named after the ascii character.

    Yields:
        List of Polymino class instances.
    """

    # get coordinates for each polymino in the ascii drawing
    polydic = defaultdict(list)
    for i, row in enumerate(ascii_drawing.split('\n')):
        for j, name in enumerate(row):
            if name != ' ':
                polydic[name].append((i, j))

    # yield Polymino objects with each polymino shifted to (0, 0)
    for key in sorted(polydic.keys()):
        polymino = Polymino(key, polydic[key])
        polymino.absolute_shift(0, 0)
        yield polymino

# Ascii drawing of the 12 free pentominoes
PENTOMINOES = """
    I
    I  L   N                         Y
 FF I  L   N PP TTT       V   W  X  YY ZZ
FF  I  L  NN PP  T  U U   V  WW XXX  Y  Z
 F  I  LL N  P   T  UUU VVV WW   X   Y  ZZ
"""

pentominoes = []
for pentomino in generate_polyminoes(PENTOMINOES):
    pentominoes.append(pentomino)

print("Number of free pentominoes = {}".format(len(pentominoes)))
```

    Number of free pentominoes = 12


The second step is to generate all "fixed" pentominoes by flipping and rotating the pentominoes. The number of distinct orientations for each pentomino vary between 1 to 8 depending on the symmetries of the piece.


```python
from copy import deepcopy

def generate_polymino_orientations(polyminoes):
    """Generate all orientations (flips and rotations) of a polymino.

    Args:
        polyminoes: List of polyminoes to generate orientations of.

    Yields:
        Polymino instance.
    """
    polyminoes = [polyminoes] if isinstance(polyminoes, Polymino) else polyminoes

    for polymino in polyminoes:

        for _ in range(4):
            yield deepcopy(polymino)
            polymino.rotate()

        polymino.flip()

        for _ in range(4):
            yield deepcopy(polymino)
            polymino.rotate()

orientations = []
count = defaultdict(int)
for pentomino in generate_polymino_orientations(pentominoes):
    if pentomino not in orientations: # only include unique orientations
        orientations.append(pentomino)
        count[pentomino.name] += 1

for name, cnt in count.items():
    print("{} pentomino has {} distinct flips/rotations".format(name, cnt))
print("Total number of distinct pentomino orientations = {}".format(len(orientations)))
```

    F pentomino has 8 distinct flips/rotations
    I pentomino has 2 distinct flips/rotations
    L pentomino has 8 distinct flips/rotations
    N pentomino has 8 distinct flips/rotations
    P pentomino has 8 distinct flips/rotations
    T pentomino has 4 distinct flips/rotations
    U pentomino has 4 distinct flips/rotations
    V pentomino has 4 distinct flips/rotations
    W pentomino has 4 distinct flips/rotations
    X pentomino has 1 distinct flips/rotations
    Y pentomino has 8 distinct flips/rotations
    Z pentomino has 4 distinct flips/rotations
    Total number of distinct pentomino orientations = 63


Finally, each fixed pentomino is placed at all valid positions on the grid.


```python
from polymino import Grid

def generate_polymino_positions(polyminoes, grid):
    """Place polyminoes on all valid positions in the grid

    Args:
        polyminoes: List of polyminoes to place on grid
        grid: Grid instance.

    Yields:
        Polymino instance.
    """
    polyminoes = [polyminoes] if isinstance(polyminoes, Polymino) else polyminoes

    for polymino in polyminoes:
        for i in range(grid.min_i, grid.min_i+grid.size[0]):
            for j in range(grid.min_j, grid.min_j+grid.size[1]):
                polymino.absolute_shift(i, j)
                if grid.valid_position(polymino):
                    yield deepcopy(polymino)

# create 8x8 grid with 2x2 hole in the middle
grid = Grid((8, 8), holes=[(3, 3), (3, 4), (4, 3), (4, 4)])
positions = []
for pentomino in generate_polymino_positions(orientations, grid):
    positions.append(pentomino)

print("Total number of fixed pentomino positions on grid = {}".format(len(positions)))
```

    Total number of fixed pentomino positions on grid = 1568


## Solving the problem with DLX

The result of placing the fixed pentominoes on the grid yields a collection of sets, that the DLX algorithm uses for finding exact cover solutions.

The final steps before running the solver is to calculate a list of all the elements present in one or more of the sets in the collection. Each of these elements represent a constraint that must be satisfied once, and only once, by the solution. The problem at hand has two constraints:

* The grid points: A grid point has to be covered by a pentomino, but pentominoes cannot overlap.
* The pentomino names: Each pentomino has to be used once, and cannot be used after that.


```python
# convert list of Polymino instances to list of lists (for input to DLX)
positions = [pentomino.aslist for pentomino in positions]

# DLX also needs a set of labels (constraints) as input. Generate from the list of pentominoes
labels = list(set([element for pentomino in positions for element in pentomino]))
labels = sorted(labels, key=sortkey)
```

Finally, we are ready to run the DLX solver


```python
from dlx import DLX

# sort key for sorted function
# puts strings to the front of list, and coordinates after strings
def sortkey(x):
    x = str(x)
    return (len(x), x)

dlx = DLX(labels, positions)

solutions = dlx.run_search(key=sortkey)
# convert to Grid objects
solutions = [Grid.from_DLX(solution) for solution in solutions]

print("Number of solutions found by DLX = {}".format(len(solutions)))
```

    Number of solutions found by DLX = 520


DLX finds a total of 520 solutions, however, because the algorithm does not take rotations and reflections into account, each solution is replicated eight times with different orientations (see below). The number of distinct solutions is therefore equal to $520/8 = 65$.

<figure align="middle">
  <img src="{filename}/Puzzles/images/scott_pentomino_non_unique.svg" title="Non-distinct solutions">
  <figcaption>Eight solutions of the pentomino problem that are all different orientations of the same solution.</figcaption>
</figure>

It is easy to root out the non-distinct solutions. One simply generates all eight orientations of each solution, only including the solution if it is not already in the list of accepted solutions.


```python
def generate_grid_orientations(grids):
    """Generate all orientations (flips and rotations) of a grid.

    Args:
        grids: List of grids to generate orientations of.

    Yields:
        Grid instance.
    """

    grids = [grids] if isinstance(grids, Grid) else grids

    for grid in grids:
        for _ in range(4):
            yield deepcopy(grid)
            grid.rotate()

        grid.flip()

        for _ in range(4):
            yield deepcopy(grid)
            grid.rotate()

def unique_grids(grids):

    grids = list(grids)

    unique = [grids[0]]

    for grid in grids[1:]:
        is_unique = True
        for orientation in generate_grid_orientations(grid):
            if orientation in unique:
                is_unique = False
                break
        if is_unique:
            unique.append(grid)

    return unique

distinct = unique_grids(solutions)

print("Number of distinct solutions = {}".format(len(distinct)))
```

    Number of distinct solutions = 65


The 65 distinct solutions are shown below

![]({filename}/Puzzles/images/scott_distinct_solutions1.svg)


## Optimising the algorithm


```python
%timeit dlx.run_search(key=sortkey)
print("Number of calls to recursive search function = {}".format(sum(dlx.kcount)))
```

    1min 31s ± 852 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    Number of calls to recursive search function = 293045


The algorithm runs in 1.5 minutes, and makes 293045 calls to the recursive search function. While the runtime is certainly better than the 3.5 hours achieved by Scott in 1958, it is far from optimal.

The most obvious way to reduce the runtime would be to implement the algorithm in a compiled language or run it on a faster computer. However, none of these solutions would reduce the number of calls to the recursive search function, which is what drives the running time.

Scott noticed that you can avoid generating a lot of the redundant solutions by restricting the placement of the X pentomino in such a way that none of the accepted placements can be flipped or rotated into one of the other placements. Doing this, there is in fact only three valid placements of the X pentomino:

![]({filename}/Puzzles/images/X.svg)

Restricting the placement of the X piece reduces the branching of the search tree, and therefore also the number of calls to the recursive search function.

Note, that restricting the placement of the X polymino to the three shown positions does not remove all symmetries as the rightmost placement retains one axis of symmetry with the diagonal. Although it is possible to remove this ambiguity by restricting the placement of one additional pentomino, I will ignore this here, and simply remove all non-distinct solutions after running the solver anew.


```python
print("Number of valid positions before removing X pieces = {}".format(len(positions)))

# remove X pieces from position except those with the desired coordinates
positions = [pos for pos in positions if not (pos[0] == 'X' and pos[1] not in [(5, 4), (5, 5), (4, 5)])]

print("Number of valid positions after removing X pieces = {}".format(len(positions)))

# run DLX algorithm again
dlx = DLX(labels, positions)

solutions = dlx.run_search(key=sortkey)
# convert to Grid objects
solutions = [Grid.from_DLX(solution) for solution in solutions]

print("")
print("Number of solutions found by DLX with X pieces removed = {}".format(len(solutions)))

#remove non-distinct solutions
distinct = unique_grids(solutions)

print("Number of distinct solutions found by DLX with X pieces removed = {}".format(len(distinct)))

print("")
%timeit dlx.run_search(key=sortkey)
print("Number of calls to recursive search function = {}".format(sum(dlx.kcount)))
```

    Number of valid positions before removing X pieces = 1568
    Number of valid positions after removing X pieces = 1547
    
    Number of solutions found by DLX with X pieces removed = 91
    Number of distinct solutions found by DLX with X pieces removed = 65
    
    15.5 s ± 320 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    Number of calls to recursive search function = 49874


Restricting the positioning of the X pentominoes reduces both the runtime and the number of recusive calls by a factor of six.

## Code for Polymino and Grid classes


```python
class Polymino:
    """Polymino piece

    Attributes:
        coord (list of tuples): Polymino coordinates.
        name (Str): Name of the polymino piece.
    """

    def __init__(self, name, coord):
        """init method

        Args:
            name (Str): Name of the polymino piece.
            coord (list of tuples): Polymino coordinates.
        """
        self.name = name
        self.coord = sorted(coord)
        # get boundaries
        self.min_i, self.min_j = map(min, *coord)
        self.max_i, self.max_j = map(max, *coord)

    @classmethod
    def from_list(cls, lst):
        """Polymino ame and coordinates from list

        Args:
            lst (list): List with polymino coordinates, and name

        Returns:
            Polymino instance

        Raises:
            ValueError: If name and/or coordinates missing from list.
        """
        coord = []
        name = ''

        for i in lst:
            if isinstance(i, tuple):
                coord.append(i)
            elif isinstance(i, str):
                name = i

        if not coord:
            raise ValueError("No coordinates in list")
        if name == '':
            raise ValueError("No name in list")

        return cls(name, coord)

    @property
    def limit(self):
        """Get boundaries

        Returns:
            list of boundaries
        """
        return [self.min_i, self.max_i, self.min_j, self.max_j]

    @property
    def size(self):
        """Get size of polymino

        Returns:
            List of size in i and j coordinate.
        """
        return [self.max_i-self.min_i+1, self.max_j-self.min_j+1]

    @property
    def aslist(self):
        """Convert Polymino instance to list

        Returns:
            List with polymino name and coordinates
        """
        return [self.name] + self.coord

    def relative_shift(self, delta_i, delta_j):
        """Shift the coordinates of the Polymino relative to its current
        position.

        Args:
            delta_i (int): Shift of i coordinate
            delta_j (int): Shift of j coordinate
        """
        self.min_i += delta_i
        self.max_i += delta_i
        self.min_j += delta_j
        self.max_j += delta_j

        self.coord = [(i+delta_i, j+delta_j) for i, j in self.coord]

    def absolute_shift(self, i_0, j_0):
        """Shift the coordinates of the Polymino to an absolute position.

        Args:
            i_0 (int): i coordinate value of the upper left corner.
            j_0 (int): j coordinate velue of the upper left corner.
        """
        self.coord = [(i-self.min_i+i_0, j-self.min_j+j_0) for i, j in self.coord]

        self.max_i += i_0 - self.min_i
        self.max_j += j_0 - self.min_j
        self.min_i = i_0
        self.min_j = j_0

    def flip(self, ftype='vertical', reset=True):
        """Flip the polymino around the vertical or horizontal axes.

        The flips are always relative to the origin (0, 0).

        Args:
            ftype (str, optional): Flip around vertical or horizontal axis.
            (Default: vertical)
            reset (bool, optional): Reset the coordinates of the upper left
            corner to the original value after the flip (Default: True).

        Raises:
            ValueError: If ftype is not 'vertical' or 'horizontal'.
        """
        old_min_i, _, old_min_j, _ = self.limit

        if ftype == 'vertical':
            self.coord = sorted([(i, -j) for i, j in self.coord])
            self.min_j, self.max_j = -self.max_j, -self.min_j
        elif ftype == 'horizontal':
            self.coord = sorted([(-i, j) for i, j in self.coord])
            self.min_i, self.max_i = -self.max_i, -self.min_i
        else:
            raise ValueError("ftype must be either horizontal or vertical")

        if reset:
            self.absolute_shift(old_min_i, old_min_j)

    def rotate(self, reset=True):
        """Rotate Polymino 90 degree counter clockvise around (0, 0)

        Args:
            reset (bool, optional): Reset the coordinates of the upper left
            corner to the original value after the flip (Default: True).
        """
        old_min_i, _, old_min_j, _ = self.limit

        self.coord = sorted([(-j, i) for i, j in self.coord])

        self.min_i, self.max_i, self.min_j, self.max_j = \
        -self.max_j, -self.min_j, self.min_i, self.max_i

        if reset:
            self.absolute_shift(old_min_i, old_min_j)

    def ascii(self, empty=' '):
        """Print an ascii drawing of the polymino.

        Args:
            empty (str, optional): Ascii character to use for holes in the grid.

        Returns:
            Str
        """
        height, width = self.size

        grid = []
        for i in range(height):
            grid.append([empty for j in range(width)])

        for i, j in self.coord:
            grid[i-self.min_i][j-self.min_j] = self.name

        return '\n'.join([''.join(row) for row in grid])

    def __str__(self):
        return self.ascii()

    def __hash__(self):
        return hash(tuple(self.aslist))

    def __eq__(self, other):
        if isinstance(other, Polymino):
            return (self.name, self.coord) == (other.name, other.coord)
        return False

class Grid:
    """Grid on which to place polyminoes

    Attributes:
        coord (list of tuples): Grid coordinates.
        polyminoes (List of Polymino objects): Polyminoes on the grid.
        size (tuple): Size of the grid
    """

    def __init__(self, size, i_0=0, j_0=0, polyminoes=None, holes=None):
        """init method

        Args:
            size (tuple): Size of grid (n_i, n_j).
            i_0 (int, optional): i coordinate value at upper left corner
            j_0 (int, optional): j coordinate value at upper left corner
            polyminoes (None, optional): Polyminoes on the grid
            holes (None, optional): Coordinates of "holes" in the grid.
        """
        n_i, n_j = size
        holes = [] if holes is None else holes
        polyminoes = [] if polyminoes is None else polyminoes

        self.size = size
        self.coord = []
        self.polyminoes = []
        self.min_i, self.max_i = i_0, i_0+n_i-1
        self.min_j, self.max_j = j_0, j_0+n_j-1

        # build self.grid with (i, j) coordinates for each gridpoint
        for i in range(n_i):
            for j in range(n_j):
                # only include the grid point if not part of a "hole"
                if (i, j) not in holes:
                    self.coord.append((i, j))

        # check that all polyminoes are in a valid position
        for polymino in polyminoes:
            self.add(polymino)

    @classmethod
    def from_DLX(cls, solution):
        """Generate grid from DLX solution

        Args:
            solution: solution as returned from the DLX class

        Returns:
            Instance of Grid class
        """

        polyminoes = [Polymino.from_list(polymino) for polymino in solution]

        # get boundaries for the grid
        min_i = min([polymino.min_i for polymino in polyminoes])
        max_i = max([polymino.max_i for polymino in polyminoes])
        min_j = min([polymino.min_j for polymino in polyminoes])
        max_j = max([polymino.max_j for polymino in polyminoes])

        # size of the grid from boundaries
        size = (max_i-min_i+1, max_j-min_j+1)

        # Holes in the grid. Assumed to be grid points within the boundary,
        # not coevered by a polymino
        holes = []
        for i in range(min_i, max_i+1):
            for j in range(min_j, max_j+1):
                holes.append((i, j))

        holes = set(holes)
        for polymino in polyminoes:
            holes = holes - set(polymino.coord)

        holes = list(holes)

        return cls(size, min_i, min_j, polyminoes, holes)

    @property
    def limit(self):
        """Get boundaries

        Returns:
            list of boundaries
        """
        return [self.min_i, self.max_i, self.min_j, self.max_j]

    def valid_position(self, polymino):
        """Test to see if a polymino is in a valid position.

        A valid position is on the grid, and without overlapping any other
        polymino

        Args:
            polymino: Polymino instance.

        Returns:
            bool
        """
        set_polymino = set(polymino.coord)

        for poly in self.polyminoes:
            if set_polymino.intersection(poly.coord):
                return False

        if set_polymino.intersection(self.coord) != set_polymino:
            return False

        return True

    def add(self, polymino):
        """Add a polymino to the grid.

        Args:
            polymino: Polymino instance.

        Raises:
            ValueError: If a polymino is outside the grid or covers another
            polymino.
        """
        if self.valid_position(polymino):
            self.polyminoes.append(deepcopy(polymino))
        else:
            raise ValueError("Polymino not in a valid grid position")

    def relative_shift(self, delta_i, delta_j):
        """Shift the coordinates of the Grid relative to its current position.

        Args:
            delta_i (int): Shift of i coordinate
            delta_j (int): Shift of j coordinate
        """
        self.min_i += delta_i
        self.max_i += delta_i
        self.min_j += delta_j
        self.max_j += delta_j

        _ = [p.relative_shift(delta_i, delta_j) for p in self.polyminoes]

    def absolute_shift(self, i_0, j_0):
        """Shift the coordinates of the Grid to an absolute position.

        Args:
            i_0 (int): i coordinate value of the upper left corner.
            j_0 (int): j coordinate velue of the upper left corner.
        """

        delta_i, delta_j = i_0-self.min_i, j_0-self.min_j

        self.relative_shift(delta_i, delta_j)

    def flip(self, ftype='vertical', reset=True):
        """Flip the Grid around the vertical or horizontal axes.

        The flips are always relative to the origin (0, 0).

        Args:
            ftype (str, optional): Flip around vertical or horizontal axis.
            (Default: vertical)
            reset (bool, optional): Reset the coordinates of the upper left
            corner to the original value after the flip (Default: True).

        Raises:
            ValueError: If ftype is not 'vertical' or 'horizontal'.
        """
        old_min_i, _, old_min_j, _ = self.limit

        if ftype == 'vertical':
            self.min_j, self.max_j = -self.max_j, -self.min_j
        elif ftype == 'horizontal':
            self.min_i, self.max_i = -self.max_i, -self.min_i
        else:
            raise ValueError("ftype must be either horizontal or vertical")

        _ = [p.flip(ftype, reset=False) for p in self.polyminoes]

        if reset:
            self.absolute_shift(old_min_i, old_min_j)

    def rotate(self, reset=True):
        """Rotate Grid 90 degree counter clockvise around (0, 0)

        Args:
            reset (bool, optional): Reset the coordinates of the upper left
            corner to the original value after the flip (Default: True).
        """
        old_min_i, _, old_min_j, _ = self.limit

        _ = [p.rotate(reset=False) for p in self.polyminoes]

        self.min_i, self.max_i, self.min_j, self.max_j = \
        -self.max_j, -self.min_j, self.min_i, self.max_i

        if reset:
            self.absolute_shift(old_min_i, old_min_j)

    def ascii(self, empty=' ', gridpoint='+'):
        """Print an ascii drawing of the grid with the polyminoes.

        Args:
            empty (str, optional): Ascii character to use for holes in the grid.
            gridpoint (str, optional): Ascii character to use for grid points
            not covered by a polymino.

        Returns:
            Str
        """
        height, width = self.size

        grid = []
        for i in range(height):
            grid.append([empty for j in range(width)])

        for i, j in self.coord:
            grid[i-self.min_i][j-self.min_j] = gridpoint

        for polymino in self.polyminoes:
            for i, j in polymino.coord:
                grid[i-self.min_i][j-self.min_j] = polymino.name

        return '\n'.join([''.join(row) for row in grid])

    def __str__(self):
        return self.ascii()

    def __eq__(self, other):
        if not isinstance(other, Grid):
            return False
        if self.coord != other.coord:
            return False
        for polymino in self.polyminoes:
            if polymino not in other.polyminoes:
                return False
        return True
```
