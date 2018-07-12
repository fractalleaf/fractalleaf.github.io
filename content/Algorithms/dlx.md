Title: Exact cover and Donald Knuth's DLX algorithm
Date: 2018-07-08 10:15
Category: Algorithms
Tags: exact cover, python, dlx
Slug: dlx
related_posts: scott-pentomino-problem
Authors: Søren
Summary: A Python implementation of Donald Knuth's DLX algorithm for solving exact cover problems.
Image: /Algorithms/images/dlx_scott_pentomino.svg

[Link to GitHub repository](https://github.com/fractalleaf/exact-cover)

## Introduction

Given a collection of sets, $\mathcal{S}$, where each set, $\mathcal{S}^\star$ in $\mathcal{S}$ is a subset of a larger set $\mathcal{X}$. An [exact cover](https://en.wikipedia.org/wiki/Exact_cover) problem involves picking a subcollection of sets from $\mathcal{S}$ such that each element in $\mathcal{X}$ is included in one of the subsets exacly once.

Examples of exact cover problems include:

 * Tiling puzzles (such as [pentomino](https://en.wikipedia.org/wiki/Pentomino) problems) where an area has to be covered in tiles of a certain shape so that the area is covered and no tiles overlap each other.
 * Soduku solving.
 * The [N Queens problem](https://en.wikipedia.org/wiki/Eight_queens_puzzle) is an example of an generalised exact cover problem where some constraints may remain unused.

<figure align="middle">
  <img src="{filename}/Algorithms/images/dlx_scott_pentomino.svg" title="pentomino solution">
  <figcaption>A solution to "Scott's pentomino problem". A 8x8 grid with a 2x2 hole in centre, with each pentomino being used once.</figcaption>
</figure>

## Formal definition
Let $\mathcal{S}^\dagger$ be a subcollection of $\mathcal{S}$ such that $\mathcal{S}^\dagger$ covers $\mathcal{X}$. To be considered a solution to the exact cover problem three conditions have to be satisfied:

 1. The intersection between any two sets in $\mathcal{S}^\dagger$ has to be empty. That is, each element in $\mathcal{X}$ must only be present in one set in $\mathcal{S}^\dagger$.
 2. The union of all sets in $\mathcal{S}^\dagger$ must be equal to $\mathcal{X}$. That is, each element in $\mathcal{X}$ must be present in one set in $\mathcal{S}^\dagger$.
 3. The empty set, $\emptyset$, cannot be part of $\mathcal{S}^\dagger$.

## Example

Given a collection of sets, $\mathcal{S} = \left\{ A, B, C, D, E, F \right\}$, where each set in $\mathcal{S}$ is a subset of $\mathcal{X} = \left\{ 1, 2, 3, 4, 5, 6, 7 \right\}$.

 * $A = \left\{ 1, 4, 7 \right\}$.
 * $B = \left\{ 1, 4 \right\}$.
 * $C = \left\{ 4, 5, 7 \right\}$.
 * $D = \left\{ 3, 5, 6 \right\}$.
 * $E = \left\{ 2, 3, 6, 7 \right\}$.
 * $F = \left\{ 2, 7 \right\}$.

The only collection of subsets in $\mathcal{S}$ that cover $\mathcal{X}$ is $\mathcal{S}^\dagger = \left\{ B, D, F\right\}$.

## Knuth's Algorithm X

[Donald Knuth (2000)](https://arxiv.org/pdf/cs/0011047.pdf) described an algorithm for solving the exact cover problem, which he named "Algorithm X".

Algorithm X works by constructing an indidence matrix, $A$, where each column represents and element in $\mathcal{X}$ and each row represents a set in $\mathcal{S}$. The example above can be represented by the following matrix

$$ A = \begin{bmatrix} 
    1 & 0 & 0 & 1 & 0 & 0 & 1 \\ 
    1 & 0 & 0 & 1 & 0 & 0 & 0 \\ 
    0 & 0 & 0 & 1 & 1 & 0 & 1 \\ 
    0 & 0 & 1 & 0 & 1 & 1 & 0 \\ 
    0 & 1 & 1 & 0 & 0 & 1 & 1 \\ 
    0 & 1 & 0 & 0 & 0 & 0 & 1 \\ 
   \end{bmatrix} $$

The algorithm works by selecting columns and rows in the matrix, recursively reducing it as more contraints are incorporated into the partial solution. If the algorithm can go no further (either because a solution has been found or because the partial solution is a dead end) the algorithm backtracks and goes down another branch. Solutions are returned to the caller. Technically, the algorithm is a recursive, depth-first backtracking algorithm. The steps of the algorithm are:

1. If A is empty, the problem is solved; return solution.
2. Otherwise choose a column, c (deterministically).
3. Choose a row, r, such that A[r, c] = 1 (nondeterministically).
4. If there is no row in c, such that A[r, c] = 1, there is no solution for this position; return
5. Include r in the partial solution.
6. For each j such that A[r, j] = 1, 
     * delete column j from matrix A;
     * for each i such that A[i, j] = 1,
         - delete row i from matrix A.
7. Repeat this algorithm recursively on the reduced matrix A.

It is possible to limit the running time of the algorithm by always choosing the column with the fewest ones. This will limit the branching of the search tree that is being traversed, and hence also the number of recursive calls to the algorithm.

Knuth himself remarks the following about his algorithm:

> Algorithm X is simply a statement of the obvious trial-and-error approach. (Indeed, I can’t think of any other reasonable way to do the job, in general.)


## DLX algorithm and an implementation in Python

Knuth's motivation for describing Algorithm X was a specific implementation he called "Dancing Links". In Dancing Links the incidence matrix is constructed with doubly linked circular lists, which only stores the ones. This has the benefit that the incidence matrix becomes sparse, meaning that it takes up less memory and becomes faster to search through. Furthermore, removing rows and columns in the matrix can be done by simply reassigning pointers to different addresses in the linked lists.

When Algorithm X is implemented with Dancing Links, Knuth calls the algorithm DLX. Below is an implementation of DLX in Python, which is heavily inspired by an implementation by [Nicolau Werneck](https://xor0110.wordpress.com/2011/05/09/dlx-in-python-with-actual-pointers/). Python is not known for its speed, so this implementation can be expected to be significantly slower relative to a good implementation in a compiled language.


```python
class Node:
    """Node in doubly linked list.

    Attributes:
        column: A pointer to the column header.
        down: A pointer to the node below the current node.
        left: A pointer to the node to the left of the current node.
        right: A pointer to the node to the right of the current node.
        up: A pointer to the node above the current node.
    """

    def __init__(self):

        self.left = self
        self.right = self
        self.up = self
        self.down = self
        self.column = self

    def left_sweep(self):
        """Does a left sweep over nodes in the doubly linked list."""
        x = self.left
        while x != self:
            yield x
            x = x.left
        return

    def right_sweep(self):
        """Does a right sweep over nodes in the doubly linked list."""
        x = self.right
        while x != self:
            yield x
            x = x.right
        return

    def down_sweep(self):
        """Does a down sweep over nodes in the doubly linked list."""
        x = self.down
        while x != self:
            yield x
            x = x.down
        return

    def up_sweep(self):
        """Does an up sweep over nodes in the doubly linked list."""
        x = self.up
        while x != self:
            yield x
            x = x.up
        return

class DLX:
    """Implementation of Don Knuth's DLX algorithm.

    Uses the Dancing Links as described in this paper:
        https://arxiv.org/pdf/cs/0011047.pdf.

    Attributes:
        h: Root node for the list header.
        hdic: Dictionary with each key being a column header and its value the
              pointer to the column in the list header.
        kcount: List with Number of calls to the search method for each
                level inthe recursion.
    """

    def __init__(self, labels, rows):
        """Construct the incidence matrix as doubly linked lists.

        Args:
            labels: List with labels of each column.
            rows: List of lists. Each sublist represent a row in the incidence
                  matrix, and must contain the labels of the elements.
        """
        self.h = Node()
        self.hdic = dict()
        self.kcount = [0]

        h = self.h
        hdic = self.hdic

        # make header row
        for label in labels:
            # append new column to end of the doubly linked list
            h.left.right = Node()
            h.left.right.right = h
            h.left.right.left = h.left
            h.left = h.left.right

            h.left.label = label
            h.left.size = 0
            hdic[label] = h.left

        for row in rows:
            last = None
            for rlabel in row:
                element = Node()

                # get column header
                element.column = hdic[rlabel]
                element.column.size += 1

                # append Node to bottom of column
                element.column.up.down = element
                element.up = element.column.up
                element.down = element.column
                element.column.up = element

                if last:
                    element.left = last
                    element.right = last.right
                    last.right.left = element
                    last.right = element
                last = element

    def cover(self, c):
        """Cover column c.

        Args:
            c: Column to cover.
        """
        c.right.left = c.left
        c.left.right = c.right
        for i in c.down_sweep():
            for j in i.right_sweep():
                j.down.up = j.up
                j.up.down = j.down
                j.column.size -= 1

    def uncover(self, c):
        """Uncover column c.

        Args:
            c: Column to uncover
        """
        for i in c.up_sweep():
            for j in i.left_sweep():
                j.column.size += 1
                j.down.up = j
                j.up.down = j
        c.right.left = c
        c.left.right = c

    def search(self, k=0, o=None):
        """Recursive search algorithm to find exact cover solutions.

        Args:
            k: Level of the recursive call. Should initially be called with
               k=0.
            o: List of rows in the (partial) solution up to this point.

        Yields:
            List of rows constituting a solution.
        """
        if o is None:
            o = []

        if len(self.kcount) <= k:
            self.kcount.append(0)
        self.kcount[k] += 1

        if self.h.right == self.h:
            yield o
            return

        # choose the smallest column
        size = 10**9
        for column in self.h.right_sweep():
            if column.size < size:
                size = column.size
                c = column

        self.cover(c)
        for r in c.down_sweep():
            o_k = r
            for j in r.right_sweep():
                self.cover(j.column)
            yield from self.search(k=k+1, o=o+[o_k])

            for j in r.left_sweep():
                self.uncover(j.column)
        self.uncover(c)

    def get_row_labels(self, row, sort=True, key=str, reverse=False):
        """Get labels of a row in the incidence matrix.

        Args:
            row: Node in the incidence matrix.
            sort (bool): Sort labels.
            key (func): Key function to sort on (Default: str)
            reverse (bool, optional): Reverse sort.

        Returns:
            List of all column labels in the row.
        """
        labels = [row.column.label]
        for r in row.right_sweep():
            labels.append(r.column.label)

        if sort:
            labels = sorted(labels, key=key, reverse=reverse)
        return labels

    def run_search(self, **kw):
        """Wrapper for search method.

        Runs search iterator, gets the labels for the rows that are part of the
        solution, and returns the list of solutions.

        Args:
            **kw: Keyword arguments for get_row_labels.

        Returns:
            List of solutions.
        """
        self.kcount = [0] # reset call counter

        solutions = []
        for solution in self.search():
            solutions.append([self.get_row_labels(s, **kw) for s in solution])

        return solutions
```

## Test

I'm running a simple test to see if the algorithm works.


```python
cols = [1, 2, 3, 4, 5, 6, 7] # columns in incidence matrix
rows = [[1, 4, 7], # rows in incidence matrix
        [1, 4],
        [4, 5, 7],
        [3, 5, 6],
        [2, 3, 6, 7],
        [2, 7]]

# initialize DLX class
cover = DLX(cols, rows)

# run DLX algorithm and print solutions
solutions = cover.run_search()

print("Expected solution: [1, 4] + [3, 5, 6] + [2, 7]")
print("Found solution:", solutions)
print("Total number of calls to recursive search: {}".format(sum(cover.kcount)))
print("")
% timeit cover.run_search()
```

    Expected solution: [1, 4] + [3, 5, 6] + [2, 7]
    Found solution: [[[1, 4], [3, 5, 6], [2, 7]]]
    Total number of calls to recursive search: 5
    
    50.7 µs ± 751 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)

