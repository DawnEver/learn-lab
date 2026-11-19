"""Domain-free tensor containers a consumer encodes its own data into; numpy only.

:func:`mesh_graph` turns a point set -- optionally with its cells -- into a :class:`PointGraph`: the directed
edges a kernel integral sums over and the quadrature weight each edge carries, so
``sum_j weight_ij * k(x_i, x_j) * v(x_j)`` approximates ``integral k(x_i, y) v(y) dy`` over the neighbourhood of
``x_i`` on ANY point set of the domain. That is what makes a graph operator discretisation-invariant.

- Edges (``source -> target``; every node also reaches itself): within ``radius``, or the ``neighbours``
  nearest (self included), or -- with neither -- the pairs that share a cell.
- Weights: with cells, the source node's lumped measure (each simplex's length/area/volume split equally
  among its ``d + 1`` vertices) -- a quadrature of the integral over the neighbourhood; without cells,
  ``1 / degree(target)`` -- the neighbourhood MEAN, the Monte-Carlo estimate on uniformly scattered points.
- Edge features: the offset ``x_source - x_target`` and its length.

A consumer's mesh IR goes through its own adapter into ``(points, cells)``; nothing here knows a mesh format.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from learn_lab.errors import refuse, require_finite

__all__ = ['PointGraph', 'intervals', 'mesh_graph']


@dataclass(frozen=True, slots=True, eq=False)
class PointGraph:
    """Points ``(N, d)``, edges ``(2, E)`` ``source -> target``, quadrature ``weights (E,)``."""

    points: np.ndarray
    edges: np.ndarray
    weights: np.ndarray

    @property
    def edge_features(self) -> np.ndarray:
        """``(E, d + 1)``: the offset ``x_source - x_target`` and its length."""
        offset = self.points[self.edges[0]] - self.points[self.edges[1]]
        return np.column_stack([offset, np.linalg.norm(offset, axis=1)])

    def require_valid(self, what: str) -> PointGraph:
        points, edges, weights = np.asarray(self.points), np.asarray(self.edges), np.asarray(self.weights)
        if points.ndim != 2 or points.shape[0] < 1:
            refuse(f'{what} is invalid: points are (N, d), got {points.shape}')
        require_finite(points, f'{what} points')
        if edges.ndim != 2 or edges.shape[0] != 2 or not np.issubdtype(edges.dtype, np.integer):
            refuse(f'{what} is invalid: edges are integer (2, E), got {edges.dtype} {edges.shape}')
        if edges.size and (edges.min() < 0 or edges.max() >= len(points)):
            refuse(f'{what} is invalid: an edge index leaves [0, {len(points)})')
        if weights.shape != (edges.shape[1],) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
            refuse(f'{what} is invalid: weights are finite, >= 0 and one per edge, got {weights.shape}')
        return self


def _lumped(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Each node's share of the simplices it is a vertex of."""
    d = points.shape[1]
    if cells.ndim != 2 or cells.shape[1] != d + 1:
        refuse(f'cells are simplices of {d + 1} vertex indices in {d}-D, got {cells.shape}')
    if cells.min() < 0 or cells.max() >= len(points):
        refuse(f'a cell vertex index leaves [0, {len(points)})')
    corners = points[cells]
    spans = corners[:, 1:] - corners[:, :1]
    measure = np.abs(np.linalg.det(spans)) / math.factorial(d)
    lumped = np.zeros(len(points))
    np.add.at(lumped, cells, np.repeat(measure[:, None] / (d + 1), d + 1, axis=1))
    return lumped


def _cell_pairs(cells: np.ndarray, n: int) -> np.ndarray:
    pairs = np.concatenate([cells[:, [a, b]] for a in range(cells.shape[1]) for b in range(cells.shape[1])])
    pairs = np.concatenate([pairs, np.column_stack([np.arange(n), np.arange(n)])])
    return np.unique(pairs, axis=0).T


def mesh_graph(
    points: object,
    cells: object | None = None,
    *,
    radius: float | None = None,
    neighbours: int | None = None,
) -> PointGraph:
    """The :class:`PointGraph` of *points* ``(N, d)``; *cells* ``(M, d + 1)`` simplices give the weights."""
    x = require_finite(np.asarray(points, float), 'the graph points')
    if x.ndim != 2 or len(x) < 1:
        refuse(f'graph points are (N, d), got {x.shape}')
    simplices = None if cells is None else np.asarray(cells, int)
    if radius is not None and neighbours is not None:
        refuse('a graph is built within a radius OR from the nearest neighbours, not both')
    tree = cKDTree(x)
    if radius is not None:
        if not radius > 0:
            refuse(f'a graph radius is > 0, got {radius!r}')
        pairs = tree.query_pairs(radius, output_type='ndarray')
        n = np.arange(len(x))
        edges = np.concatenate([pairs.T, pairs[:, ::-1].T, np.stack([n, n])], axis=1)
    elif neighbours is not None:
        if not 1 <= neighbours <= len(x):
            refuse(f'a graph takes 1 <= neighbours <= {len(x)} nearest points, got {neighbours!r}')
        _, near = tree.query(x, k=neighbours)
        near = np.asarray(near).reshape(len(x), neighbours)
        edges = np.stack([near.ravel(), np.repeat(np.arange(len(x)), neighbours)])
    elif simplices is not None:
        edges = _cell_pairs(simplices, len(x))
    else:
        refuse('a graph needs a radius, a neighbour count or cells to draw its edges from')
    edges = edges.astype(np.int64)
    if simplices is not None:
        weights = _lumped(x, simplices)[edges[0]]
    else:
        degree = np.bincount(edges[1], minlength=len(x))
        weights = 1.0 / degree[edges[1]]
    return PointGraph(x, edges, weights)


def intervals(points: Sequence[float] | np.ndarray) -> np.ndarray:
    """The 1-D cells ``(N - 1, 2)`` joining sorted *points* -- the simplices of a line point set."""
    order = np.argsort(np.asarray(points, float).ravel())
    return np.column_stack([order[:-1], order[1:]])
