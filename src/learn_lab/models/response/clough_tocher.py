"""The Clough-Tocher interpolant: a piecewise-cubic C1 surface THROUGH the samples of a two-input response.

The nodes are Delaunay-triangulated and each triangle split into three cubic patches (Clough-Tocher), with the
nodal gradients estimated by minimising the surface's curvature (``tolerance`` and ``max_iterations`` steer that
estimate). It reproduces every sample exactly and every plane everywhere -- the bend a low-degree polynomial
cannot hold -- and it is defined only inside the convex hull of its nodes.

So its envelope is that HULL (:class:`~learn_lab.validity.HullEnvelope`), never a box, and a declared domain is
refused: a box wider than the hull would be answered by extrapolation, which learn-lab never does. A caller
with a fallback for outside the hull asks ``fitted.envelope.classify`` first and answers those points itself.

The interpolant is :class:`scipy.interpolate.CloughTocher2DInterpolator`; the fit stores only its nodes, values
and settings and rebuilds it on first use, which is deterministic, so a reloaded artifact answers bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator

from learn_lab.errors import refuse
from learn_lab.protocol import Give, Prediction, query_points, require_gives, training_data
from learn_lab.validity import BoxEnvelope, HullEnvelope

__all__ = ['CloughTocher', 'FittedCloughTocher']


@dataclass(frozen=True, slots=True)
class CloughTocher:
    """A C1 interpolant through ``(n, 2)`` nodes; *tolerance* and *max_iterations* steer the gradient estimate."""

    tolerance: float = 1e-6
    max_iterations: int = 400

    def __post_init__(self) -> None:
        if not self.tolerance > 0:
            refuse(f'a gradient-estimate tolerance is > 0, got {self.tolerance!r}')
        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            refuse(f'a gradient-estimate iteration cap is an integer >= 1, got {self.max_iterations!r}')

    @property
    def gives(self) -> frozenset[Give]:
        return frozenset()

    def fit(self, x, y, *, domain: BoxEnvelope | None = None) -> FittedCloughTocher:
        if domain is not None:
            refuse(
                'an interpolant answers only inside the hull of its nodes; a declared domain wider than that '
                'would be answered by extrapolation -- fit without one and classify against fitted.envelope'
            )
        x, y, _ = training_data(x, y)
        if x.shape[1] != 2:
            refuse(f'a Clough-Tocher interpolant takes two inputs, got {x.shape[1]}')
        envelope = HullEnvelope.of(x)
        if envelope.rank != 2:
            refuse(
                f'a Clough-Tocher interpolant needs nodes spanning the plane; these are collinear (rank {envelope.rank})'
            )
        return FittedCloughTocher(x, y, self.tolerance, self.max_iterations, envelope)


@dataclass(frozen=True, eq=False)
class FittedCloughTocher:
    """The interpolant through ``nodes (n, 2)`` and ``values (n, k)``, answering inside ``envelope`` (their hull)."""

    nodes: np.ndarray
    values: np.ndarray
    tolerance: float
    max_iterations: int
    envelope: HullEnvelope

    @property
    def gives(self) -> frozenset[Give]:
        return frozenset()

    @cached_property
    def _interpolant(self) -> CloughTocher2DInterpolator:
        return CloughTocher2DInterpolator(self.nodes, self.values, tol=self.tolerance, maxiter=self.max_iterations)

    def predict(self, x, *, variance: bool = False, gradient: bool = False) -> Prediction:
        require_gives(self.gives, variance=variance, gradient=gradient, what='a Clough-Tocher interpolant')
        x = query_points(x, self.envelope)
        mean = np.asarray(self._interpolant(x), dtype=float).reshape(len(x), -1)
        undefined = np.flatnonzero(np.isnan(mean).any(axis=1))
        if undefined.size:
            refuse(
                f'{undefined.size} point(s) sit on the hull boundary but outside every triangle; '
                f'first row {int(undefined[0])} = {x[undefined[0]]}'
            )
        return Prediction(mean)
