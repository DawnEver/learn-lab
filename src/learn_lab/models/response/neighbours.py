"""D52 -- k nearest neighbours regression (Euclidean distance).

``uniform`` averages the k nearest values; ``distance`` weights each by ``1 / distance`` and, at a
query that coincides with training points, returns the mean of those points' values alone. The
prediction is piecewise, so it gives no gradient and no variance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from learn_lab.errors import refuse
from learn_lab.protocol import Give, Prediction, query_points, require_gives, training_data

__all__ = ['NEIGHBOUR_WEIGHTS', 'FittedNeighbours', 'NearestNeighbours']

#: The declared weightings.
NEIGHBOUR_WEIGHTS: Final = ('uniform', 'distance')


@dataclass(frozen=True, slots=True)
class NearestNeighbours:
    """The mean of the *k* nearest measured values, weighted per :data:`NEIGHBOUR_WEIGHTS`."""

    k: int = 5
    weights: str = 'uniform'

    def __post_init__(self) -> None:
        if not isinstance(self.k, int) or self.k < 1:
            refuse(f'k is an integer >= 1, got {self.k!r}')
        if self.weights not in NEIGHBOUR_WEIGHTS:
            refuse(f'neighbour weights are one of {list(NEIGHBOUR_WEIGHTS)}, got {self.weights!r}')

    @property
    def gives(self) -> frozenset[Give]:
        return frozenset()

    def fit(self, x, y) -> FittedNeighbours:
        x, y = training_data(x, y)
        if self.k > len(x):
            refuse(f'{self.k} neighbours were asked for and only {len(x)} points were measured')
        return FittedNeighbours(self.k, self.weights, x, y)


@dataclass(frozen=True, slots=True, eq=False)
class FittedNeighbours:
    """The measured points and values the neighbours are drawn from."""

    k: int
    weights: str
    x: np.ndarray
    y: np.ndarray

    @property
    def gives(self) -> frozenset[Give]:
        return frozenset()

    def predict(self, x, *, variance: bool = False, gradient: bool = False) -> Prediction:
        require_gives(self.gives, variance=variance, gradient=gradient, what='nearest neighbours')
        q = query_points(x, self.x.shape[1])
        distance = np.sqrt(np.maximum(np.sum((q[:, None, :] - self.x[None, :, :]) ** 2, axis=2), 0.0))
        nearest = np.argsort(distance, axis=1, kind='stable')[:, : self.k]
        near = np.take_along_axis(distance, nearest, axis=1)
        if self.weights == 'uniform':
            w = np.ones_like(near)
        else:
            exact = near == 0.0
            with np.errstate(divide='ignore'):
                w = np.where(exact.any(axis=1, keepdims=True), exact.astype(float), 1.0 / near)
        w = w / w.sum(axis=1, keepdims=True)
        return Prediction(np.einsum('mk,mkc->mc', w, self.y[nearest]))
