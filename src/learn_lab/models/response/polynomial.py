"""D43 -- polynomial response surfaces: ordinary least squares, ridge, lasso.

The basis is every monomial of total degree 1..``degree`` in the inputs; the intercept is fitted
UNPENALISED by centring the basis and the values (so a penalty shrinks the shape, never the level).

* no penalty -- the minimum-norm least-squares solution;
* :class:`Ridge` -- ``|y - X w|^2 + alpha |w|^2``, solved as the augmented least-squares problem;
* :class:`Lasso` -- ``|y - X w|^2 / (2 n) + alpha |w|_1``, by cyclic coordinate descent with soft
  thresholding until no coefficient moves by more than ``tolerance``.

The mean's gradient is the exact derivative of the polynomial.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np
from numerics_lab.linear.dense import least_squares

from learn_lab.errors import refuse
from learn_lab.protocol import Give, Prediction, query_points, require_gives, training_data
from learn_lab.validity import BoxEnvelope

__all__ = ['FittedPolynomial', 'Lasso', 'Polynomial', 'Ridge', 'monomial_exponents']


@dataclass(frozen=True, slots=True)
class Ridge:
    """An l2 penalty ``alpha |w|^2`` (alpha > 0, in the squared units of the values)."""

    alpha: float

    def __post_init__(self) -> None:
        if not self.alpha > 0:
            refuse(f'a ridge penalty is > 0, got {self.alpha!r}')


@dataclass(frozen=True, slots=True)
class Lasso:
    """An l1 penalty ``alpha |w|_1`` on the mean squared error / 2; *tolerance* is in coefficient units."""

    alpha: float
    tolerance: float = 1e-12
    max_sweeps: int = 100_000

    def __post_init__(self) -> None:
        if not self.alpha > 0:
            refuse(f'a lasso penalty is > 0, got {self.alpha!r}')


def monomial_exponents(d: int, degree: int) -> np.ndarray:
    """``(p, d)`` exponents of every monomial of total degree 1..*degree* in *d* inputs."""
    rows = [
        np.bincount(combo, minlength=d)
        for total in range(1, degree + 1)
        for combo in combinations_with_replacement(range(d), total)
    ]
    return np.array(rows, dtype=int).reshape(-1, d)


def _features(x: np.ndarray, exponents: np.ndarray) -> np.ndarray:
    return np.prod(x[:, None, :] ** exponents[None, :, :], axis=2)


def _lasso(features: np.ndarray, values: np.ndarray, penalty: Lasso) -> np.ndarray:
    n, p = features.shape
    norms = np.einsum('ij,ij->j', features, features)
    w = np.zeros(p)
    residual = values.copy()
    for _ in range(penalty.max_sweeps):
        largest = 0.0
        for j in range(p):
            if norms[j] == 0.0:
                continue
            rho = features[:, j] @ residual + norms[j] * w[j]
            new = np.sign(rho) * max(abs(rho) - n * penalty.alpha, 0.0) / norms[j]
            if new != w[j]:
                residual -= features[:, j] * (new - w[j])
                largest = max(largest, abs(new - w[j]))
                w[j] = new
        if largest <= penalty.tolerance:
            return w
    refuse(f'lasso coordinate descent did not settle to {penalty.tolerance} in {penalty.max_sweeps} sweeps')
    return None


@dataclass(frozen=True, slots=True)
class Polynomial:
    """A polynomial surface of total *degree*, with an optional :class:`Ridge` or :class:`Lasso` penalty."""

    degree: int = 2
    penalty: Ridge | Lasso | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.degree, int) or self.degree < 1:
            refuse(f'a polynomial degree is an integer >= 1, got {self.degree!r}')

    @property
    def gives(self) -> frozenset[Give]:
        return frozenset({Give.GRADIENT})

    def fit(self, x, y, *, domain: BoxEnvelope | None = None) -> FittedPolynomial:
        x, y, envelope = training_data(x, y, domain)
        exponents = monomial_exponents(x.shape[1], self.degree)
        features = _features(x, exponents)
        centre, level = features.mean(axis=0), y.mean(axis=0)
        a, b = features - centre, y - level
        if isinstance(self.penalty, Lasso):
            w = np.column_stack([_lasso(a, b[:, c], self.penalty) for c in range(y.shape[1])])
        elif isinstance(self.penalty, Ridge):
            p = a.shape[1]
            w = least_squares(
                np.vstack([a, np.sqrt(self.penalty.alpha) * np.eye(p)]), np.vstack([b, np.zeros((p, b.shape[1]))])
            )
        else:
            w = least_squares(a, b)
        return FittedPolynomial(exponents, w, level - centre @ w, envelope)


@dataclass(frozen=True, slots=True, eq=False)
class FittedPolynomial:
    """``y = intercept + features(x) @ coefficients``; ``coefficients`` is ``(terms, outputs)``."""

    exponents: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    envelope: BoxEnvelope

    @property
    def gives(self) -> frozenset[Give]:
        return frozenset({Give.GRADIENT})

    def predict(self, x, *, variance: bool = False, gradient: bool = False) -> Prediction:
        require_gives(self.gives, variance=variance, gradient=gradient, what='a polynomial surface')
        x = query_points(x, self.envelope)
        mean = self.intercept + _features(x, self.exponents) @ self.coefficients
        if not gradient:
            return Prediction(mean)
        d = x.shape[1]
        slopes = np.empty((x.shape[0], self.coefficients.shape[1], d))
        for i in range(d):
            lowered = self.exponents.copy()
            lowered[:, i] = np.maximum(lowered[:, i] - 1, 0)
            dfeatures = self.exponents[:, i] * _features(x, lowered)
            slopes[:, :, i] = dfeatures @ self.coefficients
        return Prediction(mean, gradient=slopes)
