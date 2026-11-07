"""D55 -- polynomial chaos expansion over independent uniform inputs on the unit cube.

``y(x) = sum_a c_a Psi_a(x)`` with ``Psi_a = prod_j psi_{a_j}(x_j)``, ``psi_m(x) = sqrt(2m + 1)
P_m(2x - 1)`` the Legendre polynomials ORTHONORMAL under the uniform measure on ``[0, 1]``, over
every multi-index of total degree ``<= degree``; ``c`` by least squares. Orthonormality makes the
moments and Sobol indices read straight off the coefficients: the mean is ``c_0``, the variance
``sum_{a != 0} c_a^2``, the first-order index of input ``i`` the share of the variance carried by
terms in ``x_i`` alone, its total index the share of every term containing ``x_i``. The mean's
gradient is exact. The caller maps its design space onto the unit cube.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numerics_lab.linear.dense import least_squares

from learn_lab.errors import refuse
from learn_lab.models.response.polynomial import monomial_exponents
from learn_lab.protocol import Give, Prediction, query_points, require_gives, training_data

__all__ = ['ChaosExpansion', 'FittedChaos', 'legendre_table']


def legendre_table(x: np.ndarray, degree: int) -> tuple[np.ndarray, np.ndarray]:
    """``psi_m(x)`` and ``d psi_m / dx`` for ``m = 0..degree``, each ``(n, d, degree + 1)``, on ``[0, 1]``."""
    t = 2 * np.asarray(x, dtype=float) - 1
    p = [np.ones_like(t), t]
    dp = [np.zeros_like(t), np.ones_like(t)]
    for m in range(1, degree):
        p.append(((2 * m + 1) * t * p[m] - m * p[m - 1]) / (m + 1))
        dp.append(dp[m - 1] + (2 * m + 1) * p[m])
    norm = np.sqrt(2 * np.arange(degree + 1) + 1)
    return np.stack(p[: degree + 1], axis=-1) * norm, 2 * np.stack(dp[: degree + 1], axis=-1) * norm


def _basis(table: np.ndarray, indices: np.ndarray) -> np.ndarray:
    d = indices.shape[1]
    return np.prod(table[:, np.arange(d)[None, :], indices], axis=2)


@dataclass(frozen=True, slots=True)
class ChaosExpansion:
    """A Legendre chaos of total *degree* on the unit cube."""

    degree: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.degree, int) or self.degree < 1:
            refuse(f'a chaos degree is an integer >= 1, got {self.degree!r}')

    @property
    def gives(self) -> frozenset[Give]:
        return frozenset({Give.GRADIENT})

    def fit(self, x, y) -> FittedChaos:
        x, y = training_data(x, y)
        if np.any((x < 0) | (x > 1)):
            refuse('a Legendre chaos is fitted on the unit cube; encode the points first')
        d = x.shape[1]
        indices = np.vstack([np.zeros((1, d), dtype=int), monomial_exponents(d, self.degree)])
        if len(indices) > len(x):
            refuse(f'a degree-{self.degree} chaos in {d} inputs has {len(indices)} terms, more than {len(x)} points')
        table, _ = legendre_table(x, self.degree)
        return FittedChaos(self.degree, indices, least_squares(_basis(table, indices), y))


@dataclass(frozen=True, slots=True, eq=False)
class FittedChaos:
    """The multi-indices ``(terms, d)`` (the first is all zeros) and coefficients ``(terms, k)``."""

    degree: int
    indices: np.ndarray
    coefficients: np.ndarray

    @property
    def gives(self) -> frozenset[Give]:
        return frozenset({Give.GRADIENT})

    @property
    def mean(self) -> np.ndarray:
        """The output's mean under uniform inputs, ``(k,)``."""
        return self.coefficients[0]

    @property
    def variance(self) -> np.ndarray:
        """The output's variance under uniform inputs, ``(k,)``."""
        return np.sum(self.coefficients[1:] ** 2, axis=0)

    @property
    def first_order(self) -> np.ndarray:
        """First-order Sobol indices, ``(d, k)``."""
        alone = (self.indices > 0) & (np.sum(self.indices > 0, axis=1, keepdims=True) == 1)
        return (alone.T.astype(float) @ self.coefficients**2) / self.variance

    @property
    def total_order(self) -> np.ndarray:
        """Total Sobol indices, ``(d, k)``."""
        return ((self.indices > 0).T.astype(float) @ self.coefficients**2) / self.variance

    def predict(self, x, *, variance: bool = False, gradient: bool = False) -> Prediction:
        require_gives(self.gives, variance=variance, gradient=gradient, what='a chaos expansion')
        q = query_points(x, self.indices.shape[1])
        table, slopes = legendre_table(q, self.degree)
        mean = _basis(table, self.indices) @ self.coefficients
        if not gradient:
            return Prediction(mean)
        d = q.shape[1]
        jac = np.empty((len(q), self.coefficients.shape[1], d))
        for j in range(d):
            swapped = table.copy()
            swapped[:, j, :] = slopes[:, j, :]
            jac[:, :, j] = _basis(swapped, self.indices) @ self.coefficients
        return Prediction(mean, gradient=jac)
