"""D44 -- radial basis interpolation with a polynomial tail.

``s(x) = sum_i c_i phi(epsilon |x - x_i|) + sum_j a_j p_j(x)`` with ``p_j`` every monomial of total
degree 0..``degree``. The coefficients solve the saddle system

    [[Phi + smoothing I, P], [P^T, 0]] [c; a] = [y; 0]

through numerics-lab's KKT solve (the system is symmetric and indefinite). ``smoothing = 0``
interpolates. A conditionally positive definite kernel needs a tail of at least its minimum degree;
a lower one is refused. The kernels and their signs are the usual ones (``-r`` for linear, ``-r^5``
for quintic, ``-sqrt(1 + r^2)`` for the multiquadric) so that each is positive definite on the
polynomial-orthogonal subspace. The mean's gradient is exact.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
from numerics_lab.linear.kkt import solve_kkt

from learn_lab.errors import refuse
from learn_lab.models.response.polynomial import monomial_exponents
from learn_lab.protocol import Give, Prediction, query_points, require_gives, training_data
from learn_lab.validity import BoxEnvelope

__all__ = ['RADIAL_KERNELS', 'FittedRadialBasis', 'RadialBasis']

_Fn = Callable[[np.ndarray], np.ndarray]


def _tps(s: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(s > 0, s**2 * np.log(np.where(s > 0, s, 1.0)), 0.0)


def _tps_slope(s: np.ndarray) -> np.ndarray:
    return np.where(s > 0, 2 * np.log(np.where(s > 0, s, 1.0)) + 1, 0.0)


def _linear_slope(s: np.ndarray) -> np.ndarray:
    return np.where(s > 0, -1 / np.where(s > 0, s, 1.0), 0.0)


#: kernel -> (phi(s), phi'(s) / s, minimum tail degree; -1 = none needed).
_KERNELS: Final[dict[str, tuple[_Fn, _Fn, int]]] = {
    'linear': (lambda s: -s, _linear_slope, 0),
    'thin_plate_spline': (_tps, _tps_slope, 1),
    'cubic': (lambda s: s**3, lambda s: 3 * s, 1),
    'quintic': (lambda s: -(s**5), lambda s: -5 * s**3, 2),
    'multiquadric': (lambda s: -np.sqrt(1 + s**2), lambda s: -1 / np.sqrt(1 + s**2), 0),
    'inverse_multiquadric': (lambda s: 1 / np.sqrt(1 + s**2), lambda s: -((1 + s**2) ** -1.5), -1),
    'inverse_quadratic': (lambda s: 1 / (1 + s**2), lambda s: -2 / (1 + s**2) ** 2, -1),
    'gaussian': (lambda s: np.exp(-(s**2)), lambda s: -2 * np.exp(-(s**2)), -1),
}

#: The declared kernels; a stranger is refused naming them.
RADIAL_KERNELS: Final = tuple(_KERNELS)


def _tail(x: np.ndarray, exponents: np.ndarray) -> np.ndarray:
    return np.prod(x[:, None, :] ** exponents[None, :, :], axis=2)


def _tail_exponents(d: int, degree: int) -> np.ndarray:
    if degree < 0:
        return np.zeros((0, d), dtype=int)
    return np.vstack([np.zeros((1, d), dtype=int), monomial_exponents(d, degree)])


@dataclass(frozen=True, slots=True)
class RadialBasis:
    """A radial basis surface: *kernel* from :data:`RADIAL_KERNELS`, shape *epsilon*, *smoothing*, tail *degree*.

    ``degree=None`` takes the kernel's minimum (0 when it has none); ``-1`` adds no tail.
    """

    kernel: str = 'thin_plate_spline'
    epsilon: float = 1.0
    smoothing: float = 0.0
    degree: int | None = None

    def __post_init__(self) -> None:
        if self.kernel not in _KERNELS:
            refuse(f'a radial kernel is one of {list(RADIAL_KERNELS)}, got {self.kernel!r}')
        if not self.epsilon > 0 or self.smoothing < 0:
            refuse(f'epsilon is > 0 and smoothing >= 0, got {self.epsilon!r} and {self.smoothing!r}')
        minimum = _KERNELS[self.kernel][2]
        if self.degree is not None and (self.degree < -1 or (minimum >= 0 and self.degree < minimum)):
            refuse(f'the {self.kernel} kernel needs a polynomial tail of degree >= {minimum}, got {self.degree}')

    @property
    def gives(self) -> frozenset[Give]:
        return frozenset({Give.GRADIENT})

    def fit(self, x, y, *, domain: BoxEnvelope | None = None) -> FittedRadialBasis:
        x, y, envelope = training_data(x, y, domain)
        phi = _KERNELS[self.kernel][0]
        degree = max(_KERNELS[self.kernel][2], 0) if self.degree is None else self.degree
        exponents = _tail_exponents(x.shape[1], degree)
        tail = _tail(x, exponents)
        if tail.shape[1] > x.shape[0]:
            refuse(f'a degree-{degree} tail has {tail.shape[1]} terms, more than the {x.shape[0]} points')
        r = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
        system = phi(self.epsilon * r) + self.smoothing * np.eye(x.shape[0])
        weights, tails = [], []
        for column in y.T:
            solution = solve_kkt(system, tail.T, column, np.zeros(tail.shape[1]))
            weights.append(solution.x)
            tails.append(solution.multipliers)
        return FittedRadialBasis(
            self.kernel, self.epsilon, x, np.column_stack(weights), exponents, np.column_stack(tails), envelope
        )


@dataclass(frozen=True, slots=True, eq=False)
class FittedRadialBasis:
    """The centres, their weights ``(n, k)``, and the tail's exponents and coefficients ``(terms, k)``."""

    kernel: str
    epsilon: float
    centres: np.ndarray
    weights: np.ndarray
    exponents: np.ndarray
    tail: np.ndarray
    envelope: BoxEnvelope

    @property
    def gives(self) -> frozenset[Give]:
        return frozenset({Give.GRADIENT})

    def predict(self, x, *, variance: bool = False, gradient: bool = False) -> Prediction:
        require_gives(self.gives, variance=variance, gradient=gradient, what='a radial basis surface')
        x = query_points(x, self.envelope)
        phi, slope, _ = _KERNELS[self.kernel]
        offset = x[:, None, :] - self.centres[None, :, :]
        s = self.epsilon * np.linalg.norm(offset, axis=2)
        mean = phi(s) @ self.weights + _tail(x, self.exponents) @ self.tail
        if not gradient:
            return Prediction(mean)
        radial = np.einsum('mn,mnd,nk->mkd', slope(s) * self.epsilon**2, offset, self.weights)
        tail = np.empty_like(radial)
        for i in range(x.shape[1]):
            lowered = self.exponents.copy()
            lowered[:, i] = np.maximum(lowered[:, i] - 1, 0)
            tail[:, :, i] = (self.exponents[:, i] * _tail(x, lowered)) @ self.tail
        return Prediction(mean, gradient=radial + tail)
