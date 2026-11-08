"""Equation identification: which terms of a declared candidate library govern a rate, and by how much.

SINDy's sequentially thresholded least squares: fit every rate on the whole :class:`Library`, zero each
coefficient below ``threshold`` (in the COEFFICIENT's unit: rate unit per term unit), refit on the
survivors, repeat until the support stops changing. The result names its terms, so the identified law is
read, not decoded. Rates come from the caller (measured) or from :func:`time_derivative` of sampled states.

A PDE is the same regression over a different state (PDE-FIND): at every grid point of ``u(t, x)`` the state is
``(u, u_x, u_xx, ...)`` by finite differences (:func:`pde_data`), the library is ``u^p`` times each derivative
(:func:`pde_library`), and only ``u`` has a rate -- the regression's ``targets``. Finite differences amplify
noise by ``dx^-order``: noisy data identifies a wrong law rather than being refused, because how smooth is
smooth enough is a property of the data, not of the library; the caller's support check is the arbiter.
numpy only: nothing here needs autograd, so it lives outside :mod:`learn_lab.deep`.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from learn_lab.errors import refuse, require_finite
from learn_lab.validity import BoxEnvelope

__all__ = [
    'IdentifiedLaw',
    'Library',
    'Monomial',
    'SparseRegression',
    'pde_data',
    'pde_library',
    'polynomial_library',
    'time_derivative',
]


@dataclass(frozen=True, slots=True)
class Monomial:
    """A candidate term as data: the product of the variables at ``factors`` (repeats are powers; none is 1).

    A term that is data persists with the law it identified (:mod:`learn_lab.artifact`); any other callable
    remains a valid term, and a law built on one is refused at save, naming it.
    """

    factors: tuple[int, ...]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.prod(x[:, list(self.factors)], axis=1)


@dataclass(frozen=True, slots=True, eq=False)
class Library:
    """Named candidate terms over named variables: ``evaluate (n, len(variables)) -> (n, len(names))``."""

    variables: tuple[str, ...]
    names: tuple[str, ...]
    terms: tuple[Callable[[np.ndarray], np.ndarray], ...]

    def __post_init__(self) -> None:
        if len(self.names) != len(self.terms) or len(set(self.names)) != len(self.names):
            refuse(f'a library names each of its terms once, got {self.names} for {len(self.terms)} terms')

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        if x.ndim != 2 or x.shape[1] != len(self.variables):
            refuse(f'a library over variables {self.variables} takes (n, {len(self.variables)}), got {x.shape}')
        return np.stack([term(x) for term in self.terms], axis=1)


def polynomial_library(variables: Sequence[str], *, degree: int) -> Library:
    """Every monomial up to *degree*, constant first, named ``'1'``, ``'x'``, ``'x^2'``, ``'x*y'``."""
    if degree < 0:
        refuse(f'a polynomial degree is >= 0, got {degree}')
    variables = tuple(variables)
    names, terms = [], []
    for d in range(degree + 1):
        for combo in itertools.combinations_with_replacement(range(len(variables)), d):
            counts = {i: combo.count(i) for i in dict.fromkeys(combo)}
            names.append('*'.join(variables[i] + (f'^{c}' if c > 1 else '') for i, c in counts.items()) or '1')
            terms.append(Monomial(combo))
    return Library(variables, tuple(names), tuple(terms))


def time_derivative(states: np.ndarray, dt: float) -> np.ndarray:
    """``d states / dt`` of uniformly sampled ``(T, d)`` states, second-order central (one-sided at the ends)."""
    if not dt > 0:
        refuse(f'a sampling step is > 0, got {dt}')
    return np.gradient(np.asarray(states, float), dt, axis=0, edge_order=2)


def _derivative_name(order: int) -> str:
    return 'u' + ('_' + 'x' * order if order else '')


def pde_library(*, order: int, degree: int) -> Library:
    """``u^p * d^k u / dx^k`` for ``k <= order``, ``p <= degree``, over the variables ``('u', 'u_x', 'u_xx', ...)``.

    Named ``'1'``, ``'u'``, ``'u^2'``, ``'u_x'``, ``'u*u_x'``, ...: derivative-major, power-minor.
    """
    if order < 1 or degree < 0:
        refuse(f'a PDE library has derivative order >= 1 and degree >= 0, got {order}, {degree}')
    variables = tuple(_derivative_name(k) for k in range(order + 1))
    names, terms = [], []
    for k in range(order + 1):
        for p in range(degree + 1):
            power = '' if p == 0 else 'u' + (f'^{p}' if p > 1 else '')
            names.append('*'.join(filter(None, [power, _derivative_name(k) if k else ''])) or '1')
            terms.append(Monomial((0,) * p + ((k,) if k else ())))
    return Library(variables, tuple(names), tuple(terms))


def pde_data(u: np.ndarray, dt: float, dx: float, *, order: int) -> tuple[np.ndarray, np.ndarray]:
    """``u (T, N)`` on a uniform grid -> ``states (n, order + 1)`` = ``(u, u_x, ...)`` and ``rates (n, 1)`` = ``u_t``.

    Derivatives are compact second-order central stencils -- the 3-point second difference, applied ``k // 2``
    times, then the central first difference if ``k`` is odd -- on interior points only: the ``order`` points
    nearest each spatial edge have no full stencil and are dropped. (Repeating a first difference instead
    would widen the stencil and quadruple the error of ``u_xx``.)
    """
    u = require_finite(np.asarray(u, float), 'the field')
    if not (dt > 0 and dx > 0) or order < 1:
        refuse(f'a PDE grid has dt > 0, dx > 0 and order >= 1, got {dt}, {dx}, {order}')
    if u.ndim != 2 or u.shape[0] < 3 or u.shape[1] < 2 * order + 3:
        refuse(f'u (T >= 3, N >= {2 * order + 3}) is needed for an order-{order} stencil, got {u.shape}')
    derivatives = []
    for k in range(order + 1):
        d = u
        for _ in range(k // 2):
            d = (d[:, 2:] - 2 * d[:, 1:-1] + d[:, :-2]) / dx**2
        if k % 2:
            d = (d[:, 2:] - d[:, :-2]) / (2 * dx)
        edge = order - (k + 1) // 2
        derivatives.append(d[:, edge : d.shape[1] - edge])
    states = np.stack(derivatives, axis=-1).reshape(-1, order + 1)
    return states, time_derivative(u, dt)[:, order : u.shape[1] - order].reshape(-1, 1)


@dataclass(frozen=True, slots=True)
class SparseRegression:
    threshold: float = 0.1
    iterations: int = 20

    def __post_init__(self) -> None:
        if self.threshold < 0 or self.iterations < 1:
            refuse(f'a threshold is >= 0 and iterations >= 1, got {self.threshold}, {self.iterations}')

    def fit(
        self, library: Library, states: np.ndarray, rates: np.ndarray, *, targets: Sequence[str] | None = None
    ) -> IdentifiedLaw:
        """``states (n, d)`` over ``library.variables``; ``rates (n, len(targets))``: the rate of each target.

        ``targets`` defaults to every variable (an ODE); a PDE names the one field whose rate is measured.
        """
        targets = library.variables if targets is None else tuple(targets)
        if not targets or not set(targets) <= set(library.variables) or len(set(targets)) != len(targets):
            refuse(f'targets are distinct variables of {library.variables}, got {targets}')
        x = require_finite(np.asarray(states, float), 'the states')
        y = require_finite(np.asarray(rates, float), 'the rates')
        if y.shape != (x.shape[0], len(targets)) or x.shape[0] <= len(library.names):
            refuse(
                f'identification needs one rate per target {targets} per state row and more rows than terms; '
                f'got {x.shape}, {y.shape}'
            )
        theta = library.evaluate(x)
        coefficients = np.linalg.lstsq(theta, y, rcond=None)[0]
        for _ in range(self.iterations):
            keep = np.abs(coefficients) >= self.threshold
            refit = np.zeros_like(coefficients)
            for j in range(y.shape[1]):
                if keep[:, j].any():
                    refit[keep[:, j], j] = np.linalg.lstsq(theta[:, keep[:, j]], y[:, j], rcond=None)[0]
            coefficients, converged = refit, np.array_equal(np.abs(refit) >= self.threshold, keep)
            if converged:
                break
        return IdentifiedLaw(library=library, targets=targets, coefficients=coefficients, envelope=BoxEnvelope.of(x))


@dataclass(frozen=True, slots=True, eq=False)
class IdentifiedLaw:
    """``d targets/dt = library(x) @ coefficients``; ``coefficients (terms, targets)``, zero off the support."""

    library: Library
    targets: tuple[str, ...]
    coefficients: np.ndarray
    envelope: BoxEnvelope

    def terms(self) -> dict[str, dict[str, float]]:
        """Per target, its nonzero terms by name."""
        return {
            variable: {name: float(c) for name, c in zip(self.library.names, column, strict=True) if c != 0.0}
            for variable, column in zip(self.targets, self.coefficients.T, strict=True)
        }

    def rate(self, states: np.ndarray) -> np.ndarray:
        """``(n, targets)`` rates at in-domain states; a state outside the identification data is refused."""
        return self.library.evaluate(self.envelope.require(states, 'an identified-law query')) @ self.coefficients
