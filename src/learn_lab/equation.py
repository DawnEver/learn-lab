"""Equation identification: which terms of a declared candidate library govern ``dx/dt``, and by how much.

SINDy's sequentially thresholded least squares: fit every rate on the whole :class:`Library`, zero each
coefficient below ``threshold`` (in the COEFFICIENT's unit: rate unit per term unit), refit on the
survivors, repeat until the support stops changing. The result names its terms, so the identified law is
read, not decoded. Rates come from the caller (measured) or from :func:`time_derivative` of sampled states.
numpy only: nothing here needs autograd, so it lives outside :mod:`learn_lab.deep`.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from learn_lab.errors import refuse, require_finite
from learn_lab.validity import BoxEnvelope

__all__ = ['IdentifiedLaw', 'Library', 'SparseRegression', 'polynomial_library', 'time_derivative']


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
            terms.append(lambda x, combo=combo: np.prod(x[:, list(combo)], axis=1))
    return Library(variables, tuple(names), tuple(terms))


def time_derivative(states: np.ndarray, dt: float) -> np.ndarray:
    """``d states / dt`` of uniformly sampled ``(T, d)`` states, second-order central (one-sided at the ends)."""
    if not dt > 0:
        refuse(f'a sampling step is > 0, got {dt}')
    return np.gradient(np.asarray(states, float), dt, axis=0, edge_order=2)


@dataclass(frozen=True, slots=True)
class SparseRegression:
    threshold: float = 0.1
    iterations: int = 20

    def __post_init__(self) -> None:
        if self.threshold < 0 or self.iterations < 1:
            refuse(f'a threshold is >= 0 and iterations >= 1, got {self.threshold}, {self.iterations}')

    def fit(self, library: Library, states: np.ndarray, rates: np.ndarray) -> IdentifiedLaw:
        """``states (n, d)`` over ``library.variables``, ``rates (n, d)``: the rate of each variable."""
        x = require_finite(np.asarray(states, float), 'the states')
        y = require_finite(np.asarray(rates, float), 'the rates')
        if y.shape != x.shape or x.shape[0] <= len(library.names):
            refuse(
                f'identification needs rates shaped like the states and more rows than terms; got {x.shape}, {y.shape}'
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
        return IdentifiedLaw(library=library, coefficients=coefficients, envelope=BoxEnvelope.of(x))


@dataclass(frozen=True, slots=True, eq=False)
class IdentifiedLaw:
    """``dx/dt = library(x) @ coefficients``; ``coefficients (terms, d)``, zero off the identified support."""

    library: Library
    coefficients: np.ndarray
    envelope: BoxEnvelope

    def terms(self) -> dict[str, dict[str, float]]:
        """Per variable, its nonzero terms by name."""
        return {
            variable: {name: float(c) for name, c in zip(self.library.names, column, strict=True) if c != 0.0}
            for variable, column in zip(self.library.variables, self.coefficients.T, strict=True)
        }

    def rate(self, states: np.ndarray) -> np.ndarray:
        """``dx/dt (n, d)`` at in-domain states; a state outside the identification data is refused."""
        return self.library.evaluate(self.envelope.require(states, 'an identified-law query')) @ self.coefficients
