"""The maximum-likelihood machinery every Kriging model shares: one Gaussian log-likelihood and one fit.

* :func:`gaussian_log_likelihood` -- ``log N(values; 0, C)`` and its gradient
  ``0.5 tr((alpha alpha^T - C^-1) dC/dtheta_i)`` for each slope ``dC/dtheta_i``; the factorisation
  is numerics-lab's jittered Cholesky, and a matrix it cannot factor has likelihood ``-inf``.
* :class:`Box` -- ``theta = lo + (hi - lo) sigmoid(u)``, so an unconstrained ``u`` is a bounded
  ``theta``; its slope converts a gradient in ``theta`` to one in ``u``.
* :func:`maximise_likelihood` -- the multi-start fit: numerics-lab's ``minimize_lbfgs`` on ``-log L``
  in ``u``, from the declared start and *restarts* seeded draws from the box, keeping the best. A
  start that stops unconverged (line search or budget) still offers the point it reached.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numerics_lab.linear import LinearSolveError
from numerics_lab.linear.cholesky import JitteredCholesky, jittered_cholesky
from numerics_lab.minimize import minimize_lbfgs
from scipy.special import expit

__all__ = ['Box', 'factor', 'gaussian_log_likelihood', 'maximise_likelihood']

_LOG_2PI = float(np.log(2 * np.pi))

Objective = Callable[[np.ndarray], tuple[float, np.ndarray]]


def factor(covariance: np.ndarray) -> JitteredCholesky | None:
    """numerics-lab's jittered Cholesky of *covariance*, or ``None`` when no rung factors it."""
    try:
        return jittered_cholesky(covariance)
    except LinearSolveError:
        return None


def gaussian_log_likelihood(covariance: np.ndarray, values: np.ndarray, slopes: np.ndarray) -> tuple[float, np.ndarray]:
    """``log N(values; 0, covariance)`` and its gradient along each of ``slopes (p, n, n)``."""
    chol = factor(covariance)
    if chol is None:
        return -np.inf, np.full(len(slopes), np.nan)
    alpha = chol.solve(values)
    value = -0.5 * values @ alpha - np.sum(np.log(np.diag(chol.lower))) - 0.5 * values.size * _LOG_2PI
    inner = np.outer(alpha, alpha) - chol.solve(np.eye(values.size))
    return float(value), 0.5 * np.einsum('ij,pji->p', inner, slopes)


@dataclass(frozen=True, slots=True, eq=False)
class Box:
    """Bounds ``lo < theta < hi`` reached from an unconstrained ``u`` by ``lo + (hi - lo) sigmoid(u)``."""

    lo: np.ndarray
    hi: np.ndarray

    def theta(self, u: np.ndarray) -> np.ndarray:
        return self.lo + (self.hi - self.lo) * expit(u)

    def slope(self, u: np.ndarray) -> np.ndarray:
        s = expit(u)
        return (self.hi - self.lo) * s * (1 - s)

    def u(self, theta: np.ndarray) -> np.ndarray:
        p = np.clip((np.asarray(theta, dtype=float) - self.lo) / (self.hi - self.lo), 1e-12, 1 - 1e-12)
        return np.log(p / (1 - p))


def maximise_likelihood(
    log_likelihood: Objective, start: np.ndarray, box: Box, *, restarts: int, seed: int
) -> tuple[np.ndarray, float]:
    """The best ``(theta, log-likelihood)`` over *start* and *restarts* seeded uniform draws inside *box*."""

    def negative(u: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = log_likelihood(box.theta(u))
        return -value, -gradient * box.slope(u)

    rng = np.random.default_rng(seed)
    starts = [box.u(start)] + [box.u(rng.uniform(box.lo, box.hi)) for _ in range(restarts)]
    best = min(
        (minimize_lbfgs(negative, u0, rtol=0.0, atol=1e-8) for u0 in starts),
        key=lambda m: m.fun if np.isfinite(m.fun) else np.inf,
    )
    return box.theta(best.x), -best.fun
