"""D50 -- the intrinsic coregionalisation model (ICM): k correlated outputs, one shared correlation.

With every output measured at the same points, the stacked data ``vec(Y)`` (output-major: all of
output 1, then output 2, ...) is ``N(0, kron(B, R) + kron(diag(noise), I))``: ``R`` a kernel
correlation over the points, ``B = W W^T + diag(kappa)`` the outputs' coregionalisation matrix (``W``
is ``k x rank``), ``noise`` one variance per output. ``theta = [log l_1..l_d, W (row-major), log
kappa_1..kappa_k, log noise_1..noise_k]`` maximises the likelihood
of the NORMALISED columns; ``W`` is bounded to ``[-10, 10]`` (the columns have unit deviation).
:attr:`FittedMultiOutput.correlation` is ``B`` scaled to unit diagonal: how alike the outputs are.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numerics_lab.linear.dense import solve_triangular

from learn_lab.errors import refuse
from learn_lab.models.response.kernel import correlation, input_slopes, length_slopes, require_family
from learn_lab.models.response.likelihood import Box, factor, gaussian_log_likelihood, maximise_likelihood
from learn_lab.protocol import Give, Prediction, query_points, require_gives, training_data

__all__ = ['FittedMultiOutput', 'MultiOutputKriging']


@dataclass(frozen=True, slots=True)
class MultiOutputKriging:
    """ICM of the given *rank* over a *family* correlation, fitted from *restarts* extra seeded starts."""

    rank: int = 1
    family: str = 'squared_exponential'
    restarts: int = 2
    seed: int = 0

    def __post_init__(self) -> None:
        require_family(self.family)
        if self.rank < 1:
            refuse(f'the coregionalisation rank is >= 1, got {self.rank!r}')

    @property
    def gives(self) -> frozenset[Give]:
        return frozenset({Give.VARIANCE, Give.GRADIENT})

    def _parts(self, theta: np.ndarray, d: int, k: int):
        lengths = np.exp(theta[:d])
        w = theta[d : d + k * self.rank].reshape(k, self.rank)
        kappa = np.exp(theta[d + k * self.rank : d + k * self.rank + k])
        return lengths, w, kappa, np.exp(theta[d + k * self.rank + k :])

    def initial(self, d: int, k: int) -> np.ndarray:
        """The declared start: unit lengths, ``W = 1/sqrt(rank)`` in its first column and 0.1 after, ``kappa = 0.5``, noise 1e-2 each."""
        w = np.full((k, self.rank), 0.1)
        w[:, 0] = 1 / np.sqrt(self.rank)
        return np.concatenate([np.zeros(d), w.ravel(), np.log(np.full(k, 0.5)), np.log(np.full(k, 1e-2))])

    def covariance(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        """The covariance of ``vec(Y)`` at ``theta`` (whose size fixes the number of outputs)."""
        theta = np.asarray(theta, dtype=float)
        return self._covariance(theta, x, (theta.size - x.shape[1]) // (self.rank + 2))[0]

    def _covariance(self, theta: np.ndarray, x: np.ndarray, k: int) -> tuple[np.ndarray, list[np.ndarray]]:
        lengths, w, kappa, noise = self._parts(theta, x.shape[1], k)
        r = correlation(self.family, x, x, lengths)
        b = w @ w.T + np.diag(kappa)
        eye = np.eye(len(x))
        slopes = [np.kron(b, dr) for dr in length_slopes(self.family, x, lengths)]
        for a in range(k):
            for c in range(self.rank):
                db = np.zeros((k, k))
                db[a, :] += w[:, c]
                db[:, a] += w[:, c]
                slopes.append(np.kron(db, r))
        for a in range(k):
            unit = np.zeros((k, k))
            unit[a, a] = kappa[a]
            slopes.append(np.kron(unit, r))
        for a in range(k):
            unit = np.zeros((k, k))
            unit[a, a] = noise[a]
            slopes.append(np.kron(unit, eye))
        return np.kron(b, r) + np.kron(np.diag(noise), eye), slopes

    def log_likelihood(self, theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        """``log N(vec(y); 0, C(theta))`` (output-major) and its gradient in ``theta``."""
        c, slopes = self._covariance(np.asarray(theta, dtype=float), x, y.shape[1])
        return gaussian_log_likelihood(c, y.T.reshape(-1), np.array(slopes))

    def fit(self, x, y) -> FittedMultiOutput:
        x, y = training_data(x, y)
        _n, d = x.shape
        k = y.shape[1]
        level, scale = y.mean(axis=0), np.where(y.std(axis=0) > 0, y.std(axis=0), 1.0)
        normalised = (y - level) / scale
        size = d + k * self.rank + 2 * k
        lo, hi = np.full(size, np.log(1e-5)), np.full(size, np.log(1e5))
        lo[d : d + k * self.rank], hi[d : d + k * self.rank] = -10.0, 10.0
        theta, _ = maximise_likelihood(
            lambda t: self.log_likelihood(t, x, normalised),
            self.initial(d, k),
            Box(lo, hi),
            restarts=self.restarts,
            seed=self.seed,
        )
        c, _ = self._covariance(theta, x, k)
        chol = factor(c)
        if chol is None:
            refuse(f'the ICM covariance at theta={theta} cannot be factored')
        lengths, w, kappa, _ = self._parts(theta, d, k)
        return FittedMultiOutput(
            self.family,
            x,
            lengths,
            w @ w.T + np.diag(kappa),
            chol.lower,
            chol.solve(normalised.T.reshape(-1)),
            level,
            scale,
        )


@dataclass(frozen=True, slots=True, eq=False)
class FittedMultiOutput:
    """The fitted ICM: lengths, ``B``, the Cholesky factor, ``C^-1 vec(Y)`` and the column normalisation."""

    family: str
    x: np.ndarray
    lengths: np.ndarray
    coregionalisation: np.ndarray
    lower: np.ndarray
    alpha: np.ndarray
    level: np.ndarray
    scale: np.ndarray

    @property
    def gives(self) -> frozenset[Give]:
        return frozenset({Give.VARIANCE, Give.GRADIENT})

    @property
    def correlation(self) -> np.ndarray:
        """``B`` scaled to unit diagonal (dimensionless)."""
        s = np.sqrt(np.diag(self.coregionalisation))
        return self.coregionalisation / np.outer(s, s)

    def predict(self, x, *, variance: bool = False, gradient: bool = False) -> Prediction:
        require_gives(self.gives, variance=variance, gradient=gradient, what='an ICM model')
        q = query_points(x, self.x.shape[1])
        b = self.coregionalisation
        r = correlation(self.family, q, self.x, self.lengths)
        dr = input_slopes(self.family, q, self.x, self.lengths) if gradient else None
        means, variances, slopes = [], [], []
        for a in range(len(b)):
            cross = np.hstack([b[a, c] * r for c in range(len(b))])
            means.append(self.level[a] + self.scale[a] * cross @ self.alpha)
            if variance:
                v = solve_triangular(self.lower, cross.T, lower=True)
                variances.append(np.maximum(b[a, a] - np.sum(v**2, axis=0), 0.0) * self.scale[a] ** 2)
            if gradient:
                rows = np.concatenate([b[a, c] * dr for c in range(len(b))], axis=1)
                slopes.append(self.scale[a] * np.einsum('mnd,n->md', rows, self.alpha))
        return Prediction(
            np.column_stack(means),
            np.column_stack(variances) if variance else None,
            np.stack(slopes, axis=1) if gradient else None,
        )
