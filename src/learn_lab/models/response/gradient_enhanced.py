"""D49 -- gradient-enhanced Kriging: one GP conditioned on values AND gradients.

With the squared-exponential covariance ``k(a, b) = amplitude exp(-sum_j (a_j - b_j)^2 / (2 l_j^2))``
the joint covariance of a value and the derivatives is ``k`` differentiated in each argument:

    cov(f(a), df(b)/db_j)          =  k D_j / l_j^2              (D = a - b)
    cov(df(a)/da_i, df(b)/db_j)    =  k (delta_ij / l_i^2 - D_i D_j / (l_i^2 l_j^2))

The data vector stacks the n values, then the n x d gradients point by point. ``theta = log[
amplitude, l_1..l_d]`` maximises the likelihood of the NORMALISED data (values to zero mean and unit
deviation, gradients divided by the same deviation); a nugget of ``1e-8 amplitude`` on the diagonal
keeps the notoriously ill-conditioned matrix factorable. Predictions give the mean, its gradient
(itself a cross-covariance prediction) and the latent variance of the value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numerics_lab.linear.dense import solve_triangular

from learn_lab.errors import refuse
from learn_lab.models.response.likelihood import Box, factor, gaussian_log_likelihood, maximise_likelihood
from learn_lab.protocol import Give, Prediction, query_points, require_gives, training_data

__all__ = ['FittedGradientEnhanced', 'GradientEnhancedKriging']

#: The nugget, relative to the amplitude (dimensionless).
_NUGGET: Final = 1e-8


def _blocks(theta: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, ...]:
    """``ff (na, nb)``, ``fg (na, nb, d)``, ``gf (na, d, nb)``, ``gg (na, d, nb, d)`` and the pieces they share."""
    amplitude, lengths = np.exp(theta[0]), np.exp(theta[1:])
    inv = 1 / lengths**2
    diff = a[:, None, :] - b[None, :, :]
    k = amplitude * np.exp(-0.5 * np.sum(diff**2 * inv, axis=2))
    scaled = diff * inv
    fg = k[:, :, None] * scaled
    gf = -np.moveaxis(fg, 2, 1)
    gg = k[:, None, :, None] * (
        np.eye(len(lengths))[None, :, None, :] * inv[None, :, None, None] - np.einsum('abi,abj->aibj', scaled, scaled)
    )
    return k, fg, gf, gg, diff, inv


def _assemble(ff: np.ndarray, fg: np.ndarray, gf: np.ndarray, gg: np.ndarray) -> np.ndarray:
    na, nb, d = fg.shape
    return np.block([[ff, fg.reshape(na, nb * d)], [gf.reshape(na * d, nb), gg.reshape(na * d, nb * d)]])


@dataclass(frozen=True, slots=True)
class GradientEnhancedKriging:
    """Gradient-enhanced Kriging (squared-exponential), fitted from *restarts* extra seeded starts within *bounds*."""

    restarts: int = 2
    seed: int = 0
    bounds: tuple[float, float] = (1e-5, 1e5)

    @property
    def gives(self) -> frozenset[Give]:
        return frozenset({Give.VARIANCE, Give.GRADIENT})

    def covariance(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        """The joint covariance of the values and gradients at *x*, nugget included."""
        ff, fg, gf, gg, _, _ = _blocks(np.asarray(theta, dtype=float), x, x)
        c = _assemble(ff, fg, gf, gg)
        return c + _NUGGET * np.exp(theta[0]) * np.eye(len(c))

    def log_likelihood(self, theta: np.ndarray, x: np.ndarray, values: np.ndarray, gradients: np.ndarray):
        """``log N([values; gradients]; 0, C(theta))`` and its gradient in ``theta``."""
        theta = np.asarray(theta, dtype=float)
        ff, fg, gf, gg, diff, inv = _blocks(theta, x, x)
        n, d = x.shape
        eye = np.eye(d)
        slopes = [_assemble(ff, fg, gf, gg) + _NUGGET * np.exp(theta[0]) * np.eye(n * (1 + d))]
        k = ff
        scaled = diff * inv
        for m in range(d):
            q = diff[:, :, m] ** 2 * inv[m]
            dfg = q[:, :, None] * fg - 2 * eye[m][None, None, :] * fg
            dgf = -np.moveaxis(dfg, 2, 1)
            dgg = q[:, None, :, None] * gg + k[:, None, :, None] * (
                -2 * (eye[m][:, None] * eye)[None, :, None, :] * inv[None, :, None, None]
                + 2
                * (eye[m][None, :, None, None] + eye[m][None, None, None, :])
                * np.einsum('abi,abj->aibj', scaled, scaled)
            )
            slopes.append(_assemble(k * q, dfg, dgf, dgg))
        stacked = np.concatenate([values, gradients.reshape(-1)])
        return gaussian_log_likelihood(slopes[0], stacked, np.array(slopes))

    def fit(self, x, y, gradients) -> FittedGradientEnhanced:
        x, y = training_data(x, y)
        g = np.asarray(gradients, dtype=float)
        if g.shape != (*y.shape, x.shape[1]):
            refuse(f'gradients are (points, outputs, inputs) = {(*y.shape, x.shape[1])}, got {g.shape}')
        d = x.shape[1]
        box = Box(np.full(d + 1, np.log(self.bounds[0])), np.full(d + 1, np.log(self.bounds[1])))
        level, scale = y.mean(axis=0), np.where(y.std(axis=0) > 0, y.std(axis=0), 1.0)
        columns = []
        for c in range(y.shape[1]):
            values, slopes = (y[:, c] - level[c]) / scale[c], g[:, c, :] / scale[c]
            theta, _ = maximise_likelihood(
                lambda t, v=values, s=slopes: self.log_likelihood(t, x, v, s),
                np.zeros(d + 1),
                box,
                restarts=self.restarts,
                seed=self.seed,
            )
            chol = factor(self.covariance(theta, x))
            if chol is None:
                refuse(f'the gradient-enhanced covariance at theta={theta} cannot be factored')
            stacked = np.concatenate([values, slopes.reshape(-1)])
            columns.append((theta, chol.lower, chol.solve(stacked), float(level[c]), float(scale[c])))
        return FittedGradientEnhanced(x, tuple(columns))


@dataclass(frozen=True, slots=True, eq=False)
class FittedGradientEnhanced:
    """Per output: ``(theta, Cholesky factor, C^-1 data, level, scale)``."""

    x: np.ndarray
    columns: tuple[tuple[np.ndarray, np.ndarray, np.ndarray, float, float], ...]

    @property
    def gives(self) -> frozenset[Give]:
        return frozenset({Give.VARIANCE, Give.GRADIENT})

    def predict(self, x, *, variance: bool = False, gradient: bool = False) -> Prediction:
        require_gives(self.gives, variance=variance, gradient=gradient, what='gradient-enhanced Kriging')
        q = query_points(x, self.x.shape[1])
        m, d = q.shape
        means, variances, slopes = [], [], []
        for theta, lower, alpha, level, scale in self.columns:
            ff, fg, gf, gg, _, _ = _blocks(theta, q, self.x)
            cross = np.hstack([ff, fg.reshape(m, -1)])
            means.append(level + scale * cross @ alpha)
            if variance:
                v = solve_triangular(lower, cross.T, lower=True)
                variances.append(np.maximum(np.exp(theta[0]) - np.sum(v**2, axis=0), 0.0) * scale**2)
            if gradient:
                rows = np.concatenate([gf.reshape(m, d, -1), gg.reshape(m, d, -1)], axis=2)
                slopes.append(scale * rows @ alpha)
        return Prediction(
            np.column_stack(means),
            np.column_stack(variances) if variance else None,
            np.stack(slopes, axis=1) if gradient else None,
        )
