"""D51 -- autoregressive (AR1) co-Kriging over fidelity levels 0 (cheapest) .. L-1 (the truth).

Level 0 is a :class:`~learn_lab.models.response.gaussian_process.GaussianProcess`. Each level above it is

    y_t(x) = scale_t * mu_{t-1}(x) + offset_t + delta_t(x),     delta_t ~ GP(0, amplitude R + noise I)

fitted recursively (Le Gratiet): ``mu_{t-1}`` is the level below's predictive mean at level t's
points, ``(scale_t, offset_t)`` its generalised-least-squares estimate at each candidate kernel
``theta``, and ``theta`` maximises the resulting likelihood (by the envelope theorem its gradient is
the likelihood gradient at the GLS coefficients). The designs need not be nested. A prediction
chains the levels: ``mu_t = scale_t mu_{t-1} + offset_t + delta_t``, ``var_t = scale_t^2 var_{t-1}
+ var(delta_t)``, and the gradient likewise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from learn_lab.errors import refuse
from learn_lab.models.response.gaussian_process import (
    FittedGaussianProcess,
    GaussianProcess,
    GaussianProcessColumn,
    covariance,
)
from learn_lab.models.response.kernel import require_family
from learn_lab.models.response.likelihood import Box, factor, gaussian_log_likelihood, maximise_likelihood
from learn_lab.protocol import Give, Prediction, query_points, require_gives, training_data

__all__ = ['CoKriging', 'FittedCoKriging']


@dataclass(frozen=True, slots=True, eq=False)
class _Step:
    scale: float
    offset: float
    discrepancy: FittedGaussianProcess


def _gls(c: np.ndarray, trend: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    chol = factor(c)
    if chol is None:
        return None
    ct = chol.solve(trend)
    coefficients = np.linalg.solve(trend.T @ ct, ct.T @ values)
    return coefficients, values - trend @ coefficients


@dataclass(frozen=True, slots=True)
class CoKriging:
    """AR1 co-Kriging with a *family* kernel at every level, *restarts* extra seeded starts per fit."""

    family: str = 'squared_exponential'
    restarts: int = 4
    seed: int = 0
    bounds: tuple[float, float] = (1e-5, 1e5)

    def __post_init__(self) -> None:
        require_family(self.family)

    @property
    def gives(self) -> frozenset[Give]:
        return frozenset({Give.VARIANCE, Give.GRADIENT})

    def fit(self, x, y, level) -> FittedCoKriging:
        x, y = training_data(x, y)
        level = np.asarray(level)
        if level.shape != (len(x),) or level.min() != 0:
            refuse(f'level is one integer per point, from 0 (cheapest), got shape {level.shape}')
        top = int(level.max())
        missing = [t for t in range(top + 1) if not np.any(level == t)]
        if missing:
            refuse(f'every fidelity level from 0 to {top} needs data; level(s) {missing} have none')
        base = GaussianProcess(self.family, self.restarts, self.seed, self.bounds).fit(x[level == 0], y[level == 0])
        steps: list[tuple[_Step, ...]] = []
        for t in range(1, top + 1):
            xt, yt = x[level == t], y[level == t]
            if len(xt) < 3:
                refuse(f'level {t} has {len(xt)} point(s); a scale, an offset and a discrepancy need at least 3')
            below = FittedCoKriging(base, tuple(steps)).predict(xt).mean
            steps.append(tuple(self._step(xt, yt[:, c], below[:, c]) for c in range(y.shape[1])))
        return FittedCoKriging(base, tuple(steps))

    def _step(self, x: np.ndarray, values: np.ndarray, below: np.ndarray) -> _Step:
        spread = float(np.std(values)) or 1.0
        v, trend = values / spread, np.column_stack([below / spread, np.ones(len(x))])

        def likelihood(theta: np.ndarray) -> tuple[float, np.ndarray]:
            c, slopes = covariance(self.family, theta, x)
            fitted = _gls(c, trend, v)
            if fitted is None:
                return -np.inf, np.full(theta.size, np.nan)
            return gaussian_log_likelihood(c, fitted[1], slopes)

        d = x.shape[1]
        box = Box(np.full(d + 2, np.log(self.bounds[0])), np.full(d + 2, np.log(self.bounds[1])))
        theta, _ = maximise_likelihood(
            likelihood, np.log(np.r_[1.0, np.ones(d), 1e-2]), box, restarts=self.restarts, seed=self.seed
        )
        coefficients, residual = _gls(covariance(self.family, theta, x)[0], trend, v)
        column = GaussianProcessColumn.at(self.family, theta, x, residual, level=0.0, scale=spread)
        return _Step(float(coefficients[0]), float(coefficients[1] * spread), FittedGaussianProcess(x, (column,)))


@dataclass(frozen=True, slots=True, eq=False)
class FittedCoKriging:
    """The level-0 GP and, per level above it, one ``(scale, offset, discrepancy)`` step per output."""

    base: FittedGaussianProcess
    steps: tuple[tuple[_Step, ...], ...]

    @property
    def gives(self) -> frozenset[Give]:
        return frozenset({Give.VARIANCE, Give.GRADIENT})

    @property
    def scales(self) -> np.ndarray:
        """``scale_t`` per level above 0 and output, ``(L - 1, k)`` (dimensionless ratio of the levels)."""
        return np.array([[s.scale for s in level] for level in self.steps])

    def predict(self, x, *, variance: bool = False, gradient: bool = False) -> Prediction:
        require_gives(self.gives, variance=variance, gradient=gradient, what='co-Kriging')
        q = query_points(x, self.base.x.shape[1])
        p = self.base.predict(q, variance=variance, gradient=gradient)
        mean = p.mean.copy()
        var = None if p.variance is None else p.variance.copy()
        slope = None if p.gradient is None else p.gradient.copy()
        for level in self.steps:
            for c, step in enumerate(level):
                delta = step.discrepancy.predict(q, variance=variance, gradient=gradient)
                mean[:, c] = step.scale * mean[:, c] + step.offset + delta.mean[:, 0]
                if variance:
                    var[:, c] = step.scale**2 * var[:, c] + delta.variance[:, 0]
                if gradient:
                    slope[:, c] = step.scale * slope[:, c] + delta.gradient[:, 0]
        return Prediction(mean, var, slope)
