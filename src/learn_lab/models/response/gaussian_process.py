"""D46-D48 -- Gaussian process regression (Kriging): maximum-likelihood fit, closed-form LOO, mean/variance/gradient.

Each output column is its own GP on the column NORMALISED to zero mean and unit standard deviation,
with covariance

    C = amplitude R(x, x'; lengths) + noise I,     theta = log[amplitude, length_1..length_d, noise]

(``R`` a :mod:`~learn_lab.models.response.kernel` family). ``theta`` maximises the log marginal likelihood
inside ``bounds`` (each of amplitude, length and noise), from the declared start and *restarts*
seeded draws (:func:`~learn_lab.models.response.likelihood.maximise_likelihood`).

A prediction's ``variance`` is LATENT -- of the mean function, without the noise term -- and in the
values' squared units. :meth:`FittedGaussianProcess.leave_one_out` is Dubrule's closed form at the
fitted ``theta``: each point predicted from all the others, its variance that of the held-out
OBSERVATION (noise included), without n refits.

For model-based optimisation (P11) a fitted GP also gives the JOINT latent posterior
(:meth:`FittedGaussianProcess.posterior_covariance`, :meth:`~FittedGaussianProcess.posterior`), joint
draws from it under caller-held standard normals (:meth:`~FittedGaussianProcess.draw`, so a caller
fixes its base samples), its observation noise, and :meth:`~FittedGaussianProcess.condition` -- the
model with further points appended at the SAME hyperparameters and normalisation: a fantasy, never a
refit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numerics_lab.linear.dense import solve_triangular

from learn_lab.errors import refuse
from learn_lab.models.response.kernel import correlation, input_slopes, length_slopes, require_family
from learn_lab.models.response.likelihood import Box, factor, gaussian_log_likelihood, maximise_likelihood
from learn_lab.protocol import Give, Prediction, query_points, require_gives, training_data

__all__ = ['FittedGaussianProcess', 'GaussianProcess', 'GaussianProcessColumn', 'covariance', 'log_likelihood']

_GIVES = frozenset({Give.VARIANCE, Give.GRADIENT})


def covariance(family: str, theta: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``C`` at ``theta`` and its slopes ``dC/dtheta`` (``(d + 2, n, n)``)."""
    amplitude, lengths, noise = np.exp(theta[0]), np.exp(theta[1:-1]), np.exp(theta[-1])
    r = correlation(family, x, x, lengths)
    eye = np.eye(len(x))
    slopes = np.concatenate([(amplitude * r)[None], amplitude * length_slopes(family, x, lengths), (noise * eye)[None]])
    return amplitude * r + noise * eye, slopes


def log_likelihood(family: str, theta: np.ndarray, x: np.ndarray, values: np.ndarray) -> tuple[float, np.ndarray]:
    """The log marginal likelihood of NORMALISED *values* ``(n,)`` at ``theta``, and its gradient."""
    c, slopes = covariance(require_family(family), np.asarray(theta, dtype=float), x)
    return gaussian_log_likelihood(c, values, slopes)


def _normalise(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    level, scale = y.mean(axis=0), y.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    return (y - level) / scale, level, scale


@dataclass(frozen=True, slots=True)
class GaussianProcess:
    """A GP per output column: kernel *family*, *restarts* extra starts drawn from *seed*, *bounds* on each parameter."""

    family: str = 'squared_exponential'
    restarts: int = 4
    seed: int = 0
    bounds: tuple[float, float] = (1e-5, 1e5)

    def __post_init__(self) -> None:
        require_family(self.family)
        if self.restarts < 0 or not 0 < self.bounds[0] < 1 < self.bounds[1]:
            refuse(f'restarts is >= 0 and bounds bracket 1 from above 0, got {self.restarts} and {self.bounds}')

    @property
    def gives(self) -> frozenset[Give]:
        return _GIVES

    def fit(self, x, y) -> FittedGaussianProcess:
        x, y = training_data(x, y)
        normalised, level, scale = _normalise(y)
        d = x.shape[1]
        box = Box(np.full(d + 2, np.log(self.bounds[0])), np.full(d + 2, np.log(self.bounds[1])))
        start = np.log(np.r_[1.0, np.ones(d), 1e-2])
        columns = []
        for k in range(y.shape[1]):
            column = normalised[:, k]
            theta, _ = maximise_likelihood(
                lambda t, column=column: log_likelihood(self.family, t, x, column),
                start,
                box,
                restarts=self.restarts,
                seed=self.seed,
            )
            columns.append(GaussianProcessColumn.at(self.family, theta, x, column, level=level[k], scale=scale[k]))
        return FittedGaussianProcess(x, tuple(columns))


@dataclass(frozen=True, slots=True, eq=False)
class GaussianProcessColumn:
    """One fitted column: its ``theta``, the Cholesky factor of ``C``, ``alpha = C^-1 y`` and its normalisation."""

    family: str
    theta: np.ndarray
    lower: np.ndarray
    alpha: np.ndarray
    values: np.ndarray
    level: float
    scale: float
    log_likelihood: float

    @classmethod
    def at(cls, family: str, theta: np.ndarray, x: np.ndarray, values: np.ndarray, *, level: float, scale: float):
        """The column at a given ``theta``: factor, ``alpha`` and likelihood of the normalised *values*."""
        c, _ = covariance(family, theta, x)
        chol = factor(c)
        if chol is None:
            refuse(f'the {family} covariance at theta={theta} cannot be factored even with jitter')
        value, _ = log_likelihood(family, theta, x, values)
        return cls(family, theta, chol.lower, chol.solve(values), values, float(level), float(scale), value)


@dataclass(frozen=True, slots=True, eq=False)
class FittedGaussianProcess:
    """The training points and one :class:`GaussianProcessColumn` per output."""

    x: np.ndarray
    columns: tuple[GaussianProcessColumn, ...]

    @property
    def gives(self) -> frozenset[Give]:
        return _GIVES

    @property
    def hyperparameters(self) -> tuple[np.ndarray, ...]:
        """Each column's ``theta = log[amplitude, lengths..., noise]``."""
        return tuple(c.theta for c in self.columns)

    @property
    def log_likelihood(self) -> tuple[float, ...]:
        """Each column's maximised log marginal likelihood (of its normalised values)."""
        return tuple(c.log_likelihood for c in self.columns)

    def predict(self, x, *, variance: bool = False, gradient: bool = False) -> Prediction:
        require_gives(self.gives, variance=variance, gradient=gradient, what='a Gaussian process')
        q = query_points(x, self.x.shape[1])
        means, variances, slopes = [], [], []
        for c in self.columns:
            amplitude, lengths = np.exp(c.theta[0]), np.exp(c.theta[1:-1])
            k = amplitude * correlation(c.family, q, self.x, lengths)
            means.append(c.level + c.scale * (k @ c.alpha))
            if variance:
                v = solve_triangular(c.lower, k.T, lower=True)
                variances.append(np.maximum(amplitude - np.sum(v**2, axis=0), 0.0) * c.scale**2)
            if gradient:
                slopes.append(
                    c.scale * amplitude * np.einsum('mnd,n->md', input_slopes(c.family, q, self.x, lengths), c.alpha)
                )
        return Prediction(
            np.column_stack(means),
            np.column_stack(variances) if variance else None,
            np.stack(slopes, axis=1) if gradient else None,
        )

    @property
    def noise(self) -> np.ndarray:
        """Each column's fitted observation-noise variance ``(k,)``, in the values' squared units."""
        return np.array([np.exp(c.theta[-1]) * c.scale**2 for c in self.columns])

    def posterior_covariance(self, a, b) -> np.ndarray:
        """The latent posterior covariance between points *a* ``(p, d)`` and *b* ``(q, d)``: ``(k, p, q)``."""
        a, b = query_points(a, self.x.shape[1]), query_points(b, self.x.shape[1])
        blocks = []
        for c in self.columns:
            amplitude, lengths = np.exp(c.theta[0]), np.exp(c.theta[1:-1])
            va = solve_triangular(c.lower, amplitude * correlation(c.family, self.x, a, lengths), lower=True)
            vb = solve_triangular(c.lower, amplitude * correlation(c.family, self.x, b, lengths), lower=True)
            blocks.append((amplitude * correlation(c.family, a, b, lengths) - va.T @ vb) * c.scale**2)
        return np.stack(blocks)

    def posterior(self, x) -> tuple[np.ndarray, np.ndarray]:
        """The joint latent posterior at ``(m, d)`` points: mean ``(m, k)`` and covariance ``(k, m, m)``."""
        return self.predict(x).mean, self.posterior_covariance(x, x)

    def draw(self, x, base: np.ndarray) -> np.ndarray:
        """Joint posterior draws at ``(m, d)`` points under standard normals *base* ``(s, m, k)``: ``(s, m, k)``.

        The factor is numerics-lab's jittered Cholesky of each column's covariance; one *base* gives
        one set of draws, which is what keeps an acquisition built on them deterministic.
        """
        mean, cov = self.posterior(x)
        base = np.asarray(base, dtype=float)
        if base.ndim != 3 or base.shape[1:] != mean.shape:
            refuse(
                f'base normals are (samples, points, outputs) = (s, {mean.shape[0]}, {mean.shape[1]}), got {base.shape}'
            )
        draws = np.empty_like(base)
        for k, block in enumerate(cov):
            chol = factor(0.5 * (block + block.T))
            if chol is None:
                refuse('the posterior covariance cannot be factored even with jitter')
            draws[:, :, k] = mean[:, k] + base[:, :, k] @ chol.lower.T
        return draws

    def condition(self, x, y) -> FittedGaussianProcess:
        """The model with ``(x (p, d), y (p, k))`` appended, at the same hyperparameters and normalisation."""
        x, y = training_data(x, y)
        if x.shape[1] != self.x.shape[1] or y.shape[1] != len(self.columns):
            refuse(f'conditioning data is (p, {self.x.shape[1]}) points and (p, {len(self.columns)}) values')
        joined = np.vstack([self.x, x])
        columns = tuple(
            GaussianProcessColumn.at(
                c.family,
                c.theta,
                joined,
                np.concatenate([c.values, (y[:, k] - c.level) / c.scale]),
                level=c.level,
                scale=c.scale,
            )
            for k, c in enumerate(self.columns)
        )
        return FittedGaussianProcess(joined, columns)

    def leave_one_out(self) -> Prediction:
        """Each training point predicted from the others (Dubrule): mean, and the held-out observation's variance."""
        means, variances = [], []
        for c in self.columns:
            inverse = solve_triangular(
                c.lower.T, solve_triangular(c.lower, np.eye(len(self.x)), lower=True), lower=False
            )
            diagonal = np.diag(inverse)
            means.append(c.level + c.scale * (c.values - c.alpha / diagonal))
            variances.append(c.scale**2 / diagonal)
        return Prediction(np.column_stack(means), np.column_stack(variances))
