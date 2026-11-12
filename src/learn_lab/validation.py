"""D57 -- cross-validation, regression scores and predictive calibration.

* :func:`folds` -- k test folds partitioning ``0..n-1``: contiguous (the first ``n % k`` one point
  larger) without a seed, a seeded permutation of the same sizes with one.
* :func:`kfold` -- the ``(train, test)`` split of each of those folds; :func:`region` -- the split holding out
  exactly the rows inside a declared box of the input domain (a design region the fit never sees).
* :func:`split_scores` -- per split, the model fitted on its train rows and scored on its test rows (held out)
  and on its train rows (in sample): an overfit model shows the gap.
* :func:`cross_validate` -- every point predicted by the model fitted on the other folds.
* :func:`score` -- one of :data:`SCORES` against the MEASURED values (first argument), averaged
  uniformly over the output columns: ``r2``, ``mse``, ``rmse`` (the column RMSEs averaged),
  ``mae``, ``median_absolute_error``.
* :func:`calibration` -- how far a predicted Gaussian ``N(mean, variance)`` is borne out: the
  observed coverage of each central interval, the variance of the standardised residuals (1 when
  calibrated, above when over-confident) and the mean negative log predictive density.
* :func:`scored_fit` -- a model fitted on all rows beside every one of :data:`SCORES` of its
  out-of-fold predictions; :func:`box_predictions` -- such a fit's predictions at uniform points of
  the rows' bounding box, and its ``r2``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.special import ndtri

from learn_lab.errors import refuse
from learn_lab.protocol import Fitted, Prediction, Surrogate, training_data
from learn_lab.validity import BoxEnvelope

__all__ = [
    'COVERAGE_LEVELS',
    'SCORES',
    'Calibration',
    'FoldScore',
    'ScoredFit',
    'box_predictions',
    'calibration',
    'cross_validate',
    'folds',
    'kfold',
    'region',
    'score',
    'scored_fit',
    'split_scores',
]

#: ``(train, test)`` row indices.
Split = tuple[np.ndarray, np.ndarray]


def _r2(measured: np.ndarray, predicted: np.ndarray) -> float:
    residual = np.sum((measured - predicted) ** 2, axis=0)
    spread = np.sum((measured - measured.mean(axis=0)) ** 2, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        per = np.where(spread > 0, 1 - residual / spread, np.where(residual > 0, 0.0, 1.0))
    return float(np.mean(per))


_SCORERS: Final[dict[str, Callable[[np.ndarray, np.ndarray], float]]] = {
    'r2': _r2,
    'mse': lambda m, p: float(np.mean((m - p) ** 2)),
    'rmse': lambda m, p: float(np.mean(np.sqrt(np.mean((m - p) ** 2, axis=0)))),
    'mae': lambda m, p: float(np.mean(np.abs(m - p))),
    'median_absolute_error': lambda m, p: float(np.mean(np.median(np.abs(m - p), axis=0))),
}

#: The declared scores.
SCORES: Final = tuple(_SCORERS)

#: The central-interval levels :func:`calibration` reports.
COVERAGE_LEVELS: Final = (0.5, 0.9, 0.95)


def _matrix(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[:, None] if array.ndim == 1 else array


def score(metric: str, measured: object, predicted: object) -> float:
    """*metric* of *predicted* against *measured* (in that order), averaged over the output columns."""
    if metric not in _SCORERS:
        refuse(f'a score is one of {list(SCORES)}, got {metric!r}')
    m, p = _matrix(measured), _matrix(predicted)
    if m.shape != p.shape:
        refuse(f'measured {m.shape} and predicted {p.shape} values must have one shape')
    return _SCORERS[metric](m, p)


def folds(n: int, k: int, *, seed: int | None = None) -> tuple[np.ndarray, ...]:
    """*k* test folds partitioning ``0..n-1`` (contiguous without *seed*; a seeded permutation with one)."""
    if not 2 <= k <= n:
        refuse(f'cross-validation needs 2 <= k <= n folds, got k={k} for n={n}')
    order = np.arange(n) if seed is None else np.random.default_rng(seed).permutation(n)
    sizes = np.full(k, n // k)
    sizes[: n % k] += 1
    edges = np.concatenate([[0], np.cumsum(sizes)])
    return tuple(order[edges[i] : edges[i + 1]] for i in range(k))


def kfold(n: int, k: int, *, seed: int | None = None) -> tuple[Split, ...]:
    """The ``(train, test)`` split of each of :func:`folds` ``(n, k, seed=seed)``; the tests partition ``0..n-1``."""
    everything = np.arange(n)
    return tuple((np.setdiff1d(everything, test), test) for test in folds(n, k, seed=seed))


def region(x: object, lower: object, upper: object) -> Split:
    """Hold out exactly the rows of *x* inside the closed box ``[lower, upper]``; train on every other row."""
    x = _matrix(x)
    lo, hi = (
        np.broadcast_to(np.asarray(lower, float), x.shape[1:]),
        np.broadcast_to(np.asarray(upper, float), x.shape[1:]),
    )
    if np.any(lo > hi):
        refuse(f'a held-out region runs lower <= upper per input, got {lo} .. {hi}')
    inside = np.all((x >= lo) & (x <= hi), axis=1)
    if inside.all() or not inside.any():
        refuse(f'a held-out region must hold some rows and leave some: it holds {int(inside.sum())} of {len(x)}')
    return np.flatnonzero(~inside), np.flatnonzero(inside)


@dataclass(frozen=True, slots=True)
class FoldScore:
    """One split: its rows, the score on its held-out rows and on the rows it was fitted on."""

    train: np.ndarray
    test: np.ndarray
    held_out: float
    in_sample: float


def split_scores(
    surrogate: Surrogate,
    x: object,
    y: object,
    splits: tuple[Split, ...] | list[Split],
    *,
    metric: str = 'rmse',
    domain: BoxEnvelope | None = None,
) -> tuple[FoldScore, ...]:
    """Per split, *surrogate* fitted on its train rows and scored by *metric* held out and in sample.

    Every split is fitted in ONE envelope -- *domain*, else the box all rows span -- so a held-out region at the
    edge of the data is scored, never refused as out of its fold's domain.
    """
    if metric not in _SCORERS:
        refuse(f'a score is one of {list(SCORES)}, got {metric!r}')
    x, y, envelope = training_data(x, y, domain)
    scores = []
    for train, test in splits:
        if len(train) == 0 or len(test) == 0 or len(np.intersect1d(train, test)):
            refuse('a split trains and tests on disjoint, non-empty row sets')
        fitted = surrogate.fit(x[train], y[train], domain=envelope)
        held_out = score(metric, y[test], fitted.predict(x[test]).mean)
        in_sample = score(metric, y[train], fitted.predict(x[train]).mean)
        scores.append(FoldScore(train, test, held_out, in_sample))
    return tuple(scores)


def cross_validate(
    surrogate: Surrogate,
    x: object,
    y: object,
    *,
    k: int = 5,
    seed: int | None = None,
    domain: BoxEnvelope | None = None,
) -> Prediction:
    """Each point's prediction by *surrogate* fitted on the other folds (with the variance when it gives one).

    Every fold is fitted in ONE envelope -- *domain*, else the box all rows span -- so a held-out row is
    never out of its fold's domain merely because it was the extreme one.
    """
    x, y, envelope = training_data(x, y, domain)
    want = 'variance' in surrogate.gives
    mean = np.empty_like(y)
    var = np.empty_like(y) if want else None
    for train, test in kfold(len(x), k, seed=seed):
        p = surrogate.fit(x[train], y[train], domain=envelope).predict(x[test], variance=want)
        mean[test] = p.mean
        if want:
            var[test] = p.variance
    return Prediction(mean, var)


@dataclass(frozen=True, slots=True)
class Calibration:
    """Observed coverage per nominal central level, standardised-residual variance, mean NLPD."""

    coverage: dict[float, float]
    standardised_variance: float
    negative_log_density: float


def calibration(measured: object, prediction: Prediction) -> Calibration:
    """How well *prediction*'s ``N(mean, variance)`` describes *measured* (see the module docstring)."""
    if prediction.variance is None:
        refuse('calibration reads a predicted variance; this prediction carries none')
    m = _matrix(measured)
    if m.shape != prediction.mean.shape:
        refuse(f'measured {m.shape} and predicted {prediction.mean.shape} values must have one shape')
    if np.any(prediction.variance <= 0):
        refuse('calibration needs a variance > 0 at every point')
    z = (m - prediction.mean) / np.sqrt(prediction.variance)
    coverage = {level: float(np.mean(np.abs(z) <= ndtri(0.5 + level / 2))) for level in COVERAGE_LEVELS}
    density = 0.5 * np.log(2 * np.pi * prediction.variance) + 0.5 * z**2
    return Calibration(coverage, float(np.mean(z**2)), float(np.mean(density)))


@dataclass(frozen=True, slots=True)
class ScoredFit:
    """A model fitted on every row, and each of :data:`SCORES` of its out-of-fold predictions."""

    scores: dict[str, float]
    fitted: Fitted


def scored_fit(
    surrogate: Surrogate, x: object, y: object, *, seed: int = 3, domain: BoxEnvelope | None = None
) -> ScoredFit:
    """*surrogate* fitted on ``(x, y)`` in *domain*, scored by ``min(5, n)``-fold cross-validation split by *seed*."""
    x, y, envelope = training_data(x, y, domain)
    predicted = cross_validate(surrogate, x, y, k=min(5, len(x)), seed=seed, domain=envelope).mean
    return ScoredFit({metric: score(metric, y, predicted) for metric in SCORES}, surrogate.fit(x, y, domain=envelope))


def box_predictions(surrogate: Surrogate, x: object, y: object, *, samples: int, seed: int) -> tuple[np.ndarray, float]:
    """Predicted means at *samples* uniform points of the rows' box (``default_rng(seed).uniform``), and the
    fit's cross-validated ``r2``.
    """
    fit = scored_fit(surrogate, x, y)
    x = np.asarray(x, dtype=float)
    sample = np.random.default_rng(seed).uniform(x.min(axis=0), x.max(axis=0), (samples, x.shape[1]))
    return fit.fitted.predict(sample).mean, float(fit.scores['r2'])
