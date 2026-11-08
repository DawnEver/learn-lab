"""A run's surrogate generations: refit on each generation's rows, scored and sampled.

A run is its complete rows -- a design point and one measured value per objective -- plus the
cumulative row count closing each generation (:class:`RunHistory`). Per objective, a registry model
is fitted on inputs scaled to the rows' unit box and values standardised, so one key behaves the same
whatever the units. Its error is the cross-validated MAE, relative to the objective's mean magnitude.
A PREDICTION IS NEVER A ROW: the predictions returned are arrays, never measurements.
"""

import itertools
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from learn_lab.errors import refuse
from learn_lab.registry import surrogate
from learn_lab.validation import ScoredFit, scored_fit
from learn_lab.validity import BoxEnvelope

__all__ = ['RunHistory', 'generation_errors', 'generation_predictions']


@dataclass(frozen=True)
class RunHistory:
    """Complete rows of a run and the cumulative row count closing each generation."""

    axes: tuple[str, ...]
    x: np.ndarray
    objectives: Mapping[str, np.ndarray]
    generation_rows: tuple[int, ...]

    def __post_init__(self) -> None:
        n = len(self.x)
        if self.x.ndim != 2 or self.x.shape[1] != len(self.axes):
            refuse(f'x must be (rows, {len(self.axes)}) for axes {self.axes}, got {self.x.shape}')
        bad = [k for k, v in self.objectives.items() if np.shape(v) != (n,)]
        if bad:
            refuse(f'objective(s) {bad} do not carry one value per row ({n})')
        cuts = self.generation_rows
        if not cuts or any(b <= a for a, b in itertools.pairwise(cuts)) or cuts[0] < 1 or cuts[-1] > n:
            refuse(f'generation cuts {cuts} must rise strictly within 1..{n} complete rows')
        if cuts[0] < 2:
            refuse(f'the first generation closes at {cuts[0]} row(s); a cross-validated refit needs at least 2')

    def row_generation(self) -> np.ndarray:
        """The generation index that evaluated each row: row i is in the first k with i < cuts[k]."""
        return np.searchsorted(np.asarray(self.generation_rows), np.arange(len(self.x)), side='right')


def _settings(key: str) -> dict:
    """The GP learns one length per axis and a noise variance from normalized data, from a fixed seed."""
    return {'restarts': 3, 'seed': 0} if key == 'gaussian_process' else {}


def _fit(key: str, x: np.ndarray, y: np.ndarray, lo: np.ndarray, span: np.ndarray) -> tuple[ScoredFit, float, float]:
    """A generation's fit on the rows' box mapped to the unit cube -- its declared domain, which every later
    generation's samples lie in.
    """
    mu, sd = float(y.mean()), float(y.std()) or 1.0
    cube = BoxEnvelope(np.zeros(x.shape[1]), np.ones(x.shape[1]))
    fit = scored_fit(surrogate(key, **_settings(key)), (x - lo) / span, ((y - mu) / sd)[:, np.newaxis], domain=cube)
    return fit, mu, sd


def _box(history: RunHistory) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = history.x.min(axis=0), history.x.max(axis=0)
    return lo, np.where(hi > lo, hi - lo, 1.0)


def generation_errors(history: RunHistory, *, key: str = 'gaussian_process') -> dict[str, list[float]]:
    """Per objective, the cross-validated MAE in percent of mean |value|, one entry per generation."""
    lo, span = _box(history)
    out: dict[str, list[float]] = {}
    for name, values in history.objectives.items():
        series = []
        for n in history.generation_rows:
            y = values[:n]
            fit, _, sd = _fit(key, history.x[:n], y, lo, span)
            series.append(100.0 * float(fit.scores['mae']) * sd / (float(np.mean(np.abs(y))) or 1.0))
        out[name] = series
    return out


def generation_predictions(
    history: RunHistory, *, key: str = 'gaussian_process', samples: int = 5000, seed: int
) -> tuple[list[np.ndarray], list[dict[str, np.ndarray]]]:
    """Per generation, its own uniform samples over the rows' box (seed ``seed + k``) and the predictions there.

    A FRESH sample set per generation, so generations plotted together interleave rather than stack.
    """
    lo, span = _box(history)
    points, predictions = [], []
    for k, n in enumerate(history.generation_rows):
        here = lo + span * np.random.default_rng(seed + k).random((samples, len(history.axes)))
        generation = {}
        for name, values in history.objectives.items():
            fit, mu, sd = _fit(key, history.x[:n], values[:n], lo, span)
            generation[name] = fit.fitted.predict((here - lo) / span).mean[:, 0] * sd + mu
        points.append(here)
        predictions.append(generation)
    return points, predictions
