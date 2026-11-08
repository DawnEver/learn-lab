"""Hyper-parameter tuning: every combination of candidate settings, judged by cross-validation.

:func:`tune` builds the surrogate *name* (:data:`~learn_lab.registry.SURROGATES`) at every point of the
Cartesian grid of *candidates*, scores each by the k-fold out-of-fold predictions of
:func:`~learn_lab.validation.cross_validate` under one of :data:`~learn_lab.validation.SCORES`, and returns
every :class:`Trial` beside the best. ``r2`` is maximised, every error score minimised. The folds are the same
for every trial (one *seed*) and every fold is fitted in ONE envelope (*domain*, else the rows' span), so
trials differ only in their settings. Ties keep the first in grid order.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from learn_lab.errors import refuse
from learn_lab.protocol import Surrogate, training_data
from learn_lab.registry import surrogate
from learn_lab.validation import SCORES, cross_validate, score
from learn_lab.validity import BoxEnvelope

__all__ = ['Trial', 'Tuning', 'tune']


@dataclass(frozen=True, slots=True)
class Trial:
    """One combination of *settings* and its cross-validated *score*."""

    settings: dict[str, object]
    score: float


@dataclass(frozen=True, slots=True)
class Tuning:
    """The best :class:`Trial`'s declaration and every trial in grid order, under *metric*."""

    name: str
    metric: str
    best: Trial
    surrogate: Surrogate
    trials: tuple[Trial, ...]


def tune(
    name: str,
    candidates: Mapping[str, Sequence[object]],
    x: object,
    y: object,
    *,
    metric: str = 'r2',
    k: int = 5,
    seed: int | None = None,
    domain: BoxEnvelope | None = None,
) -> Tuning:
    """The settings of surrogate *name* from the grid of *candidates* that score best in *k*-fold validation."""
    if metric not in SCORES:
        refuse(f'a score is one of {list(SCORES)}, got {metric!r}')
    empty = [key for key, values in candidates.items() if len(values) == 0]
    if empty:
        refuse(f'every tuned setting needs at least one candidate; {empty} have none')
    x, y, envelope = training_data(x, y, domain)
    keys = tuple(candidates)
    trials = []
    for values in itertools.product(*(candidates[key] for key in keys)):
        settings = dict(zip(keys, values, strict=True))
        predicted = cross_validate(surrogate(name, **settings), x, y, k=k, seed=seed, domain=envelope).mean
        trials.append(Trial(settings, score(metric, y, predicted)))
    scores = np.array([t.score for t in trials])
    if not np.isfinite(scores).any():
        refuse(f'no candidate of {name!r} scored a finite {metric}; there is nothing to choose')
    sign = -1.0 if metric == 'r2' else 1.0
    best = trials[int(np.nanargmin(sign * scores))]
    return Tuning(name, metric, best, surrogate(name, **best.settings), tuple(trials))
