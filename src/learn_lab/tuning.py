"""How a setting of a model family scores: an :class:`Objective` to MINIMISE, for an optimiser to search.

learn-lab states WHAT is tuned -- every family's default :class:`~learn_lab.space.Space`, beside its
constructor in :mod:`learn_lab.registry` -- and how one point of it scores; the search is optimi-lab's.

:func:`objective` fixes the data, the folds and the envelope once, so every call differs only in its
settings. A call builds the family at *fixed* plus the given settings (unnamed ones keep the constructor's
default), takes the *k*-fold out-of-fold predictions of :func:`~learn_lab.validation.cross_validate` and
scores them by one of :data:`~learn_lab.validation.SCORES`: an error score as it stands, ``r2`` negated. A
setting outside the family's space is refused before any fit; a fit that fails numerically (refused, a
singular solve, a non-finite loss) scores ``inf`` and its reason is kept. Every call is recorded in
:attr:`Objective.evaluations`.

A deep family (``[torch]``) takes sample-indexed arrays -- ``x (n, ...)``, ``y (n, ...)`` -- folded along
the first axis, and may take a :class:`~learn_lab.deep.fit.Checkpoint`: each setting checkpoints under its
own subdirectory (named by a digest of the settings, so a rerun resumes whatever order it searches in), each
fold under ``member-<fold>``.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np

from learn_lab.errors import Refused, refuse
from learn_lab.protocol import training_data
from learn_lab.registry import family
from learn_lab.space import Choice, Family, Integer, Real, Space
from learn_lab.validation import SCORES, cross_validate, score
from learn_lab.validity import BoxEnvelope

__all__ = ['Choice', 'Evaluation', 'Integer', 'Objective', 'Real', 'Space', 'objective']

#: What a numerically failing fit raises; anything else is a defect and propagates.
_NUMERICAL: tuple[type[Exception], ...] = (Refused, np.linalg.LinAlgError, FloatingPointError, ArithmeticError)


@dataclass(frozen=True, slots=True)
class Evaluation:
    """One call: its *settings*, the *value* returned, and why the fit failed when it is ``inf``."""

    settings: Mapping[str, object]
    value: float
    failure: str | None = None


class Objective:
    """``objective(settings) -> float`` to minimise, for family *name* on fixed data and folds."""

    def __init__(
        self,
        name: str,
        row: Family,
        data: tuple[np.ndarray, np.ndarray, BoxEnvelope, tuple[tuple[int, ...], tuple[int, ...]]],
        *,
        metric: str,
        k: int,
        seed: int | None,
        fixed: Mapping[str, object],
        checkpoint: object | None,
    ) -> None:
        self.name, self.metric, self.k, self.seed = name, metric, k, seed
        self.fixed = MappingProxyType(dict(fixed))
        self._row, self._data, self._checkpoint = row, data, checkpoint
        self._evaluations: list[Evaluation] = []

    @property
    def space(self) -> Space:
        """The settings this objective tunes: the family's declared space."""
        return self._row.space

    @property
    def evaluations(self) -> tuple[Evaluation, ...]:
        """Every call evaluated so far, in order."""
        return tuple(self._evaluations)

    def build(self, settings: Mapping[str, object]) -> object:
        """The declaration a call with *settings* fits (refused outside the space)."""
        problems = self.space.problems(settings)
        if problems:
            refuse(f'{self.name} is tuned within its declared space: ' + '; '.join(problems))
        if not self._row.shaped:
            return self._row.build(**self.fixed, **settings)
        checkpoint = self._checkpoint
        if checkpoint is not None:
            digest = hashlib.sha256(repr(sorted(settings.items())).encode()).hexdigest()[:16]
            checkpoint = replace(checkpoint, directory=checkpoint.directory / digest)
        return self._row.build(shapes=self._data[3], checkpoint=checkpoint, **self.fixed, **settings)

    def __call__(self, settings: Mapping[str, object]) -> float:
        settings = dict(settings)
        declaration = self.build(settings)
        x, y, envelope, _ = self._data
        failure = None
        try:
            with np.errstate(all='raise'):
                predicted = cross_validate(declaration, x, y, k=self.k, seed=self.seed, domain=envelope).mean
            value = score(self.metric, y, predicted)
            value = -value if self.metric == 'r2' else value
            if not math.isfinite(value):
                failure, value = f'the {self.metric} is {value}', math.inf
        except _NUMERICAL as exc:
            failure, value = f'{type(exc).__name__}: {exc}', math.inf
        self._evaluations.append(Evaluation(MappingProxyType(settings), value, failure))
        return value


def objective(
    name: str,
    x: object,
    y: object,
    *,
    metric: str = 'r2',
    k: int = 5,
    seed: int | None = None,
    domain: BoxEnvelope | None = None,
    fixed: Mapping[str, object] | None = None,
    checkpoint: object | None = None,
) -> Objective:
    """The :class:`Objective` of family *name* on ``(x, y)``: *k* folds split by *seed*, scored by *metric*."""
    if metric not in SCORES:
        refuse(f'a score is one of {list(SCORES)}, got {metric!r}')
    row = family(name)
    fixed = dict(fixed or {})
    tuned = sorted(set(fixed) & set(row.space))
    if tuned:
        refuse(f'{tuned} are tuned settings of {name}; a fixed setting is held, not tuned')
    if checkpoint is not None and not row.shaped:
        refuse(f'{name} is not a deep family; only a deep fit takes a checkpoint')
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim == 0 or y.ndim == 0:
        refuse(f'an objective takes one sample per row of x and y, got {x.shape} and {y.shape}')
    shapes = (x.shape[1:], y.shape[1:])
    if row.shaped:
        x, y = x.reshape(len(x), -1), y.reshape(len(y), -1)
    x, y, envelope = training_data(x, y, domain)
    return Objective(
        name, row, (x, y, envelope, shapes), metric=metric, k=k, seed=seed, fixed=fixed, checkpoint=checkpoint
    )
