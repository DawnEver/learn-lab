"""A response model is ``fit -> Fitted``; a fitted model is ``predict -> Prediction``.

This is the L6 protocol (D42) whose owner is learn-lab: optimi-lab's ``optimi_lab.model.protocol`` moves
here in A1 and is deleted there, so there is one definition. A :class:`Prediction` is the mean ``(n, k)``
and, when asked for and given, the variance ``(n, k)`` (LATENT: of the mean function) and the gradient of
the mean ``(n, k, d)``. A model declares what it ``gives`` beyond the mean; asking for anything else is
refused naming what it does give.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np

from learn_lab.errors import refuse, require_finite

__all__ = ['Fitted', 'Give', 'Prediction', 'Surrogate', 'require_gives', 'training_data']


class Give(StrEnum):
    """What a fitted model can return beside its mean."""

    VARIANCE = 'variance'
    GRADIENT = 'gradient'


def _frozen(array: np.ndarray) -> np.ndarray:
    array = np.array(array, dtype=float)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True, eq=False)
class _PredictionFields:
    mean: np.ndarray
    variance: np.ndarray | None
    gradient: np.ndarray | None


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Prediction(_PredictionFields):
    """``mean (n, k)``; ``variance (n, k) >= 0`` or ``None``; ``gradient (n, k, d)`` of the mean or ``None``."""

    def __init__(self, mean, variance=None, gradient=None) -> None:
        mean = _frozen(mean)
        if mean.ndim != 2:
            refuse(f'a predicted mean is (points, outputs), got shape {mean.shape}')
        if variance is not None:
            variance = _frozen(variance)
            if variance.shape != mean.shape:
                refuse(f'the variance has shape {variance.shape}, the mean {mean.shape}')
            if np.any(variance < 0):
                refuse('a variance is >= 0; a negative one is an arithmetic defect, never a prediction')
        if gradient is not None:
            gradient = _frozen(gradient)
            if gradient.ndim != 3 or gradient.shape[:2] != mean.shape:
                refuse(f'the gradient is (points, outputs, inputs) = {mean.shape} + (d,), got {gradient.shape}')
        _PredictionFields.__init__(self, mean, variance, gradient)


class Fitted(Protocol):
    @property
    def gives(self) -> frozenset[Give]: ...

    def predict(self, x: np.ndarray, *, variance: bool = False, gradient: bool = False) -> Prediction: ...


class Surrogate(Protocol):
    @property
    def gives(self) -> frozenset[Give]: ...

    def fit(self, x: np.ndarray, y: np.ndarray) -> Fitted: ...


def training_data(x: object, y: object) -> tuple[np.ndarray, np.ndarray]:
    """``(x (n, d), y (n, k))`` as finite float matrices with one row per point, or a refusal."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 2:
        refuse(f'a fit takes points (n, d) and values (n, k), one row per point; got {x.shape} and {y.shape}')
    if x.shape[0] != y.shape[0] or x.shape[0] == 0:
        refuse(f'the values must hold one row per point, got {y.shape[0]} row(s) for {x.shape[0]} point(s)')
    return require_finite(x, 'the points a model is fitted on'), require_finite(y, 'the values it is fitted on')


def require_gives(gives: frozenset[Give], *, variance: bool, gradient: bool, what: str) -> None:
    """Refuse a request for a part *what* does not give, naming what it does give."""
    asked = {part for part, wanted in ((Give.VARIANCE, variance), (Give.GRADIENT, gradient)) if wanted}
    missing = sorted(asked - gives)
    if missing:
        refuse(f'{what} gives no {", ".join(missing)}; it gives the mean and {sorted(gives) or "nothing else"}')
