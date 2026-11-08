"""D42 -- the L6 protocol: a model is ``fit -> Fitted``; a fitted model is ``predict -> Prediction``.

A :class:`Prediction` is the mean ``(n, k)`` and, when asked for and given, the variance ``(n, k)``
(LATENT: of the mean function, not of a new noisy observation) and the gradient of the mean
``(n, k, d)``. A model declares what it ``gives`` beyond the mean as a set of :class:`Give`; asking
for anything else is refused naming what it does give -- never answered with a placeholder.

Every model sees points as an ``(n, d)`` matrix and values as an ``(n, k)`` matrix, one row per
point, finite. A prediction is never an observation: whoever files it labels it as predicted.

Every fit carries its ENVELOPE (a :class:`~learn_lab.validity.BoxEnvelope`): the ``domain`` the caller
declares at fit -- the design space a search will query, which must hold every training point -- else the
box the training points span. A query outside it is refused as OUT_OF_DOMAIN, never extrapolated; a caller
with a fallback asks ``fitted.envelope.classify`` first. :func:`training_data` makes it, :func:`query_points`
holds every query to it: one rule, read by every response model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np

from learn_lab.errors import refuse, require_finite
from learn_lab.validity import BoxEnvelope

__all__ = ['Fitted', 'Give', 'Prediction', 'Surrogate', 'query_points', 'require_gives', 'training_data']


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
    """A fitted model: what it gives, where it may be asked, and its prediction at ``(m, d)`` points."""

    @property
    def gives(self) -> frozenset[Give]: ...

    @property
    def envelope(self) -> BoxEnvelope: ...

    def predict(self, x: np.ndarray, *, variance: bool = False, gradient: bool = False) -> Prediction: ...


class Surrogate(Protocol):
    """A model declaration: its settings, what its fit gives, and ``fit(x, y, domain=) -> Fitted``.

    Models that need more than values -- gradient-enhanced Kriging (gradients), co-Kriging
    (fidelity levels) -- take it as a further ``fit`` argument and return the same :class:`Fitted`.
    """

    @property
    def gives(self) -> frozenset[Give]: ...

    def fit(self, x: np.ndarray, y: np.ndarray, *, domain: BoxEnvelope | None = None) -> Fitted: ...


def training_data(
    x: object, y: object, domain: BoxEnvelope | None = None
) -> tuple[np.ndarray, np.ndarray, BoxEnvelope]:
    """``(x (n, d), y (n, k))`` as finite float matrices with one row per point, and the fit's envelope.

    The envelope is *domain* when declared -- refused unless it holds every training point -- else the
    box the points span.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 2:
        refuse(f'a fit takes points (n, d) and values (n, k), one row per point; got {x.shape} and {y.shape}')
    if x.shape[0] != y.shape[0] or x.shape[0] == 0:
        refuse(f'the values must hold one row per point, got {y.shape[0]} row(s) for {x.shape[0]} point(s)')
    x = require_finite(x, 'the points a surrogate is fitted on')
    y = require_finite(y, 'the values it is fitted on')
    if domain is None:
        return x, y, BoxEnvelope.of(x)
    if not isinstance(domain, BoxEnvelope):
        refuse(f'a declared domain is a BoxEnvelope, got {type(domain).__qualname__}')
    domain.require(x, 'a declared domain holds every training point')
    return x, y, domain


def query_points(x: object, envelope: BoxEnvelope) -> np.ndarray:
    """``(m, d)`` finite query points inside the fit's *envelope*, or a refusal naming the first stranger."""
    x = np.asarray(x, dtype=float)
    d = envelope.width
    if x.ndim != 2 or x.shape[1] != d:
        refuse(f'a prediction takes (m, {d}) points -- the width the model was fitted on -- got {x.shape}')
    require_finite(x, 'the points a surrogate predicts at')
    return envelope.require(x, 'a prediction is asked only inside the envelope of its fit')


def require_gives(gives: frozenset[Give], *, variance: bool, gradient: bool, what: str) -> None:
    """Refuse a request for a part *what* does not give, naming what it does give."""
    asked = {part for part, wanted in ((Give.VARIANCE, variance), (Give.GRADIENT, gradient)) if wanted}
    missing = sorted(asked - gives)
    if missing:
        refuse(f'{what} gives no {", ".join(missing)}; it gives the mean and {sorted(gives) or "nothing else"}')
