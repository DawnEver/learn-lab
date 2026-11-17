"""A known law as the mean and a learned residual on top: ``y = law(x) + residual(x)``.

The *law* is any surrogate that states the trend the data are expected to follow (a :class:`PowerLaw` scaling,
a low-degree :class:`Polynomial`); it is fitted on ``(x, y)`` first. The *base* surrogate (a Gaussian process by
default) is then fitted on ``y - law(x)`` with its inputs mapped onto the unit box of the fit's envelope, so no
input's unit sets its length scales. The prediction's mean is the sum, its variance the residual's (the law is
taken as exact), and its gradient the law's plus the residual's, chained through that map -- a gradient is given
only when both give one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from learn_lab.models.response.gaussian_process import GaussianProcess
from learn_lab.models.response.power_law import PowerLaw
from learn_lab.protocol import Give, Prediction, Surrogate, query_points, require_gives, training_data
from learn_lab.validity import BoxEnvelope

__all__ = ['FittedLawResidual', 'LawResidual']


def _gives(law: frozenset[Give], residual: frozenset[Give]) -> frozenset[Give]:
    """The residual's parts, less a gradient the law cannot give."""
    return frozenset(g for g in residual if g is not Give.GRADIENT or Give.GRADIENT in law)


def _span(envelope: BoxEnvelope) -> np.ndarray:
    """The envelope's width per input; a constant input (zero width) keeps unit scale."""
    span = envelope.upper - envelope.lower
    return np.where(span > 0, span, 1.0)


def _unit(x: np.ndarray, envelope: BoxEnvelope) -> np.ndarray:
    """*x* mapped onto the unit box of *envelope*: the residual's inputs."""
    return (x - envelope.lower) / _span(envelope)


@dataclass(frozen=True, slots=True)
class LawResidual:
    """*law* fitted as the mean, *base* fitted on what it leaves."""

    law: Surrogate = field(default_factory=PowerLaw)
    base: Surrogate = field(default_factory=GaussianProcess)

    @property
    def gives(self) -> frozenset[Give]:
        return _gives(self.law.gives, self.base.gives)

    def fit(self, x, y, *, domain: BoxEnvelope | None = None) -> FittedLawResidual:
        x, y, envelope = training_data(x, y, domain)
        law = self.law.fit(x, y, domain=envelope)
        unit = BoxEnvelope(np.zeros(x.shape[1]), np.ones(x.shape[1]))
        residual = self.base.fit(_unit(x, envelope), y - law.predict(x).mean, domain=unit)
        return FittedLawResidual(law, residual, envelope, y.shape[1])


@dataclass(frozen=True, slots=True, eq=False)
class FittedLawResidual:
    """``law(x) + residual(x)`` over ``outputs`` columns; the residual reads *x* on the envelope's unit box."""

    law: object
    residual: object
    envelope: BoxEnvelope
    outputs: int

    @property
    def gives(self) -> frozenset[Give]:
        return _gives(self.law.gives, self.residual.gives)

    def predict(self, x, *, variance: bool = False, gradient: bool = False) -> Prediction:
        require_gives(self.gives, variance=variance, gradient=gradient, what='a law with a learned residual')
        x = query_points(x, self.envelope)
        law = self.law.predict(x, gradient=gradient)
        residual = self.residual.predict(_unit(x, self.envelope), variance=variance, gradient=gradient)
        return Prediction(
            law.mean + residual.mean,
            residual.variance,
            None if not gradient else law.gradient + residual.gradient / _span(self.envelope),
        )
