"""D56 -- an ensemble: members fitted on the same data and combined by their cross-validated R2.

Each member's weight is its k-fold R2 (:func:`~learn_lab.validation.cross_validate`, seeded
by ``seed``), clamped at zero and normalised (``weighting='score'``), or all of it on the best member
(``weighting='best'``). An ensemble whose every member scores ``r2 <= 0`` is REFUSED: a uniform
average then would be a mean, not a weight. The ensemble gives what EVERY member gives: the
gradient is the weighted gradient, the variance the mixture's (law of total variance).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from learn_lab.errors import refuse
from learn_lab.protocol import Fitted, Give, Prediction, Surrogate, query_points, require_gives, training_data
from learn_lab.validation import cross_validate, score
from learn_lab.validity import BoxEnvelope

__all__ = ['WEIGHTINGS', 'Ensemble', 'FittedEnsemble']

#: The declared weightings.
WEIGHTINGS: Final = ('score', 'best')


@dataclass(frozen=True, slots=True)
class Ensemble:
    """*members* (surrogate declarations), their *weighting*, and the *k*-fold, *seed*-ed validation that weighs them."""

    members: tuple[Surrogate, ...]
    weighting: str = 'score'
    k: int = 5
    seed: int | None = None

    def __post_init__(self) -> None:
        if not self.members:
            refuse('an ensemble combines at least one member, got none')
        if self.weighting not in WEIGHTINGS:
            refuse(f'an ensemble weighting is one of {list(WEIGHTINGS)}, got {self.weighting!r}')

    @property
    def gives(self) -> frozenset[Give]:
        return frozenset.intersection(*(m.gives for m in self.members))

    def fit(self, x, y, *, domain: BoxEnvelope | None = None) -> FittedEnsemble:
        x, y, envelope = training_data(x, y, domain)
        r2 = np.array([
            score('r2', y, cross_validate(m, x, y, k=self.k, seed=self.seed, domain=envelope).mean)
            for m in self.members
        ])
        if not np.any(r2 > 0):
            refuse(
                f'every ensemble member scored r2 <= 0 in cross-validation ({r2.tolist()}); there is no weight to give'
            )
        weights = (
            np.eye(len(r2))[int(np.argmax(r2))]
            if self.weighting == 'best'
            else np.maximum(r2, 0) / np.maximum(r2, 0).sum()
        )
        return FittedEnsemble(tuple(m.fit(x, y, domain=envelope) for m in self.members), weights, self.gives)


@dataclass(frozen=True, slots=True, eq=False)
class FittedEnsemble:
    """The fitted members and their weights."""

    members: tuple[Fitted, ...]
    weights: np.ndarray
    gives: frozenset[Give]

    @property
    def envelope(self) -> BoxEnvelope:
        """The one envelope every member was fitted in (held by the members)."""
        return self.members[0].envelope

    def predict(self, x, *, variance: bool = False, gradient: bool = False) -> Prediction:
        require_gives(self.gives, variance=variance, gradient=gradient, what='this ensemble')
        x = query_points(x, self.envelope)
        live = [(w, m) for w, m in zip(self.weights, self.members, strict=True) if w > 0]
        parts = [(w, m.predict(x, variance=variance, gradient=gradient)) for w, m in live]
        mean = sum(w * p.mean for w, p in parts)
        var = sum(w * (p.variance + p.mean**2) for w, p in parts) - mean**2 if variance else None
        slope = sum(w * p.gradient for w, p in parts) if gradient else None
        return Prediction(mean, None if var is None else np.maximum(var, 0.0), slope)
