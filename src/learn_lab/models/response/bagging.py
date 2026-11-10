"""Bagging: one declared surrogate fitted on bootstrap resamples, the fits averaged.

:class:`Bagging` fits ``n_members`` copies of *member*, each on ``round(max_samples * n)`` rows drawn
with replacement by ``default_rng(seed)``, and predicts their mean. Every member is fitted in the ONE
envelope of the bag (the declared ``domain``, else the box all rows span), so a resample that misses an
extreme row never narrows where the bag answers. It always gives a variance: the spread of the member
means (law of total variance, plus each member's own variance when the member gives one). It gives a
gradient when the member does -- the mean of the member gradients. The default member is a
:class:`~learn_lab.models.response.tree.RegressionTree`: Breiman's bagged trees.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from learn_lab.errors import refuse
from learn_lab.models.response.tree import RegressionTree
from learn_lab.protocol import Fitted, Give, Prediction, Surrogate, query_points, require_gives, training_data
from learn_lab.validity import BoxEnvelope

__all__ = ['Bagging', 'FittedBagging']


@dataclass(frozen=True, slots=True)
class Bagging:
    """*n_members* fits of *member* on resamples of ``max_samples`` of the rows, drawn by *seed*."""

    member: Surrogate = field(default_factory=RegressionTree)
    n_members: int = 10
    max_samples: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.n_members < 2:
            refuse(f'bagging averages at least two members, got {self.n_members}')
        if not 0.0 < self.max_samples <= 1.0:
            refuse(f'max_samples is a share of the rows in (0, 1], got {self.max_samples}')

    @property
    def gives(self) -> frozenset[Give]:
        return frozenset({Give.VARIANCE}) | (self.member.gives & {Give.GRADIENT})

    def fit(self, x, y, *, domain: BoxEnvelope | None = None) -> FittedBagging:
        x, y, envelope = training_data(x, y, domain)
        rng = np.random.default_rng(self.seed)
        size = max(1, round(self.max_samples * len(x)))
        draws = [rng.integers(0, len(x), size) for _ in range(self.n_members)]
        members = tuple(self.member.fit(x[d], y[d], domain=envelope) for d in draws)
        return FittedBagging(members, self.gives, frozenset(self.member.gives))


@dataclass(frozen=True, slots=True, eq=False)
class FittedBagging:
    """The fitted members, averaged with equal weight."""

    members: tuple[Fitted, ...]
    gives: frozenset[Give]
    member_gives: frozenset[Give]

    @property
    def envelope(self) -> BoxEnvelope:
        """The one envelope every member was fitted in (held by the members)."""
        return self.members[0].envelope

    def predict(self, x, *, variance: bool = False, gradient: bool = False) -> Prediction:
        require_gives(self.gives, variance=variance, gradient=gradient, what='this bagging surrogate')
        x = query_points(x, self.envelope)
        own = variance and Give.VARIANCE in self.member_gives
        parts = [m.predict(x, variance=own, gradient=gradient) for m in self.members]
        means = np.stack([p.mean for p in parts])
        var = None
        if variance:
            var = means.var(axis=0) + (np.mean([p.variance for p in parts], axis=0) if own else 0.0)
        slope = np.mean([p.gradient for p in parts], axis=0) if gradient else None
        return Prediction(means.mean(axis=0), var, slope)
