"""Power laws: ``y_k = exp(c_k) * prod_i x_i ** a_ki``, a physics scaling law as a learned family.

Iron-loss (Bertotti) and magnet-loss scalings ``P = C f^a B^b`` are this family. The fit is least squares
in log space, ``log y = c + log(x) @ a``; :attr:`PowerLaw.fixed` pins any exponent to a known value (a
number pins it, ``None`` leaves it free), so with every exponent pinned only the level ``c_k`` is fitted.
A logarithm needs strictly positive data: a non-positive input or output is refused, naming its column
and row, at fit and at prediction. The mean's gradient is exact: ``dy_k/dx_i = y_k a_ki / x_i``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numerics_lab.linear.dense import least_squares

from learn_lab.errors import refuse
from learn_lab.protocol import Give, Prediction, query_points, require_gives, training_data
from learn_lab.validity import BoxEnvelope

__all__ = ['FittedPowerLaw', 'PowerLaw']


def _positive(values: np.ndarray, what: str) -> np.ndarray:
    """*values* unchanged when every entry is > 0, else a refusal naming the first non-positive one."""
    bad = np.argwhere(~(values > 0))
    if bad.size:
        row, column = (int(i) for i in bad[0])
        refuse(f'a power law takes strictly positive {what}: column {column}, row {row} is {values[row, column]!r}')
    return values


@dataclass(frozen=True, slots=True)
class PowerLaw:
    """A power law per output; *fixed* is an ``(outputs, inputs)`` table pinning exponents (``None`` = free)."""

    fixed: tuple[tuple[float | None, ...], ...] | None = None

    def __post_init__(self) -> None:
        if self.fixed is None:
            return
        rows = tuple(tuple(row) for row in self.fixed)
        if not rows or len({len(row) for row in rows}) != 1 or not rows[0]:
            refuse(f'fixed exponents are a non-empty rectangular (outputs, inputs) table, got {self.fixed!r}')
        for row in rows:
            for a in row:
                if a is not None and (isinstance(a, bool) or not isinstance(a, int | float) or not np.isfinite(a)):
                    refuse(f'a fixed exponent is a finite number or None, got {a!r}')
        object.__setattr__(self, 'fixed', rows)

    @property
    def gives(self) -> frozenset[Give]:
        return frozenset({Give.GRADIENT})

    def fit(self, x, y, *, domain: BoxEnvelope | None = None) -> FittedPowerLaw:
        x, y, envelope = training_data(x, y, domain)
        logx = np.log(_positive(x, 'inputs'))
        logy = np.log(_positive(y, 'outputs'))
        (n, d), k = x.shape, y.shape[1]
        if self.fixed is not None and (len(self.fixed), len(self.fixed[0])) != (k, d):
            refuse(
                f'fixed exponents are a ({k}, {d}) table for this data, got ({len(self.fixed)}, {len(self.fixed[0])})'
            )
        exponents, level = np.empty((d, k)), np.empty(k)
        for out in range(k):
            pinned = (None,) * d if self.fixed is None else self.fixed[out]
            free = [i for i, a in enumerate(pinned) if a is None]
            known = np.array([0.0 if a is None else float(a) for a in pinned])
            target = logy[:, out] - logx @ known
            solution = least_squares(np.column_stack([np.ones(n), logx[:, free]]), target[:, None])[:, 0]
            known[free] = solution[1:]
            exponents[:, out], level[out] = known, solution[0]
        return FittedPowerLaw(exponents, level, envelope)


@dataclass(frozen=True, slots=True, eq=False)
class FittedPowerLaw:
    """``y = exp(level) * prod_i x_i ** exponents[i]``; ``exponents`` is ``(inputs, outputs)``, ``level`` ``(outputs,)``."""

    exponents: np.ndarray
    level: np.ndarray
    envelope: BoxEnvelope

    @property
    def gives(self) -> frozenset[Give]:
        return frozenset({Give.GRADIENT})

    def predict(self, x, *, variance: bool = False, gradient: bool = False) -> Prediction:
        require_gives(self.gives, variance=variance, gradient=gradient, what='a power law')
        x = _positive(query_points(x, self.envelope), 'inputs')
        mean = np.exp(self.level + np.log(x) @ self.exponents)
        if not gradient:
            return Prediction(mean)
        return Prediction(mean, gradient=mean[:, :, None] * self.exponents.T[None, :, :] / x[:, None, :])
