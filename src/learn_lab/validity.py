"""Where a fitted model may be asked: its training envelope, persisted with the fit.

Three answers, never a silent extrapolation:

- ``INVALID``: the point breaks a hard constraint (non-finite, wrong width) -- always refused;
- ``OUT_OF_DOMAIN``: legal, but outside the envelope the fit was trained and validated on -- refused by
  :meth:`BoxEnvelope.require`; a caller that has a fallback (the classical solver) asks :meth:`classify`;
- ``IN_DOMAIN``: inside the envelope.

The box is the simplest envelope that is honest for a vector input; richer inputs (grids, graphs) declare
their own envelope type with the same two methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from learn_lab.errors import refuse

__all__ = ['BoxEnvelope', 'Validity']


class Validity(StrEnum):
    IN_DOMAIN = 'in_domain'
    OUT_OF_DOMAIN = 'out_of_domain'
    INVALID = 'invalid'


@dataclass(frozen=True, slots=True, eq=False)
class BoxEnvelope:
    """The axis-aligned box ``lower <= x <= upper`` the training points span, widened by ``margin`` of each side."""

    lower: np.ndarray
    upper: np.ndarray

    @classmethod
    def of(cls, x: np.ndarray, *, margin: float = 0.0) -> BoxEnvelope:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[0] == 0:
            refuse(f'an envelope is spanned by (n, d) points, got shape {x.shape}')
        lower, upper = x.min(axis=0), x.max(axis=0)
        pad = margin * (upper - lower)
        return cls(lower - pad, upper + pad)

    @property
    def width(self) -> int:
        return self.lower.shape[0]

    def classify(self, x: np.ndarray) -> np.ndarray:
        """One :class:`Validity` per row of ``x``."""
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.width:
            return np.full(x.shape[0] if x.ndim else 1, Validity.INVALID, dtype=object)
        finite = np.all(np.isfinite(x), axis=1)
        inside = np.all((x >= self.lower) & (x <= self.upper), axis=1)
        out = np.where(inside, Validity.IN_DOMAIN, Validity.OUT_OF_DOMAIN).astype(object)
        out[~finite] = Validity.INVALID
        return out

    def require(self, x: np.ndarray, what: str) -> np.ndarray:
        """``x`` as a float matrix if every row is in the domain, else a refusal naming the first stranger."""
        x = np.asarray(x, dtype=float)
        verdict = self.classify(x)
        bad = np.flatnonzero(verdict != Validity.IN_DOMAIN)
        if bad.size:
            row = int(bad[0])
            refuse(
                f'{what}: {bad.size} point(s) are {verdict[row]}, first row {row} = {x[row] if x.ndim == 2 else x}; '
                f'the envelope is lower={self.lower}, upper={self.upper}'
            )
        return x
