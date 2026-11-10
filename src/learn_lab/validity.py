"""Where a fitted model may be asked: its training envelope, persisted with the fit (vectors and channel samples).

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

__all__ = ['BoxEnvelope', 'ChannelEnvelope', 'Validity', 'channel_entries']


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

    @property
    def span(self) -> np.ndarray:
        """``upper - lower`` per input, 1 where the box is flat (so :meth:`unit` stays finite)."""
        span = self.upper - self.lower
        return np.where(span > 0, span, 1.0)

    def unit(self, x: np.ndarray) -> np.ndarray:
        """``x`` mapped onto the unit box: ``(x - lower) / span``; a slope in it is ``span`` times the raw one."""
        return (np.asarray(x, dtype=float) - self.lower) / self.span

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


def channel_entries(x: np.ndarray) -> np.ndarray:
    """``(n, c, *shape) -> (n, points, c)``: every grid entry of every sample as one channel vector."""
    return np.moveaxis(x.reshape(x.shape[0], x.shape[1], -1), 1, 2)


@dataclass(frozen=True, slots=True, eq=False)
class ChannelEnvelope:
    box: BoxEnvelope
    shape: tuple[int, ...] | None

    @classmethod
    def of(cls, x: np.ndarray, *, fixed_shape: bool = True, margin: float = 0.0) -> ChannelEnvelope:
        x = np.asarray(x, float)
        if x.ndim < 2 or x.shape[0] == 0:
            refuse(f'a channel envelope is spanned by (n, c, *shape) samples, got shape {x.shape}')
        box = BoxEnvelope.of(channel_entries(x).reshape(-1, x.shape[1]), margin=margin)
        return cls(box, tuple(x.shape[2:]) if fixed_shape else None)

    @property
    def channels(self) -> int:
        return self.box.width

    def _legal(self, x: np.ndarray) -> bool:
        if x.ndim < 2 or x.shape[1] != self.channels:
            return False
        return self.shape is None or tuple(x.shape[2:]) == self.shape

    def classify(self, x: np.ndarray) -> np.ndarray:
        """One :class:`Validity` per sample: its worst entry decides."""
        x = np.asarray(x, float)
        if not self._legal(x):
            return np.full(x.shape[0] if x.ndim else 1, Validity.INVALID, dtype=object)
        entries = channel_entries(x)
        verdict = self.box.classify(entries.reshape(-1, self.channels)).reshape(entries.shape[:2])
        out = np.full(x.shape[0], Validity.IN_DOMAIN, dtype=object)
        out[np.any(verdict == Validity.OUT_OF_DOMAIN, axis=1)] = Validity.OUT_OF_DOMAIN
        out[np.any(verdict == Validity.INVALID, axis=1)] = Validity.INVALID
        return out

    def require(self, x: np.ndarray, what: str) -> np.ndarray:
        x = np.asarray(x, float)
        if not self._legal(x):
            refuse(
                f'{what}: shape {x.shape} is invalid; the envelope takes (n, {self.channels}'
                f'{", " + ", ".join(map(str, self.shape)) if self.shape else ""}{", *grid" if self.shape is None else ""})'
            )
        verdict = self.classify(x)
        bad = np.flatnonzero(verdict != Validity.IN_DOMAIN)
        if bad.size:
            refuse(
                f'{what}: {bad.size} sample(s) are {verdict[bad[0]]}, first sample {int(bad[0])}; '
                f'the channel envelope is lower={self.box.lower}, upper={self.box.upper}'
            )
        return x
