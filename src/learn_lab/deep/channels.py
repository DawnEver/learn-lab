"""Channel-wise inputs ``(n, c, *shape)``: their standardisation and their training envelope.

Fields, operators and graphs see samples whose channel axis is 1 and whose remaining axes are a grid, a
sensor line or nothing (a graph's ``(nodes, features)`` is ``n`` samples of shape ``()``). The envelope is
the per-channel value box every training entry spans, plus the trailing ``shape``: a sample of another shape
is INVALID, unless ``shape`` is ``None`` (a resolution-free family such as the FNO). numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from learn_lab.errors import refuse
from learn_lab.validity import BoxEnvelope, Validity

__all__ = ['ChannelEnvelope', 'Standardiser']


def _entries(x: np.ndarray) -> np.ndarray:
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
        box = BoxEnvelope.of(_entries(x).reshape(-1, x.shape[1]), margin=margin)
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
        entries = _entries(x)
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


@dataclass(frozen=True, slots=True, eq=False)
class Standardiser:
    """Per-channel ``(x - shift) / scale`` over axis 1; a constant channel keeps scale 1."""

    shift: np.ndarray
    scale: np.ndarray

    @classmethod
    def of(cls, x: np.ndarray) -> Standardiser:
        entries = _entries(np.asarray(x, float)).reshape(-1, x.shape[1])
        std = entries.std(0)
        return cls(entries.mean(0), np.where(std > 0, std, 1.0))

    def _broadcast(self, a: np.ndarray, ndim: int) -> np.ndarray:
        return a.reshape(1, -1, *([1] * (ndim - 2)))

    def forward(self, x: np.ndarray) -> np.ndarray:
        return (x - self._broadcast(self.shift, x.ndim)) / self._broadcast(self.scale, x.ndim)

    def back(self, z: np.ndarray) -> np.ndarray:
        return z * self._broadcast(self.scale, z.ndim) + self._broadcast(self.shift, z.ndim)
