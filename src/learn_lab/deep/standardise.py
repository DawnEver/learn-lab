"""Per-channel standardisation of ``(n, c, *shape)`` samples (their envelope is :class:`learn_lab.validity.ChannelEnvelope`)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from learn_lab.validity import channel_entries

__all__ = ['Standardiser']


@dataclass(frozen=True, slots=True, eq=False)
class Standardiser:
    """Per-channel ``(x - shift) / scale`` over axis 1; a constant channel keeps scale 1."""

    shift: np.ndarray
    scale: np.ndarray

    @classmethod
    def of(cls, x: np.ndarray) -> Standardiser:
        entries = channel_entries(np.asarray(x, float)).reshape(-1, x.shape[1])
        std = entries.std(0)
        return cls(entries.mean(0), np.where(std > 0, std, 1.0))

    def _broadcast(self, a: np.ndarray, ndim: int) -> np.ndarray:
        return a.reshape(1, -1, *([1] * (ndim - 2)))

    def forward(self, x: np.ndarray) -> np.ndarray:
        return (x - self._broadcast(self.shift, x.ndim)) / self._broadcast(self.scale, x.ndim)

    def back(self, z: np.ndarray) -> np.ndarray:
        return z * self._broadcast(self.scale, z.ndim) + self._broadcast(self.shift, z.ndim)
