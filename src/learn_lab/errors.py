"""The one refusal every learn-lab boundary raises: a request the model cannot honestly answer."""

from __future__ import annotations

from typing import NoReturn

import numpy as np

__all__ = ['Refused', 'refuse', 'require_finite']


class Refused(ValueError):
    """A request outside what a model declares it can answer; the message names what it can."""


def refuse(message: str) -> NoReturn:
    raise Refused(message)


def require_finite(array: np.ndarray, what: str) -> np.ndarray:
    if not np.all(np.isfinite(array)):
        refuse(f'{what} must be finite; {int(np.sum(~np.isfinite(array)))} entries are not')
    return array
