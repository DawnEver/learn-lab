"""D45 -- stationary correlation kernels with one length scale per input (ARD).

With ``u_j = (a_j - b_j) / l_j`` and ``r = |u|``:

* ``squared_exponential`` -- ``exp(-r^2 / 2)``;
* ``matern12`` -- ``exp(-r)``;
* ``matern32`` -- ``(1 + sqrt3 r) exp(-sqrt3 r)``;
* ``matern52`` -- ``(1 + sqrt5 r + 5 r^2 / 3) exp(-sqrt5 r)``.

Each is published with its derivative in ``log l_j`` (for the likelihood gradient) and in the first
argument (for the gradient of a prediction). Where ``r = 0`` the Matern-1/2 derivatives are not
defined and are taken as 0, as the symmetric limit.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from learn_lab.errors import refuse

__all__ = ['KERNEL_FAMILIES', 'correlation', 'input_slopes', 'length_slopes', 'require_family']

_S3: Final = float(np.sqrt(3.0))
_S5: Final = float(np.sqrt(5.0))

#: The declared kernel families; a stranger is refused naming them.
KERNEL_FAMILIES: Final = ('squared_exponential', 'matern12', 'matern32', 'matern52')


def require_family(family: object) -> str:
    """*family* when it is declared, else a refusal naming every family."""
    if family not in KERNEL_FAMILIES:
        refuse(f'a kernel family is one of {list(KERNEL_FAMILIES)}, got {family!r}')
    return str(family)


def _scaled(a: np.ndarray, b: np.ndarray, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = (a[:, None, :] - b[None, :, :]) / lengths
    return u, np.sqrt(np.sum(u**2, axis=2))


def _value(family: str, r: np.ndarray) -> np.ndarray:
    if family == 'squared_exponential':
        return np.exp(-0.5 * r**2)
    if family == 'matern12':
        return np.exp(-r)
    if family == 'matern32':
        return (1 + _S3 * r) * np.exp(-_S3 * r)
    return (1 + _S5 * r + 5 * r**2 / 3) * np.exp(-_S5 * r)


def _radial(family: str, r: np.ndarray) -> np.ndarray:
    """``-(dR/dr) / r``: the factor every derivative shares."""
    if family == 'squared_exponential':
        return np.exp(-0.5 * r**2)
    if family == 'matern12':
        return np.where(r > 0, np.exp(-r) / np.where(r > 0, r, 1.0), 0.0)
    if family == 'matern32':
        return 3 * np.exp(-_S3 * r)
    return 5 / 3 * (1 + _S5 * r) * np.exp(-_S5 * r)


def correlation(family: str, a: np.ndarray, b: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """``R(a_i, b_j)``, shape ``(len(a), len(b))``."""
    return _value(family, _scaled(a, b, lengths)[1])


def length_slopes(family: str, x: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """``dR(x_i, x_j) / d log l_k``, shape ``(d, n, n)``."""
    u, r = _scaled(x, x, lengths)
    return np.moveaxis(_radial(family, r)[:, :, None] * u**2, 2, 0)


def input_slopes(family: str, q: np.ndarray, x: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """``dR(q_m, x_n) / dq_m``, shape ``(m, n, d)``."""
    u, r = _scaled(q, x, lengths)
    return -_radial(family, r)[:, :, None] * u / lengths
