"""A learned constitutive law ``y = f(x)``: strictly monotone by construction, fitted to measured pairs.

The fitted law crosses the numpy boundary as two plain callables, :meth:`FittedLaw.value` and
:meth:`FittedLaw.derivative` -- exactly what a classical Newton solver takes (the HYBRID seam: numerics
never imports learn-lab, it is handed a callable). Monotonicity gives ``f' > 0`` everywhere, so the
solver's tangent never vanishes or flips sign; ``odd`` makes ``f(-x) = -f(x)`` exact (B-H through zero).
The law is valid on the measured range only: a query outside it is refused.

``loss`` picks what the fit makes uniform: ``'mse'`` (the default) the ABSOLUTE error, so a law spanning
decades is fitted at its large end; ``'relative'`` the error RELATIVE to ``|y|`` at each sample, so every
decade is fitted alike (a B-H curve over four decades of H). The divisor is floored at the smallest nonzero
``|y|`` so a sample at ``y = 0`` stays finite.

``input_warp`` picks the coordinate the network sees: ``'linear'`` (the default) ``x / max|x|``; ``'log'``
``asinh(x / k) / asinh(max|x| / k)`` -- odd, smooth and strictly increasing, linear below the knee ``k`` and
logarithmic above it, so each decade of a decades-spanning input gets an equal share of ``[-1, 1]``. The knee
sits half a decade above the smallest nonzero ``|x|`` measured: a law that is linear at its low end (B = mu H)
stays linear in the warped coordinate there, where a knee AT the smallest sample makes it exponential in the
warped coordinate, which tanh and softplus units fit poorly (measured, four-decade B(H), relative loss, Adam
2500..3500 steps and learning rate 2.7e-3..3.3e-3: low-decade error 0.03..0.25 % at the half-decade knee,
0.17..0.90 % at the smallest sample, 0.21..1.76 % unwarped). Under ``'linear'`` the lowest of four decades occupies
``[1e-4, 1e-3]``, far below every unit's initial knee: reaching it means driving weights up by ~1e3, and
where the optimiser stops on that path decides the low-end fit (measured: 0.26 % on Windows CPU, over 1 % on
the Linux forge, same seed). The warp is part of the law, so ``derivative`` differentiates through it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import torch

from learn_lab.deep.fit import Checkpoint, Schedule, minimise
from learn_lab.deep.networks import MonotoneScalar
from learn_lab.deep.runtime import Runtime, numpy
from learn_lab.errors import refuse
from learn_lab.validity import BoxEnvelope

__all__ = ['INPUT_WARPS', 'LOSSES', 'FittedLaw', 'MonotoneLaw']

#: The losses a law fits with; see the module docstring.
LOSSES: Final = ('mse', 'relative')

#: The input coordinates a law fits in; see the module docstring.
INPUT_WARPS: Final = ('linear', 'log')

#: The ``'log'`` warp's knee over the smallest nonzero ``|x|``: half a decade.
_KNEE_OVER_SMALLEST: Final = 10.0**0.5


def _sample_weights(loss: str, y: np.ndarray) -> np.ndarray:
    if loss == 'mse':
        return np.ones_like(y)
    nonzero = np.abs(y[y != 0])
    floor = float(nonzero.min()) if nonzero.size else 1.0
    return 1.0 / np.maximum(np.abs(y), floor) ** 2


@dataclass(frozen=True, slots=True)
class MonotoneLaw:
    hidden: int = 16
    odd: bool = True
    schedule: Schedule = field(default_factory=lambda: Schedule(adam_steps=3000, learning_rate=1e-2, lbfgs_steps=200))
    runtime: Runtime = field(default_factory=Runtime)
    loss: str = 'mse'
    input_warp: str = 'linear'

    def fit(self, x: np.ndarray, y: np.ndarray, *, checkpoint: Checkpoint | None = None) -> FittedLaw:
        x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
        if x.shape != y.shape or x.size < 2:
            refuse(f'a law is fitted on matched pairs (n >= 2), got {x.shape} and {y.shape}')
        if self.loss not in LOSSES:
            refuse(f'loss is one of {LOSSES}, got {self.loss!r}')
        if self.input_warp not in INPUT_WARPS:
            refuse(f'input_warp is one of {INPUT_WARPS}, got {self.input_warp!r}')
        rt = self.runtime
        rt.seeded()
        # Scale without shifting: a shift would break the exact oddness of the law.
        x_scale, y_scale = float(np.max(np.abs(x))) or 1.0, float(np.max(np.abs(y))) or 1.0
        net = MonotoneScalar(self.hidden, odd=self.odd).to(rt.torch_device, rt.torch_dtype)
        knee = None
        if self.input_warp == 'log':
            nonzero = np.abs(x[x != 0])
            knee = _KNEE_OVER_SMALLEST * float(nonzero.min()) / x_scale if nonzero.size else 1.0
        xt, yt = _warp(rt.tensor(x / x_scale), knee), rt.tensor(y / y_scale)
        w = _sample_weights(self.loss, y / y_scale)
        wt = rt.tensor(w / w.mean())
        history = minimise(
            net.parameters(),
            lambda: torch.mean(wt * (net(xt) - yt) ** 2),
            self.schedule,
            checkpoint=checkpoint,
            config=repr(self),
            data=[xt, yt],
        )
        span = np.abs(x).max() if self.odd else None
        envelope = BoxEnvelope(np.array([-span]), np.array([span])) if self.odd else BoxEnvelope.of(x[:, None])
        return FittedLaw(
            net=net.eval(),
            runtime=rt,
            x_scale=x_scale,
            y_scale=y_scale,
            knee=knee,
            envelope=envelope,
            history=history,
        )


def _warp(x: torch.Tensor, knee: float | None) -> torch.Tensor:
    """The network's input coordinate for scaled ``x``: identity, or ``asinh(x / knee) / asinh(1 / knee)``."""
    if knee is None:
        return x
    return torch.asinh(x / knee) / float(np.arcsinh(1.0 / knee))


@dataclass(frozen=True, slots=True, eq=False)
class FittedLaw:
    net: MonotoneScalar
    runtime: Runtime
    x_scale: float
    y_scale: float
    knee: float | None
    envelope: BoxEnvelope
    history: list[float]

    def _inputs(self, x: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
        x = np.asarray(x, float)
        self.envelope.require(x.reshape(-1, 1), 'a constitutive-law query')
        return x.ravel(), x.shape

    def value(self, x: np.ndarray) -> np.ndarray:
        flat, shape = self._inputs(x)
        with torch.no_grad():
            return (numpy(self.net(_warp(self.runtime.tensor(flat / self.x_scale), self.knee))) * self.y_scale).reshape(
                shape
            )

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """``df/dx`` -- strictly positive everywhere by construction."""
        flat, shape = self._inputs(x)
        xt = self.runtime.tensor(flat / self.x_scale, grad=True)
        (slope,) = torch.autograd.grad(self.net(_warp(xt, self.knee)).sum(), xt)
        return (numpy(slope) * self.y_scale / self.x_scale).reshape(shape)
