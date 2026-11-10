"""A learned constitutive law ``y = f(x)``: strictly monotone by construction, fitted to measured pairs.

The fitted law crosses the numpy boundary as two plain callables, :meth:`FittedLaw.value` and
:meth:`FittedLaw.derivative` -- exactly what a classical Newton solver takes (the HYBRID seam: numerics
never imports learn-lab, it is handed a callable). Monotonicity gives ``f' > 0`` everywhere, so the
solver's tangent never vanishes or flips sign; ``odd`` makes ``f(-x) = -f(x)`` exact (B-H through zero).
The law is valid on the measured range only: a query outside it is refused.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from learn_lab.deep.fit import Checkpoint, Schedule, minimise
from learn_lab.deep.networks import MonotoneScalar
from learn_lab.deep.runtime import Runtime, numpy
from learn_lab.errors import refuse
from learn_lab.validity import BoxEnvelope

__all__ = ['FittedLaw', 'MonotoneLaw']


@dataclass(frozen=True, slots=True)
class MonotoneLaw:
    hidden: int = 16
    odd: bool = True
    schedule: Schedule = field(default_factory=lambda: Schedule(adam_steps=3000, learning_rate=1e-2, lbfgs_steps=200))
    runtime: Runtime = field(default_factory=Runtime)

    def fit(self, x: np.ndarray, y: np.ndarray, *, checkpoint: Checkpoint | None = None) -> FittedLaw:
        x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
        if x.shape != y.shape or x.size < 2:
            refuse(f'a law is fitted on matched pairs (n >= 2), got {x.shape} and {y.shape}')
        rt = self.runtime
        rt.seeded()
        # Scale without shifting: a shift would break the exact oddness of the law.
        x_scale, y_scale = float(np.max(np.abs(x))) or 1.0, float(np.max(np.abs(y))) or 1.0
        net = MonotoneScalar(self.hidden, odd=self.odd).to(rt.torch_device, rt.torch_dtype)
        xt, yt = rt.tensor(x / x_scale), rt.tensor(y / y_scale)
        history = minimise(
            net.parameters(),
            lambda: torch.nn.functional.mse_loss(net(xt), yt),
            self.schedule,
            checkpoint=checkpoint,
            config=repr(self),
            data=[xt, yt],
        )
        span = np.abs(x).max() if self.odd else None
        envelope = BoxEnvelope(np.array([-span]), np.array([span])) if self.odd else BoxEnvelope.of(x[:, None])
        return FittedLaw(
            net=net.eval(), runtime=rt, x_scale=x_scale, y_scale=y_scale, envelope=envelope, history=history
        )


@dataclass(frozen=True, slots=True, eq=False)
class FittedLaw:
    net: MonotoneScalar
    runtime: Runtime
    x_scale: float
    y_scale: float
    envelope: BoxEnvelope
    history: list[float]

    def _inputs(self, x: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
        x = np.asarray(x, float)
        self.envelope.require(x.reshape(-1, 1), 'a constitutive-law query')
        return x.ravel(), x.shape

    def value(self, x: np.ndarray) -> np.ndarray:
        flat, shape = self._inputs(x)
        with torch.no_grad():
            return (numpy(self.net(self.runtime.tensor(flat / self.x_scale))) * self.y_scale).reshape(shape)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """``df/dx`` -- strictly positive everywhere by construction."""
        flat, shape = self._inputs(x)
        xt = self.runtime.tensor(flat / self.x_scale, grad=True)
        (slope,) = torch.autograd.grad(self.net(xt).sum(), xt)
        return (numpy(slope) * self.y_scale / self.x_scale).reshape(shape)
