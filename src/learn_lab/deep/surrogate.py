"""Offline response surrogate: a deep ensemble ``x (n, d) -> y (n, k)`` honouring the L6 protocol.

``members`` MLPs are trained from different seeds on standardised data. The mean is their average; the
variance is their spread (latent, of the mean function); the gradient is the autograd derivative of the
mean. The training envelope travels with the fit: a query outside it is refused, never extrapolated.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from learn_lab.deep.fit import Schedule, minimise
from learn_lab.deep.networks import mlp
from learn_lab.deep.runtime import Runtime, numpy
from learn_lab.errors import refuse
from learn_lab.protocol import Give, Prediction, require_gives, training_data
from learn_lab.validity import BoxEnvelope

__all__ = ['DeepEnsemble', 'FittedEnsemble']

_GIVES = frozenset({Give.VARIANCE, Give.GRADIENT})


@dataclass(frozen=True, slots=True)
class DeepEnsemble:
    hidden: Sequence[int] = (64, 64)
    members: int = 5
    schedule: Schedule = field(default_factory=lambda: Schedule(adam_steps=3000, learning_rate=3e-3))
    runtime: Runtime = field(default_factory=lambda: Runtime(dtype='float32'))
    margin: float = 0.0

    def __post_init__(self) -> None:
        if self.members < 1:
            refuse(f'an ensemble has at least one member, got {self.members}')

    @property
    def gives(self) -> frozenset[Give]:
        return _GIVES if self.members > 1 else frozenset({Give.GRADIENT})

    def fit(self, x: np.ndarray, y: np.ndarray) -> FittedEnsemble:
        x, y = training_data(x, y)
        rt = self.runtime
        x_shift, x_scale = x.mean(0), np.where(x.std(0) > 0, x.std(0), 1.0)
        y_shift, y_scale = y.mean(0), np.where(y.std(0) > 0, y.std(0), 1.0)
        xt, yt = rt.tensor((x - x_shift) / x_scale), rt.tensor((y - y_shift) / y_scale)
        nets, histories = [], []
        for member in range(self.members):
            replace(rt, seed=rt.seed + member).seeded()
            net = mlp([x.shape[1], *self.hidden, y.shape[1]], activation=torch.nn.SiLU)
            net = net.to(rt.torch_device, rt.torch_dtype)
            histories.append(minimise(net.parameters(), lambda net=net: ((net(xt) - yt) ** 2).mean(), self.schedule))
            nets.append(net.eval())
        return FittedEnsemble(
            nets=tuple(nets),
            runtime=rt,
            envelope=BoxEnvelope.of(x, margin=self.margin),
            x_shift=x_shift,
            x_scale=x_scale,
            y_shift=y_shift,
            y_scale=y_scale,
            gives=self.gives,
            histories=tuple(histories),
        )


@dataclass(frozen=True, slots=True, eq=False)
class FittedEnsemble:
    nets: tuple[torch.nn.Module, ...]
    runtime: Runtime
    envelope: BoxEnvelope
    x_shift: np.ndarray
    x_scale: np.ndarray
    y_shift: np.ndarray
    y_scale: np.ndarray
    gives: frozenset[Give]
    histories: tuple[list[float], ...]

    def predict(self, x: np.ndarray, *, variance: bool = False, gradient: bool = False) -> Prediction:
        require_gives(self.gives, variance=variance, gradient=gradient, what='a deep ensemble')
        x = self.envelope.require(x, 'a deep-ensemble query')
        rt = self.runtime
        xt = rt.tensor((x - self.x_shift) / self.x_scale, grad=gradient)
        with torch.set_grad_enabled(gradient):
            outs = torch.stack([net(xt) for net in self.nets])  # (members, n, k), standardised
            mean_s = outs.mean(0)
            grad = None
            if gradient:
                rows = [
                    torch.autograd.grad(mean_s[:, j].sum(), xt, retain_graph=True)[0] for j in range(mean_s.shape[1])
                ]
                grad = numpy(torch.stack(rows, 1)) * (self.y_scale[None, :, None] / self.x_scale[None, None, :])
        mean = numpy(mean_s) * self.y_scale + self.y_shift
        var = numpy(outs.var(0, unbiased=False)) * self.y_scale**2 if variance else None
        return Prediction(mean, var, grad)
