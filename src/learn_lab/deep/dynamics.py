"""Neural ODE: learn ``dx/dt = f(x)`` from sampled trajectories, integrated by a fixed-step RK4 written here.

Training is multiple shooting: from every sample of every trajectory, integrate ``horizon`` steps of the
sampling grid and match the samples reached -- short windows keep the gradient stable, full trajectories
are only predicted. States are standardised per component. The envelope is the box of the training states:
a start outside it is refused, and so is a prediction whose trajectory LEAVES it (the learned field was
never trained there), naming the first time it did.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from learn_lab.deep.fit import Checkpoint, Schedule, minimise
from learn_lab.deep.networks import MLP
from learn_lab.deep.runtime import Runtime, numpy
from learn_lab.errors import refuse, require_finite
from learn_lab.validity import BoxEnvelope, Validity

__all__ = ['FittedDynamics', 'NeuralODE', 'rk4']


def rk4(rate: Callable[[torch.Tensor], torch.Tensor], x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Classic RK4 of autonomous ``dx/dt = rate(x)`` from ``x0 (n, d)`` over the times ``t (T,)``: ``(n, T, d)``."""
    states = [x0]
    for dt in torch.diff(t):
        x = states[-1]
        k1 = rate(x)
        k2 = rate(x + dt / 2 * k1)
        k3 = rate(x + dt / 2 * k2)
        k4 = rate(x + dt * k3)
        states.append(x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
    return torch.stack(states, dim=1)


@dataclass(frozen=True, slots=True)
class NeuralODE:
    hidden: Sequence[int] = (64, 64)
    horizon: int = 10
    schedule: Schedule = field(default_factory=lambda: Schedule(adam_steps=2000, learning_rate=3e-3))
    runtime: Runtime = field(default_factory=Runtime)
    margin: float = 0.0

    def fit(self, t: np.ndarray, trajectories: np.ndarray, *, checkpoint: Checkpoint | None = None) -> FittedDynamics:
        """``t (T,)`` uniform sample times, ``trajectories (m, T, d)`` states sampled at them."""
        t = require_finite(np.asarray(t, float), 'the sample times')
        x = require_finite(np.asarray(trajectories, float), 'the trajectories')
        if t.ndim != 1 or x.ndim != 3 or x.shape[1] != t.size or t.size <= self.horizon or self.horizon < 1:
            refuse(
                f'a neural ODE takes t (T,) and trajectories (m, T, d) with T > horizon={self.horizon}; '
                f'got {t.shape}, {x.shape}'
            )
        dt = np.diff(t)
        if not np.all(dt > 0) or np.ptp(dt) > 1e-9 * dt.mean():
            refuse('the sample times must be uniform and increasing')
        rt = self.runtime
        rt.seeded()
        flat = x.reshape(-1, x.shape[2])
        shift, scale = flat.mean(0), np.where(flat.std(0) > 0, flat.std(0), 1.0)
        z = (x - shift) / scale
        windows = np.stack([z[:, i : i + self.horizon + 1] for i in range(t.size - self.horizon)], 1)
        windows = rt.tensor(windows.reshape(-1, self.horizon + 1, x.shape[2]))
        window_t = rt.tensor(t[: self.horizon + 1] - t[0])
        net = MLP([x.shape[2], *self.hidden, x.shape[2]], activation='tanh').to(rt.torch_device, rt.torch_dtype)
        time_scale = float(t[-1] - t[0])

        def loss() -> torch.Tensor:
            return ((rk4(lambda s: net(s) / time_scale, windows[:, 0], window_t) - windows) ** 2).mean()

        history = minimise(net.parameters(), loss, self.schedule, checkpoint=checkpoint, config=repr(self))
        return FittedDynamics(
            net=net.eval(),
            time_scale=time_scale,
            runtime=rt,
            envelope=BoxEnvelope.of(flat, margin=self.margin),
            shift=shift,
            scale=scale,
            history=history,
        )


@dataclass(frozen=True, slots=True, eq=False)
class FittedDynamics:
    net: MLP
    time_scale: float
    runtime: Runtime
    envelope: BoxEnvelope
    shift: np.ndarray
    scale: np.ndarray
    history: list[float]

    def _rate(self, s: torch.Tensor) -> torch.Tensor:
        """The learned rate in standardised state per unit time: the net's output over the trained time span."""
        return self.net(s) / self.time_scale

    def rate(self, x: np.ndarray) -> np.ndarray:
        """``dx/dt (n, d)`` at in-domain states."""
        x = self.envelope.require(x, 'a learned-rate query')
        with torch.no_grad():
            return numpy(self._rate(self.runtime.tensor((x - self.shift) / self.scale))) * self.scale

    def integrate(self, x0: np.ndarray, t: np.ndarray) -> np.ndarray:
        """``(n, T, d)`` from in-domain starts ``x0 (n, d)`` over increasing times ``t (T,)``, ``t[0]`` at ``x0``."""
        x0 = self.envelope.require(x0, 'a neural-ODE start')
        t = np.asarray(t, float)
        if t.ndim != 1 or t.size < 2 or not np.all(np.diff(t) > 0):
            refuse(f'integration times are increasing (T >= 2,), got {t.shape}')
        rt = self.runtime
        with torch.no_grad():
            z = numpy(rk4(self._rate, rt.tensor((x0 - self.shift) / self.scale), rt.tensor(t)))
        out = z * self.scale + self.shift
        verdict = self.envelope.classify(out.reshape(-1, out.shape[2])).reshape(out.shape[:2])
        left = np.argwhere(verdict != Validity.IN_DOMAIN)
        if left.size:
            n, k = left[0]
            refuse(f'trajectory {n} left the training envelope at t = {t[k]}: the learned rate is unknown there')
        return out
