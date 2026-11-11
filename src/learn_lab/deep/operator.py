"""Neural operators: function -> function maps learned from sampled pairs.

- :class:`DeepONet` -- a branch net reads the input function at FIXED sensors ``(n, m)``, a trunk net reads
  query coordinates ``(p, d)``; the output ``(n, p)`` is their inner product plus a bias. Queries may be any
  points inside the coordinates' training box.
- :class:`FNO` -- a Fourier neural operator on a regular grid ``(n, c, *grid)``, 1-D or 2-D (``len(modes)``).
  Each layer is a learned multiplier on the lowest ``modes`` Fourier coefficients plus a pointwise linear
  map, so the trained weights do not depend on the grid: a finer grid of the same unit domain is in-domain;
  a grid too coarse to hold the modes is refused. The grid's coordinates in ``[0, 1]`` are appended as
  input channels.

  The FNO's envelope is a per-channel value box over the input functions AT THE TRAINING NODES. A query on
  another grid is judged by its values linearly interpolated onto those nodes, never by its own samples: a
  finer grid samples peaks the training nodes straddled, so its raw range grows with resolution although the
  function is the same. The judged function is the training-grid view of the query -- no margin is added, and
  a function whose values at the training nodes leave the box is OUT_OF_DOMAIN at every resolution.
  ``chunk`` evaluates the full-batch loss that many samples at a time with activation checkpointing (as
  :class:`~learn_lab.deep.field.FieldModel`): the gradient is the full batch's, peak memory ``chunk x width``.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as recomputed

from learn_lab.deep.fit import Checkpoint, Schedule, minimise
from learn_lab.deep.networks import MLP
from learn_lab.deep.runtime import Runtime, numpy
from learn_lab.deep.standardise import Standardiser
from learn_lab.errors import refuse, require_finite
from learn_lab.validity import BoxEnvelope, ChannelEnvelope

__all__ = ['FNO', 'DeepONet', 'FittedDeepONet', 'FittedFNO', 'SpectralConv']


def _scalar(s: Standardiser, a: np.ndarray, *, back: bool = False) -> np.ndarray:
    """One scalar channel standardised: every entry of ``a`` shares the shift and scale."""
    flat = a.reshape(-1, 1)
    return (s.back(flat) if back else s.forward(flat)).reshape(a.shape)


class _DeepONetNet(nn.Module):
    def __init__(self, sensors: int, dim: int, hidden: Sequence[int], basis: int, lower, upper) -> None:
        super().__init__()
        self.arguments = {
            'sensors': sensors,
            'dim': dim,
            'hidden': tuple(hidden),
            'basis': basis,
            'lower': lower,
            'upper': upper,
        }
        self.branch = MLP([sensors, *hidden, basis], activation='gelu')
        self.trunk = MLP([dim, *hidden, basis], activation='gelu')
        self.bias = nn.Parameter(torch.zeros(()))
        span = np.where(upper > lower, upper - lower, 1.0)
        self.register_buffer('lower', torch.as_tensor(lower, dtype=torch.float64))
        self.register_buffer('span', torch.as_tensor(span, dtype=torch.float64))

    def forward(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.branch(u) @ self.trunk(2.0 * (y - self.lower) / self.span - 1.0).T + self.bias


@dataclass(frozen=True, slots=True)
class DeepONet:
    hidden: Sequence[int] = (64, 64)
    basis: int = 32
    schedule: Schedule = field(default_factory=lambda: Schedule(adam_steps=3000, learning_rate=2e-3))
    runtime: Runtime = field(default_factory=lambda: Runtime(dtype='float32'))
    margin: float = 0.0

    def fit(
        self, sensors: np.ndarray, points: np.ndarray, values: np.ndarray, *, checkpoint: Checkpoint | None = None
    ) -> FittedDeepONet:
        """``sensors (n, m)`` input functions, ``points (p, d)`` shared query points, ``values (n, p)``."""
        u = require_finite(np.asarray(sensors, float), 'the sensor values')
        y = require_finite(np.asarray(points, float), 'the query points')
        g = require_finite(np.asarray(values, float), 'the operator values')
        if u.ndim != 2 or y.ndim != 2 or g.shape != (u.shape[0], y.shape[0]) or not len(u):
            refuse(
                f'a DeepONet is fitted on sensors (n, m), points (p, d), values (n, p); '
                f'got {u.shape}, {y.shape}, {g.shape}'
            )
        rt = self.runtime
        rt.seeded()
        su, sg = Standardiser.of(u.reshape(-1, 1)), Standardiser.of(g.reshape(-1, 1))
        box = BoxEnvelope.of(y, margin=self.margin)
        net = _DeepONetNet(u.shape[1], y.shape[1], self.hidden, self.basis, box.lower, box.upper)
        net = net.to(rt.torch_device, rt.torch_dtype)
        ut, yt, gt = rt.tensor(_scalar(su, u)), rt.tensor(y), rt.tensor(_scalar(sg, g))

        def loss() -> torch.Tensor:
            return ((net(ut, yt) - gt) ** 2).mean()

        history = minimise(
            net.parameters(), loss, self.schedule, checkpoint=checkpoint, config=repr(self), data=[ut, yt, gt]
        )
        envelope = ChannelEnvelope.of(u[:, None, :], margin=self.margin)
        return FittedDeepONet(
            net=net.eval(), runtime=rt, sensors=envelope, points=box, inputs=su, outputs=sg, history=history
        )


@dataclass(frozen=True, slots=True, eq=False)
class FittedDeepONet:
    net: _DeepONetNet
    runtime: Runtime
    sensors: ChannelEnvelope
    points: BoxEnvelope
    inputs: Standardiser
    outputs: Standardiser
    history: list[float]

    def predict(self, sensors: np.ndarray, points: np.ndarray) -> np.ndarray:
        """``(n, p)``: each of the ``n`` input functions' image at each of the ``p`` points."""
        u = np.asarray(sensors, float)
        u = self.sensors.require(u[:, None, :] if u.ndim == 2 else u, 'a DeepONet input function')[:, 0]
        y = self.points.require(points, 'a DeepONet query point')
        rt = self.runtime
        with torch.no_grad():
            image = numpy(self.net(rt.tensor(_scalar(self.inputs, u)), rt.tensor(y)))
        return _scalar(self.outputs, image, back=True)


def _corners(modes: Sequence[int], sizes: Sequence[int]) -> list[tuple[tuple[slice, ...], tuple[slice, ...]]]:
    """``(block, spectrum)`` index pairs of the kept low modes: both signs on every axis but the last (rfft)."""
    pairs = []
    for signs in itertools.product((0, 1), repeat=len(modes) - 1):
        block = tuple(slice(s * m, (s + 1) * m) for s, m in zip(signs, modes, strict=False))
        spectrum = tuple(
            slice(0, m) if s == 0 else slice(n - m, n) for s, m, n in zip(signs, modes, sizes, strict=False)
        )
        pairs.append(((*block, slice(0, modes[-1])), (*spectrum, slice(0, modes[-1]))))
    return pairs


class SpectralConv(nn.Module):
    """``x -> irfftn(W * rfftn(x)[low modes])``: a learned complex multiplier on the lowest ``modes`` per axis."""

    def __init__(self, c_in: int, c_out: int, modes: Sequence[int]) -> None:
        super().__init__()
        self.modes = tuple(modes)
        block = (*(2 * m for m in self.modes[:-1]), self.modes[-1])
        scale = 1.0 / (c_in * c_out)
        self.real = nn.Parameter(scale * torch.randn(c_in, c_out, *block))
        self.imag = nn.Parameter(scale * torch.randn(c_in, c_out, *block))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        axes = tuple(range(2, x.ndim))
        spectrum = torch.fft.rfftn(x, dim=axes)
        weight = torch.complex(self.real, self.imag)
        pairs = _corners(self.modes, spectrum.shape[2:])
        low = spectrum.new_zeros((*spectrum.shape[:2], *weight.shape[2:]))
        for block, kept in pairs:
            low[(..., *block)] = spectrum[(..., *kept)]
        mixed = torch.einsum('bi...,io...->bo...', low, weight)
        out = spectrum.new_zeros((x.shape[0], weight.shape[1], *spectrum.shape[2:]))
        for block, kept in pairs:
            out[(..., *kept)] = mixed[(..., *block)]
        return torch.fft.irfftn(out, s=x.shape[2:], dim=axes)


def _pointwise(layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return layer(x.movedim(1, -1)).movedim(-1, 1)


class _FNONet(nn.Module):
    def __init__(self, c_in: int, c_out: int, *, modes: Sequence[int], width: int, layers: int) -> None:
        super().__init__()
        self.arguments = {'c_in': c_in, 'c_out': c_out, 'modes': tuple(modes), 'width': width, 'layers': layers}
        self.lift = nn.Linear(c_in + len(modes), width)
        self.spectral = nn.ModuleList(SpectralConv(width, width, modes) for _ in range(layers))
        self.mix = nn.ModuleList(nn.Linear(width, width) for _ in range(layers))
        self.head = MLP([width, 2 * width, c_out], activation='gelu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        axes = (torch.linspace(0.0, 1.0, n, dtype=x.dtype, device=x.device) for n in x.shape[2:])
        coords = torch.stack(torch.meshgrid(*axes, indexing='ij')).expand(x.shape[0], -1, *x.shape[2:])
        h = _pointwise(self.lift, torch.cat([x, coords], dim=1))
        for i, (spectral, mix) in enumerate(zip(self.spectral, self.mix, strict=True)):
            h = spectral(h) + _pointwise(mix, h)
            if i < len(self.mix) - 1:
                h = nn.functional.gelu(h)
        return _pointwise(self.head, h)


def _onto(x: np.ndarray, grid: Sequence[int]) -> np.ndarray:
    """``x (n, c, *axes)`` on a uniform grid of the unit domain, linearly interpolated onto the nodes ``grid``."""
    for axis, size in enumerate(grid, start=2):
        if x.shape[axis] != size:
            nodes = np.linspace(0.0, 1.0, x.shape[axis])
            x = np.apply_along_axis(lambda f: np.interp(np.linspace(0.0, 1.0, size), nodes, f), axis, x)
    return x


def _require_modes_fit(modes: Sequence[int], grid: Sequence[int]) -> None:
    short = any(n < 2 * m for n, m in zip(grid[:-1], modes[:-1], strict=True)) or grid[-1] // 2 + 1 < modes[-1]
    if short:
        refuse(f'a grid {tuple(grid)} cannot hold the FNO modes {tuple(modes)}: each axis needs >= 2 * modes points')


@dataclass(frozen=True, slots=True)
class FNO:
    modes: tuple[int, ...] = (12,)
    width: int = 32
    layers: int = 4
    schedule: Schedule = field(default_factory=lambda: Schedule(adam_steps=2000, learning_rate=3e-3))
    runtime: Runtime = field(default_factory=lambda: Runtime(dtype='float32'))
    margin: float = 0.0
    chunk: int | None = 32

    def __post_init__(self) -> None:
        if self.chunk is not None and (isinstance(self.chunk, bool) or self.chunk < 1):
            refuse(f'a chunk is a sample count >= 1 (or None for one pass), got {self.chunk!r}')
        if len(self.modes) not in {1, 2} or min(self.modes) < 1:
            refuse(f'an FNO keeps >= 1 mode on each of 1 or 2 axes, got modes={self.modes}')

    def fit(self, x: np.ndarray, y: np.ndarray, *, checkpoint: Checkpoint | None = None) -> FittedFNO:
        """``x (n, c_in, *grid) -> y (n, c_out, *grid)`` on a regular grid of the unit domain."""
        x = require_finite(np.asarray(x, float), 'the input functions')
        y = require_finite(np.asarray(y, float), 'the output functions')
        dims = len(self.modes)
        if x.ndim != 2 + dims or y.ndim != x.ndim or x.shape[0] != y.shape[0] or x.shape[2:] != y.shape[2:]:
            refuse(f'a {dims}-D FNO is fitted on (n, c, *{dims} grid axes) pairs, got {x.shape} -> {y.shape}')
        _require_modes_fit(self.modes, x.shape[2:])
        rt = self.runtime
        rt.seeded()
        sx, sy = Standardiser.of(x), Standardiser.of(y)
        net = _FNONet(x.shape[1], y.shape[1], modes=self.modes, width=self.width, layers=self.layers)
        net = net.to(rt.torch_device, rt.torch_dtype)
        xt, yt = rt.tensor(sx.forward(x)), rt.tensor(sy.forward(y))

        chunk = self.chunk

        def squared(xc: torch.Tensor, yc: torch.Tensor) -> torch.Tensor:
            return ((net(xc) - yc) ** 2).sum()

        def loss() -> torch.Tensor:
            if chunk is None or chunk >= len(xt):
                return squared(xt, yt) / yt.numel()
            parts = (
                recomputed(squared, xt[i : i + chunk], yt[i : i + chunk], use_reentrant=False)
                for i in range(0, len(xt), chunk)
            )
            return torch.stack(list(parts)).sum() / yt.numel()

        history = minimise(
            net.parameters(), loss, self.schedule, checkpoint=checkpoint, config=repr(self), data=[xt, yt]
        )
        envelope = ChannelEnvelope.of(x, margin=self.margin)
        return FittedFNO(
            net=net.eval(), modes=self.modes, runtime=rt, envelope=envelope, inputs=sx, outputs=sy, history=history
        )


@dataclass(frozen=True, slots=True, eq=False)
class FittedFNO:
    net: _FNONet
    modes: tuple[int, ...]
    runtime: Runtime
    envelope: ChannelEnvelope
    inputs: Standardiser
    outputs: Standardiser
    history: list[float]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """``(n, c_out, *grid)`` on any grid of the unit domain fine enough for the modes."""
        x = np.asarray(x, float)
        if x.ndim != 2 + len(self.modes):
            refuse(f'an FNO query of shape {x.shape} is invalid: it takes (n, c, *{len(self.modes)} grid axes)')
        if x.shape[1] != self.envelope.channels:
            refuse(f'an FNO query of shape {x.shape} is invalid: it takes {self.envelope.channels} input channels')
        _require_modes_fit(self.modes, x.shape[2:])
        self.envelope.require(_onto(x, self.envelope.shape), 'an FNO query (at the training nodes)')
        with torch.no_grad():
            return self.outputs.back(numpy(self.net(self.runtime.tensor(self.inputs.forward(x)))))
