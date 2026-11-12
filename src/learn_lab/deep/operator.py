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
- :class:`TransformerOperator` -- self-attention over irregular points.
- :class:`Transolver` -- Physics-Attention: points grouped into learned slices, attention among the slices;
  cost linear in the points, any point set (Wu et al., ICML 2024; Transolver++ switches).
- :class:`GraphOperator` -- kernel-integral message passing over a :class:`~learn_lab.representation.PointGraph`
  (mesh nodes, see :func:`~learn_lab.representation.mesh_graph`); discretisation-invariant: fit on one point
  set, evaluate on another.
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
from learn_lab.representation import PointGraph
from learn_lab.validity import BoxEnvelope, ChannelEnvelope

__all__ = [
    'FNO',
    'DeepONet',
    'FittedDeepONet',
    'FittedFNO',
    'FittedGraphOperator',
    'FittedTransformerOperator',
    'FittedTransolver',
    'GraphOperator',
    'SpectralConv',
    'TransformerOperator',
    'Transolver',
]


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
    last = x.ndim - 1  # a non-negative axis: ONNX export writes a negative one into Transpose's perm verbatim
    return layer(x.movedim(1, last)).movedim(last, 1)


class _FNONet(nn.Module):
    def __init__(self, c_in: int, c_out: int, *, modes: Sequence[int], width: int, layers: int) -> None:
        super().__init__()
        self.arguments = {'c_in': c_in, 'c_out': c_out, 'modes': tuple(modes), 'width': width, 'layers': layers}
        self.lift = nn.Linear(c_in + len(modes), width)
        self.spectral = nn.ModuleList(SpectralConv(width, width, modes) for _ in range(layers))
        self.mix = nn.ModuleList(nn.Linear(width, width) for _ in range(layers))
        self.head = MLP([width, 2 * width, c_out], activation='gelu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # arange cast to x's dtype, not linspace(dtype=): ONNX export types a linspace double regardless.
        axes = (torch.arange(n, device=x.device).to(x.dtype) / max(n - 1, 1) for n in x.shape[2:])
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


_KERNELS = ('softmax', 'linear')


def _causal_rank(coords: torch.Tensor) -> torch.Tensor:
    """Each point's place in time: the stable order of the first coordinate (ties keep the index order)."""
    order = torch.argsort(coords[:, 0], stable=True)
    return torch.empty_like(order).scatter_(0, order, torch.arange(order.numel(), device=order.device))


class _Attention(nn.Module):
    """Multi-head self-attention over the points; ``linear`` is the ``elu + 1`` kernel, memory O(p)."""

    def __init__(self, width: int, heads: int, kernel: str, causal: bool) -> None:
        super().__init__()
        self.heads, self.kernel, self.causal = heads, kernel, causal
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)

    def forward(self, h: torch.Tensor, rank: torch.Tensor) -> torch.Tensor:
        n, p, w = h.shape
        q, k, v = (t.reshape(n, p, self.heads, -1).transpose(1, 2) for t in self.qkv(h).chunk(3, dim=-1))
        if self.kernel == 'softmax':
            mask = rank[None, :] <= rank[:, None] if self.causal else None  # [query, key]: key not after query
            mixed = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            q, k = nn.functional.elu(q) + 1.0, nn.functional.elu(k) + 1.0
            if self.causal:  # prefix sums in time order, then back to the points' order
                order = torch.argsort(rank)
                q, k, v = q[:, :, order], k[:, :, order], v[:, :, order]
                kv = torch.cumsum(k.unsqueeze(-1) * v.unsqueeze(-2), dim=2)
                norm = torch.einsum('nhpd,nhpd->nhp', q, k.cumsum(2))
                mixed = (torch.einsum('nhpd,nhpde->nhpe', q, kv) / norm[..., None])[:, :, rank]
            else:
                kv = torch.einsum('nhpd,nhpe->nhde', k, v)
                norm = torch.einsum('nhpd,nhd->nhp', q, k.sum(2))
                mixed = torch.einsum('nhpd,nhde->nhpe', q, kv) / norm[..., None]
        return self.out(mixed.transpose(1, 2).reshape(n, p, w))


class _Block(nn.Module):
    def __init__(self, width: int, heads: int, kernel: str, causal: bool) -> None:
        super().__init__()
        self.attend, self.norm_attend = _Attention(width, heads, kernel, causal), nn.LayerNorm(width)
        self.mlp, self.norm_mlp = MLP([width, 2 * width, width], activation='gelu'), nn.LayerNorm(width)

    def forward(self, h: torch.Tensor, rank: torch.Tensor) -> torch.Tensor:
        h = h + self.attend(self.norm_attend(h), rank)
        return h + self.mlp(self.norm_mlp(h))


class _TransformerNet(nn.Module):
    def __init__(
        self, c_in: int, c_out: int, *, width: int, heads: int, layers: int, kernel: str, causal: bool, lower, upper
    ) -> None:
        super().__init__()
        self.arguments = {
            'c_in': c_in,
            'c_out': c_out,
            'width': width,
            'heads': heads,
            'layers': layers,
            'kernel': kernel,
            'causal': causal,
            'lower': lower,
            'upper': upper,
        }
        self.lift = nn.Linear(c_in + len(lower), width)
        self.blocks = nn.ModuleList(_Block(width, heads, kernel, causal) for _ in range(layers))
        self.head = MLP([width, 2 * width, c_out], activation='gelu')
        span = np.where(upper > lower, upper - lower, 1.0)
        self.register_buffer('lower', torch.as_tensor(lower, dtype=torch.float64))
        self.register_buffer('span', torch.as_tensor(span, dtype=torch.float64))

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """``x (n, c_in, p)``, ``coords (p, d)`` -> ``(n, c_out, p)``."""
        rank = _causal_rank(coords)
        position = (2.0 * (coords - self.lower) / self.span - 1.0).expand(x.shape[0], -1, -1)
        h = self.lift(torch.cat([x.transpose(1, 2), position], dim=-1))
        for block in self.blocks:
            h = block(h, rank)
        return self.head(h).transpose(1, 2)


@dataclass(frozen=True, slots=True)
class TransformerOperator:
    """Self-attention over irregular points: a token is one point's input channels and coordinates.

    ``kernel='softmax'`` is exact attention (memory O(p^2) per head); ``'linear'`` is the ``elu + 1`` kernel
    (memory O(p)). ``causal=True`` makes a history operator: a point attends only to points whose FIRST
    coordinate is not later (ties ordered by index), so its output never reads the future. A query may use any
    points inside the coordinates' training box; its inputs are refused outside the per-channel value box.
    """

    width: int = 32
    heads: int = 4
    layers: int = 2
    kernel: str = 'softmax'
    causal: bool = False
    schedule: Schedule = field(default_factory=lambda: Schedule(adam_steps=2000, learning_rate=3e-3))
    runtime: Runtime = field(default_factory=lambda: Runtime(dtype='float32'))
    margin: float = 0.0
    chunk: int | None = 32

    def __post_init__(self) -> None:
        if self.chunk is not None and (isinstance(self.chunk, bool) or self.chunk < 1):
            refuse(f'a chunk is a sample count >= 1 (or None for one pass), got {self.chunk!r}')
        if self.kernel not in _KERNELS:
            refuse(f'a transformer operator kernel is one of {list(_KERNELS)}, got {self.kernel!r}')
        if self.heads < 1 or self.layers < 1 or self.width % self.heads:
            refuse(f'width {self.width} must split evenly into heads {self.heads} (>= 1), with layers >= 1')

    def fit(
        self, x: np.ndarray, coords: np.ndarray, y: np.ndarray, *, checkpoint: Checkpoint | None = None
    ) -> FittedTransformerOperator:
        """``x (n, c_in, p)`` at the points ``coords (p, d)`` -> ``y (n, c_out, p)`` at the same points."""
        x = require_finite(np.asarray(x, float), 'the input functions')
        c = require_finite(np.asarray(coords, float), 'the coordinates')
        y = require_finite(np.asarray(y, float), 'the output functions')
        if x.ndim != 3 or y.ndim != 3 or c.ndim != 2 or x.shape[::2] != y.shape[::2] or x.shape[2] != len(c):
            refuse(
                f'a transformer operator is fitted on x (n, c_in, p), coords (p, d), y (n, c_out, p); '
                f'got {x.shape}, {c.shape}, {y.shape}'
            )
        rt = self.runtime
        rt.seeded()
        sx, sy = Standardiser.of(x), Standardiser.of(y)
        box = BoxEnvelope.of(c, margin=self.margin)
        net = _TransformerNet(
            x.shape[1],
            y.shape[1],
            width=self.width,
            heads=self.heads,
            layers=self.layers,
            kernel=self.kernel,
            causal=self.causal,
            lower=box.lower,
            upper=box.upper,
        ).to(rt.torch_device, rt.torch_dtype)
        xt, ct, yt = rt.tensor(sx.forward(x)), rt.tensor(c), rt.tensor(sy.forward(y))
        chunk = self.chunk

        def squared(xc: torch.Tensor, yc: torch.Tensor) -> torch.Tensor:
            return ((net(xc, ct) - yc) ** 2).sum()

        def loss() -> torch.Tensor:
            if chunk is None or chunk >= len(xt):
                return squared(xt, yt) / yt.numel()
            parts = (
                recomputed(squared, xt[i : i + chunk], yt[i : i + chunk], use_reentrant=False)
                for i in range(0, len(xt), chunk)
            )
            return torch.stack(list(parts)).sum() / yt.numel()

        history = minimise(
            net.parameters(), loss, self.schedule, checkpoint=checkpoint, config=repr(self), data=[xt, ct, yt]
        )
        envelope = ChannelEnvelope.of(x, fixed_shape=False, margin=self.margin)
        return FittedTransformerOperator(
            net=net.eval(), runtime=rt, envelope=envelope, points=box, inputs=sx, outputs=sy, history=history
        )


@dataclass(frozen=True, slots=True, eq=False)
class FittedTransformerOperator:
    net: _TransformerNet
    runtime: Runtime
    envelope: ChannelEnvelope
    points: BoxEnvelope
    inputs: Standardiser
    outputs: Standardiser
    history: list[float]

    def predict(self, x: np.ndarray, coords: np.ndarray) -> np.ndarray:
        """``(n, c_out, p)``: each input function's image at its ``p`` points ``coords (p, d)``."""
        x = self.envelope.require(np.asarray(x, float), 'a transformer operator input function')
        c = self.points.require(coords, 'a transformer operator point')
        if x.ndim != 3 or x.shape[2] != len(c):
            refuse(f'a transformer operator query of shape {x.shape} is invalid at {len(c)} points: (n, c_in, p)')
        rt = self.runtime
        with torch.no_grad():
            return self.outputs.back(numpy(self.net(rt.tensor(self.inputs.forward(x)), rt.tensor(c))))


class _PhysicsAttention(nn.Module):
    """Physics-Attention: points -> ``slices`` learned soft groups per head -> attention among the slice tokens.

    Per head, ``w = softmax(Linear(x_i) / tau)`` over the slices (``(p, M)``, each row sums to 1); the token
    ``z_j = sum_i w_ij v_i / sum_i w_ij``; ``z' = softmax(q k^T / sqrt(C_head)) v`` among the M tokens; the
    point's output ``sum_j w_ij z'_j``. Cost O(p M C + M^2 C): linear in the points. ``tau`` is a learnable
    per-head temperature; ``adaptive_temperature`` adds a per-point ``Linear(x_i)`` to it and ``gumbel`` adds
    Gumbel noise to the slice logits while training (Transolver++'s Ada-Temp and Rep-Slice). ``last`` keeps the
    slice weights and token scores of the latest forward (detached) for inspection.
    """

    def __init__(
        self, width: int, heads: int, slices: int, temperature: float, adaptive_temperature: bool, gumbel: bool
    ) -> None:
        super().__init__()
        self.heads, self.gumbel = heads, gumbel
        head = width // heads
        self.features = nn.Linear(width, width)  # the values each point contributes to its slices
        self.mid = nn.Linear(width, width)  # what the slice assignment reads
        self.assign = nn.Linear(head, slices)
        nn.init.orthogonal_(self.assign.weight)
        self.temperature = nn.Parameter(torch.full((1, heads, 1, 1), float(temperature)))
        self.local = nn.Linear(head, 1) if adaptive_temperature else None
        self.qkv = nn.Linear(head, 3 * head, bias=False)
        self.out = nn.Linear(width, width)
        self.last: tuple[torch.Tensor, torch.Tensor] | None = None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        n, p, w = h.shape
        v = self.features(h).reshape(n, p, self.heads, -1).transpose(1, 2)  # (n, H, p, C)
        mid = self.mid(h).reshape(n, p, self.heads, -1).transpose(1, 2)
        logits, tau = self.assign(mid), self.temperature
        if self.local is not None:
            tau = tau + self.local(mid)
        if self.gumbel and self.training:
            u = torch.rand_like(logits).clamp(1e-12, 1.0 - 1e-12)
            logits = logits - torch.log(-torch.log(u))
        weights = torch.softmax(logits / tau.clamp_min(1e-2), dim=-1)  # (n, H, p, M)
        tokens = torch.einsum('nhpm,nhpc->nhmc', weights, v) / (weights.sum(2)[..., None] + 1e-5)
        q, k, t = self.qkv(tokens).chunk(3, dim=-1)
        scores = torch.softmax(q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5, dim=-1)  # (n, H, M, M)
        mixed = torch.einsum('nhpm,nhmc->nhpc', weights, scores @ t)
        self.last = (weights.detach(), scores.detach())
        return self.out(mixed.transpose(1, 2).reshape(n, p, w))


class _TransolverBlock(nn.Module):
    def __init__(self, width: int, mlp_ratio: int, attention: _PhysicsAttention) -> None:
        super().__init__()
        self.attend, self.norm_attend = attention, nn.LayerNorm(width)
        self.mlp, self.norm_mlp = MLP([width, mlp_ratio * width, width], activation='gelu'), nn.LayerNorm(width)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = h + self.attend(self.norm_attend(h))
        return h + self.mlp(self.norm_mlp(h))


class _TransolverNet(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        *,
        width: int,
        heads: int,
        layers: int,
        slices: int,
        mlp_ratio: int,
        temperature: float,
        adaptive_temperature: bool,
        gumbel: bool,
        reference: int | None,
        lower,
        upper,
    ) -> None:
        super().__init__()
        self.arguments = {
            'c_in': c_in,
            'c_out': c_out,
            'width': width,
            'heads': heads,
            'layers': layers,
            'slices': slices,
            'mlp_ratio': mlp_ratio,
            'temperature': temperature,
            'adaptive_temperature': adaptive_temperature,
            'gumbel': gumbel,
            'reference': reference,
            'lower': lower,
            'upper': upper,
        }
        d = len(lower)
        span = np.where(upper > lower, upper - lower, 1.0)
        self.register_buffer('lower', torch.as_tensor(lower, dtype=torch.float64))
        self.register_buffer('span', torch.as_tensor(span, dtype=torch.float64))
        nodes = np.zeros((0, d))
        if reference is not None:  # the unified position: distances to a reference grid of the training box
            axis = np.linspace(-1.0, 1.0, reference)
            nodes = np.stack(np.meshgrid(*[axis] * d, indexing='ij'), axis=-1).reshape(-1, d)
        self.register_buffer('nodes', torch.as_tensor(nodes, dtype=torch.float64))
        self.lift = MLP([c_in + d + len(nodes), 2 * width, width], activation='gelu')
        self.blocks = nn.ModuleList(
            _TransolverBlock(
                width, mlp_ratio, _PhysicsAttention(width, heads, slices, temperature, adaptive_temperature, gumbel)
            )
            for _ in range(layers)
        )
        self.norm = nn.LayerNorm(width)
        self.head = MLP([width, width, c_out], activation='gelu')

    def attentions(self) -> list[_PhysicsAttention]:
        return [block.attend for block in self.blocks]

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """``x (n, c_in, p)``, ``coords (p, d)`` or ``(n, p, d)`` -> ``(n, c_out, p)``."""
        z = (2.0 * (coords - self.lower) / self.span - 1.0).expand(x.shape[0], -1, -1)
        parts = [x.transpose(1, 2), z]
        if len(self.nodes):
            parts.append(torch.cdist(z, self.nodes.expand(x.shape[0], -1, -1)))
        h = self.lift(torch.cat(parts, dim=-1))
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h)).transpose(1, 2)


def _point_sets(x: np.ndarray, coords: np.ndarray, what: str) -> None:
    shared = x.ndim == 3 and coords.ndim == 2 and x.shape[2] == len(coords)
    own = x.ndim == 3 and coords.ndim == 3 and coords.shape[:2] == (x.shape[0], x.shape[2])
    if not (shared or own):
        refuse(f'{what} of shape {x.shape} is invalid at points {coords.shape}: (n, c_in, p) at (p, d) or (n, p, d)')


@dataclass(frozen=True, slots=True)
class Transolver:
    """Transolver (Wu et al., ICML 2024): Physics-Attention over any point set, cost linear in the points.

    A token is one point's input channels and box-normalised coordinates (plus, with ``reference=r``, its
    distances to an ``r^d`` grid of the training box: the paper's unified position). Each of ``layers``
    pre-norm blocks groups the points into ``slices`` learned soft slices per head, attends among the slice
    tokens and scatters back (:class:`_PhysicsAttention`), then applies a ``mlp_ratio`` MLP. ``temperature`` is
    the learnable slice temperature's start; ``adaptive_temperature`` and ``gumbel`` switch on Transolver++'s
    per-point temperature and Gumbel slice reparameterisation (noise in training only; prediction is
    deterministic). Points are shared ``(p, d)`` or per sample ``(n, p, d)``, and their number may differ
    between fit and predict. A query's points are refused outside the training box; its inputs outside the
    per-channel value box.
    """

    width: int = 64
    heads: int = 4
    layers: int = 4
    slices: int = 32
    mlp_ratio: int = 1
    temperature: float = 0.5
    adaptive_temperature: bool = False
    gumbel: bool = False
    reference: int | None = None
    schedule: Schedule = field(default_factory=lambda: Schedule(adam_steps=2000, learning_rate=1e-3))
    runtime: Runtime = field(default_factory=lambda: Runtime(dtype='float32'))
    margin: float = 0.0
    chunk: int | None = 32

    def __post_init__(self) -> None:
        if self.chunk is not None and (isinstance(self.chunk, bool) or self.chunk < 1):
            refuse(f'a chunk is a sample count >= 1 (or None for one pass), got {self.chunk!r}')
        if self.heads < 1 or self.layers < 1 or self.width % self.heads:
            refuse(f'width {self.width} must split evenly into heads {self.heads} (>= 1), with layers >= 1')
        if self.slices < 1 or self.mlp_ratio < 1 or not self.temperature > 0:
            refuse(f'a Transolver has slices >= 1, mlp_ratio >= 1 and temperature > 0, got {self}')
        if self.reference is not None and self.reference < 2:
            refuse(f'a unified-position reference grid has >= 2 nodes per axis, got {self.reference}')

    def fit(
        self, x: np.ndarray, coords: np.ndarray, y: np.ndarray, *, checkpoint: Checkpoint | None = None
    ) -> FittedTransolver:
        """``x (n, c_in, p)`` at ``coords (p, d)`` or ``(n, p, d)`` -> ``y (n, c_out, p)`` at the same points."""
        x = require_finite(np.asarray(x, float), 'the input functions')
        c = require_finite(np.asarray(coords, float), 'the coordinates')
        y = require_finite(np.asarray(y, float), 'the output functions')
        _point_sets(x, c, 'a Transolver fit')
        if y.ndim != 3 or y.shape[::2] != x.shape[::2]:
            refuse(f'a Transolver is fitted on x (n, c_in, p) and y (n, c_out, p); got {x.shape}, {y.shape}')
        rt = self.runtime
        rt.seeded()
        sx, sy = Standardiser.of(x), Standardiser.of(y)
        box = BoxEnvelope.of(c.reshape(-1, c.shape[-1]), margin=self.margin)
        net = _TransolverNet(
            x.shape[1],
            y.shape[1],
            width=self.width,
            heads=self.heads,
            layers=self.layers,
            slices=self.slices,
            mlp_ratio=self.mlp_ratio,
            temperature=self.temperature,
            adaptive_temperature=self.adaptive_temperature,
            gumbel=self.gumbel,
            reference=self.reference,
            lower=box.lower,
            upper=box.upper,
        ).to(rt.torch_device, rt.torch_dtype)
        xt, ct, yt = rt.tensor(sx.forward(x)), rt.tensor(c), rt.tensor(sy.forward(y))
        chunk, own = self.chunk, c.ndim == 3

        def squared(xc: torch.Tensor, cc: torch.Tensor, yc: torch.Tensor) -> torch.Tensor:
            return ((net(xc, cc) - yc) ** 2).sum()

        def loss() -> torch.Tensor:
            if chunk is None or chunk >= len(xt):
                return squared(xt, ct, yt) / yt.numel()
            parts = (
                recomputed(
                    squared, xt[i : i + chunk], ct[i : i + chunk] if own else ct, yt[i : i + chunk], use_reentrant=False
                )
                for i in range(0, len(xt), chunk)
            )
            return torch.stack(list(parts)).sum() / yt.numel()

        history = minimise(
            net.parameters(), loss, self.schedule, checkpoint=checkpoint, config=repr(self), data=[xt, ct, yt]
        )
        envelope = ChannelEnvelope.of(x, fixed_shape=False, margin=self.margin)
        return FittedTransolver(
            net=net.eval(), runtime=rt, envelope=envelope, points=box, inputs=sx, outputs=sy, history=history
        )


@dataclass(frozen=True, slots=True, eq=False)
class FittedTransolver:
    net: _TransolverNet
    runtime: Runtime
    envelope: ChannelEnvelope
    points: BoxEnvelope
    inputs: Standardiser
    outputs: Standardiser
    history: list[float]

    def predict(self, x: np.ndarray, coords: np.ndarray) -> np.ndarray:
        """``(n, c_out, p)``: each input function's image at its points ``coords (p, d)`` or ``(n, p, d)``."""
        x = self.envelope.require(np.asarray(x, float), 'a Transolver input function')
        c = np.asarray(coords, float)
        _point_sets(x, c, 'a Transolver query')
        self.points.require(c.reshape(-1, c.shape[-1]), 'a Transolver point')
        rt = self.runtime
        with torch.no_grad():
            return self.outputs.back(numpy(self.net(rt.tensor(self.inputs.forward(x)), rt.tensor(c))))


class _GraphOperatorNet(nn.Module):
    def __init__(
        self, c_in: int, c_out: int, *, dimension: int, width: int, layers: int, kernel_width: int, lower, upper
    ) -> None:
        super().__init__()
        self.arguments = {
            'c_in': c_in,
            'c_out': c_out,
            'dimension': dimension,
            'width': width,
            'layers': layers,
            'kernel_width': kernel_width,
            'lower': lower,
            'upper': upper,
        }
        self.width = width
        self.lift = MLP([c_in + dimension, width, width], activation='gelu')
        self.kernels = nn.ModuleList(
            MLP([3 * dimension + 1, kernel_width, kernel_width, width * width], activation='gelu')
            for _ in range(layers)
        )
        self.local = nn.ModuleList(nn.Linear(width, width) for _ in range(layers))
        self.project = MLP([width, width, c_out], activation='gelu')
        span = np.where(upper > lower, upper - lower, 1.0)
        self.register_buffer('lower', torch.as_tensor(lower, dtype=torch.float64))
        self.register_buffer('span', torch.as_tensor(span, dtype=torch.float64))

    def forward(
        self, x: torch.Tensor, points: torch.Tensor, edges: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """``x (n, c_in, N)`` on ``points (N, d)``, ``edges (2, E)``, quadrature ``weights (E,)`` -> ``(n, c_out, N)``."""
        source, target = edges
        z = 2.0 * (points - self.lower) / self.span - 1.0
        offset = z[source] - z[target]
        geometry = torch.cat([z[target], z[source], offset, offset.norm(dim=1, keepdim=True)], dim=1)
        h = self.lift(torch.cat([x.transpose(1, 2), z.expand(x.shape[0], -1, -1)], dim=-1))
        for i, (kernel, local) in enumerate(zip(self.kernels, self.local, strict=True)):
            k = kernel(geometry).reshape(-1, self.width, self.width) * weights[:, None, None]
            message = torch.einsum('eij,nej->nei', k, h[:, source])
            h = local(h) + torch.zeros_like(h).index_add_(1, target, message)
            if i < len(self.kernels) - 1:
                h = nn.functional.gelu(h)
        return self.project(h).transpose(1, 2)


@dataclass(frozen=True, slots=True)
class GraphOperator:
    """A graph neural operator: kernel-integral layers over a :class:`~learn_lab.representation.PointGraph`.

    Each layer is ``h_i <- W h_i + sum_j weight_ij K(x_i, x_j) h_j`` with ``K`` a ``width x width`` matrix from
    an MLP of the two (box-normalised) positions and their offset; the sum is the graph's quadrature of the
    kernel integral, so the trained weights hold on ANY point set of the domain: fit on one mesh, evaluate on
    another (a finer one included). A query's points are refused outside the training points' box and its
    input values outside the per-channel value box.
    """

    width: int = 16
    layers: int = 2
    kernel_width: int = 32
    schedule: Schedule = field(default_factory=lambda: Schedule(adam_steps=2000, learning_rate=3e-3))
    runtime: Runtime = field(default_factory=lambda: Runtime(dtype='float32'))
    margin: float = 0.0

    def __post_init__(self) -> None:
        if min(self.width, self.layers, self.kernel_width) < 1:
            refuse(f'a graph operator has width, layers and kernel_width >= 1, got {self}')

    def fit(
        self, x: np.ndarray, graph: PointGraph, y: np.ndarray, *, checkpoint: Checkpoint | None = None
    ) -> FittedGraphOperator:
        """``x (n, c_in, N)`` on the ``N`` points of *graph* -> ``y (n, c_out, N)`` on the same points."""
        graph = graph.require_valid('the training graph')
        x = require_finite(np.asarray(x, float), 'the input functions')
        y = require_finite(np.asarray(y, float), 'the output functions')
        nodes = len(graph.points)
        if x.ndim != 3 or y.ndim != 3 or x.shape[0] != y.shape[0] or x.shape[2] != nodes or y.shape[2] != nodes:
            refuse(
                f'a graph operator is fitted on x (n, c_in, N), y (n, c_out, N) with N = {nodes}; '
                f'got {x.shape}, {y.shape}'
            )
        rt = self.runtime
        rt.seeded()
        sx, sy = Standardiser.of(x), Standardiser.of(y)
        box = BoxEnvelope.of(graph.points, margin=self.margin)
        net = _GraphOperatorNet(
            x.shape[1],
            y.shape[1],
            dimension=graph.points.shape[1],
            width=self.width,
            layers=self.layers,
            kernel_width=self.kernel_width,
            lower=box.lower,
            upper=box.upper,
        ).to(rt.torch_device, rt.torch_dtype)
        xt, yt = rt.tensor(sx.forward(x)), rt.tensor(sy.forward(y))
        structure = _graph_tensors(rt, graph)

        def loss() -> torch.Tensor:
            return ((net(xt, *structure) - yt) ** 2).mean()

        history = minimise(
            net.parameters(), loss, self.schedule, checkpoint=checkpoint, config=repr(self), data=[xt, yt, *structure]
        )
        envelope = ChannelEnvelope.of(x, fixed_shape=False, margin=self.margin)
        return FittedGraphOperator(
            net=net.eval(), runtime=rt, envelope=envelope, points=box, inputs=sx, outputs=sy, history=history
        )


def _graph_tensors(rt: Runtime, graph: PointGraph) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    edges = torch.as_tensor(np.asarray(graph.edges, np.int64), device=rt.torch_device)
    return rt.tensor(graph.points), edges, rt.tensor(graph.weights)


@dataclass(frozen=True, slots=True, eq=False)
class FittedGraphOperator:
    net: _GraphOperatorNet
    runtime: Runtime
    envelope: ChannelEnvelope
    points: BoxEnvelope
    inputs: Standardiser
    outputs: Standardiser
    history: list[float]

    def predict(self, x: np.ndarray, graph: PointGraph) -> np.ndarray:
        """``(n, c_out, N)``: each input function's image on the ``N`` points of *graph* (any point set)."""
        graph = graph.require_valid('a graph operator query graph')
        x = self.envelope.require(np.asarray(x, float), 'a graph operator input function')
        self.points.require(graph.points, 'a graph operator point')
        if x.ndim != 3 or x.shape[2] != len(graph.points):
            refuse(f'a graph operator query of shape {x.shape} is invalid on {len(graph.points)} points: (n, c_in, N)')
        rt = self.runtime
        with torch.no_grad():
            return self.outputs.back(numpy(self.net(rt.tensor(self.inputs.forward(x)), *_graph_tensors(rt, graph))))
