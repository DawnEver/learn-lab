"""Conditional field generation: a denoising diffusion model (DDPM) over ``(c, H, W)`` fields given a condition.

``fit(conditions (n, k), fields (n, c, H, W))`` trains a :class:`~learn_lab.deep.field.UNet` to predict the noise
``eps`` in ``z_t = sqrt(abar_t) y + sqrt(1 - abar_t) eps`` at a uniformly drawn step ``t`` of a linear schedule of
``noise_steps`` steps (:func:`linear_betas`); the net sees the noisy field, the standardised condition as constant
channels and ``t / T`` as one more. :meth:`FittedDiffusionField.sample` runs the ancestral sampler from pure
noise (posterior variance ``beta_t (1 - abar_{t-1}) / (1 - abar_t)``, or ``beta_t`` on request) and returns field
samples; :meth:`~FittedDiffusionField.moments` their mean and variance per cell.

The envelope is the conditions' box: a condition outside it is refused (OUT_OF_DOMAIN), another width is INVALID.
Fields are standardised per channel. The loss draws fresh noise every step, so it is minimised by Adam only, on a
cosine-annealed rate by default (a constant rate leaves the weights jittering on that noise): an
L-BFGS polish of a stochastic loss is refused. ``chunk`` bounds peak memory as in :mod:`learn_lab.deep.field`;
``levels`` halvings need ``H`` and ``W`` divisible by ``2**levels``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, get_args

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint as recomputed

from learn_lab.deep.field import UNet
from learn_lab.deep.fit import Checkpoint, Schedule, minimise
from learn_lab.deep.runtime import Runtime, numpy
from learn_lab.deep.standardise import Standardiser
from learn_lab.errors import refuse, require_finite
from learn_lab.validity import BoxEnvelope

__all__ = ['SAMPLER_VARIANCES', 'DiffusionField', 'FittedDiffusionField', 'SamplerVariance', 'linear_betas']

SamplerVariance = Literal['posterior', 'beta']
SAMPLER_VARIANCES: tuple[str, ...] = get_args(SamplerVariance)


def linear_betas(steps: int) -> np.ndarray:
    """DDPM's linear schedule, ``1e-4 .. 0.02`` over 1000 steps, rescaled to *steps* (Ho et al.; capped at 0.5).

    The total noise ``sum beta`` is kept, so the last step is pure noise at any *steps*. A cosine schedule ends
    with ``beta -> 1``: the sampler divides by ``sqrt(1 - beta)`` and amplifies the net's error there -- measured
    on the planted field, its sample mean missed by 28 T on a 1 T field.
    """
    scale = 1000 / steps
    return np.minimum(np.linspace(1e-4 * scale, 0.02 * scale, steps), 0.5)


def _inputs(z: torch.Tensor, condition: torch.Tensor, t: torch.Tensor, steps: int) -> torch.Tensor:
    """``(n, c + k + 1, H, W)``: the noisy field, the condition broadcast, and ``t / steps``."""
    n, _, h, w = z.shape
    planes = torch.cat([condition, (t.to(z.dtype) / steps)[:, None]], dim=1)
    return torch.cat([z, planes[:, :, None, None].expand(n, planes.shape[1], h, w)], dim=1)


@dataclass(frozen=True, slots=True)
class DiffusionField:
    width: int = 16
    levels: int = 1
    noise_steps: int = 100
    schedule: Schedule = field(default_factory=lambda: Schedule(adam_steps=3000, learning_rate=2e-3, decay='cosine'))
    runtime: Runtime = field(default_factory=lambda: Runtime(dtype='float32'))
    margin: float = 0.0
    chunk: int | None = 32

    def __post_init__(self) -> None:
        if self.chunk is not None and (isinstance(self.chunk, bool) or self.chunk < 1):
            refuse(f'a chunk is a sample count >= 1 (or None for one pass), got {self.chunk!r}')
        if self.noise_steps < 2:
            refuse(f'a diffusion takes >= 2 noise steps, got {self.noise_steps}')
        if self.schedule.lbfgs_steps:
            refuse('a diffusion loss draws fresh noise every step: it is minimised by Adam only, lbfgs_steps=0')

    def fit(
        self, conditions: np.ndarray, fields: np.ndarray, *, checkpoint: Checkpoint | None = None
    ) -> FittedDiffusionField:
        c = require_finite(np.asarray(conditions, float), 'the conditions')
        y = require_finite(np.asarray(fields, float), 'the fields')
        if c.ndim != 2 or y.ndim != 4 or c.shape[0] != y.shape[0] or not len(c):
            refuse(f'a diffusion is fitted on conditions (n, k) and fields (n, c, H, W), got {c.shape}, {y.shape}')
        if any(side % 2**self.levels for side in y.shape[2:]):
            refuse(f'{self.levels} levels need H and W divisible by {2**self.levels}, got {y.shape[2:]}')
        rt = self.runtime
        rt.seeded()
        sc, sy = Standardiser.of(c), Standardiser.of(y)
        ct, yt = rt.tensor(sc.forward(c)), rt.tensor(sy.forward(y))
        betas = linear_betas(self.noise_steps)
        abar = rt.tensor(np.cumprod(1 - betas))
        steps, chunk = self.noise_steps, self.chunk
        net = UNet(y.shape[1] + c.shape[1] + 1, y.shape[1], width=self.width, levels=self.levels)
        net = net.to(rt.torch_device, rt.torch_dtype)

        def squared(z: torch.Tensor, cc: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
            return ((net(_inputs(z, cc, t, steps)) - eps) ** 2).sum()

        def loss() -> torch.Tensor:
            t = torch.randint(0, steps, (len(yt),), device=yt.device)
            eps = torch.randn_like(yt)
            a = abar[t][:, None, None, None]
            z = a.sqrt() * yt + (1 - a).sqrt() * eps
            if chunk is None or chunk >= len(yt):
                return squared(z, ct, t, eps) / yt.numel()
            parts = (
                recomputed(
                    squared,
                    z[i : i + chunk],
                    ct[i : i + chunk],
                    t[i : i + chunk],
                    eps[i : i + chunk],
                    use_reentrant=False,
                )
                for i in range(0, len(yt), chunk)
            )
            return torch.stack(list(parts)).sum() / yt.numel()

        history = minimise(
            net.parameters(), loss, self.schedule, checkpoint=checkpoint, config=repr(self), data=[ct, yt]
        )
        return FittedDiffusionField(
            net=net.eval(),
            runtime=rt,
            envelope=BoxEnvelope.of(c, margin=self.margin),
            conditions=sc,
            outputs=sy,
            shape=tuple(int(s) for s in y.shape[1:]),
            betas=betas,
            history=history,
        )


@dataclass(frozen=True, slots=True, eq=False)
class FittedDiffusionField:
    net: UNet
    runtime: Runtime
    envelope: BoxEnvelope
    conditions: Standardiser
    outputs: Standardiser
    shape: tuple[int, ...]
    betas: np.ndarray
    history: list[float]

    def sample(
        self, conditions: np.ndarray, count: int, *, seed: int = 0, variance: SamplerVariance = 'posterior'
    ) -> np.ndarray:
        """``(m, count, c, H, W)`` fields drawn for each in-domain condition; one *seed* gives the same draws.

        *variance* is the reverse step's noise: ``'posterior'`` ``beta_t (1 - abar_{t-1}) / (1 - abar_t)`` (exact when
        the data is one point, the lower bound) or ``'beta'`` ``beta_t`` (exact when the data is N(0, I), the upper).
        """
        c = self.envelope.require(conditions, 'a diffusion condition')
        if isinstance(count, bool) or count < 1:
            refuse(f'a sample count is >= 1, got {count!r}')
        if variance not in SAMPLER_VARIANCES:
            refuse(f'a sampler variance is one of {SAMPLER_VARIANCES}, got {variance!r}')
        rt = self.runtime
        generator = torch.Generator(device=rt.torch_device).manual_seed(seed)
        steps = len(self.betas)
        betas = rt.tensor(self.betas)
        alphas = 1 - betas
        abar = torch.cumprod(alphas, 0)
        ct = rt.tensor(np.repeat(self.conditions.forward(c), count, axis=0))
        z = torch.randn((len(ct), *self.shape), generator=generator, device=rt.torch_device, dtype=rt.torch_dtype)
        with torch.no_grad():
            for step in reversed(range(steps)):
                t = torch.full((len(ct),), step, device=rt.torch_device)
                eps = self.net(_inputs(z, ct, t, steps))
                z = (z - betas[step] / (1 - abar[step]).sqrt() * eps) / alphas[step].sqrt()
                if step:
                    noise = torch.randn(z.shape, generator=generator, device=rt.torch_device, dtype=rt.torch_dtype)
                    shrink = 1 if variance == 'beta' else (1 - abar[step - 1]) / (1 - abar[step])
                    z = z + (betas[step] * shrink).sqrt() * noise
        return self.outputs.back(numpy(z)).reshape(len(c), count, *self.shape)

    def moments(
        self, conditions: np.ndarray, count: int, *, seed: int = 0, variance: SamplerVariance = 'posterior'
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per condition, the ``(c, H, W)`` mean and unbiased variance of *count* samples."""
        if isinstance(count, bool) or count < 2:
            refuse(f'a sample variance needs a count >= 2, got {count!r}')
        draws = self.sample(conditions, count, seed=seed, variance=variance)
        return draws.mean(1), draws.var(1, ddof=1)
