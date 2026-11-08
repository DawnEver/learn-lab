"""Grid -> grid field model: a small U-Net mapping ``(n, c_in, H, W) -> (n, c_out, H, W)``.

Channels are standardised on the training set. The envelope is the input channels' value box AND the grid
shape the model was trained on: another shape is INVALID (the convolutions would run, the answer would not
be the trained one), a value outside the box is OUT_OF_DOMAIN. ``levels`` halvings need ``H`` and ``W``
divisible by ``2**levels``; at the bottom every cell sees the whole grid when ``H / 2**levels <= 3``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from learn_lab.deep.fit import Checkpoint, Schedule, minimise
from learn_lab.deep.runtime import Runtime, numpy
from learn_lab.deep.standardise import Standardiser
from learn_lab.errors import refuse, require_finite
from learn_lab.validity import ChannelEnvelope

__all__ = ['FieldModel', 'FittedFieldModel', 'UNet']


def _block(a: int, b: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GELU(), nn.Conv2d(b, b, 3, padding=1), nn.GELU())


class UNet(nn.Module):
    """``levels`` halvings with doubled widths, skip connections by concatenation, a 1x1 output head."""

    def __init__(self, c_in: int, c_out: int, *, width: int, levels: int) -> None:
        super().__init__()
        self.arguments = {'c_in': c_in, 'c_out': c_out, 'width': width, 'levels': levels}
        widths = [width * 2**i for i in range(levels + 1)]
        self.down = nn.ModuleList([_block(c_in, widths[0]), *(_block(a, b) for a, b in itertools.pairwise(widths))])
        self.up = nn.ModuleList(nn.ConvTranspose2d(b, a, 2, stride=2) for a, b in itertools.pairwise(widths))
        self.merge = nn.ModuleList(_block(2 * a, a) for a in widths[:-1])
        self.head = nn.Conv2d(widths[0], c_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, block in enumerate(self.down):
            x = block(x if i == 0 else nn.functional.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, merge in zip(reversed(self.up), reversed(self.merge), strict=False):
            x = merge(torch.cat([skips.pop(), up(x)], dim=1))
        return self.head(x)


@dataclass(frozen=True, slots=True)
class FieldModel:
    width: int = 16
    levels: int = 2
    schedule: Schedule = field(default_factory=lambda: Schedule(adam_steps=3000, learning_rate=2e-3))
    runtime: Runtime = field(default_factory=lambda: Runtime(dtype='float32'))
    margin: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray, *, checkpoint: Checkpoint | None = None) -> FittedFieldModel:
        x = require_finite(np.asarray(x, float), 'the input fields')
        y = require_finite(np.asarray(y, float), 'the output fields')
        if x.ndim != 4 or y.ndim != 4 or x.shape[0] != y.shape[0] or x.shape[2:] != y.shape[2:] or not len(x):
            refuse(f'a field model is fitted on (n, c_in, H, W) -> (n, c_out, H, W), got {x.shape} -> {y.shape}')
        if any(side % 2**self.levels for side in x.shape[2:]):
            refuse(f'{self.levels} levels need H and W divisible by {2**self.levels}, got {x.shape[2:]}')
        rt = self.runtime
        rt.seeded()
        sx, sy = Standardiser.of(x), Standardiser.of(y)
        xt, yt = rt.tensor(sx.forward(x)), rt.tensor(sy.forward(y))
        net = UNet(x.shape[1], y.shape[1], width=self.width, levels=self.levels).to(rt.torch_device, rt.torch_dtype)

        def loss() -> torch.Tensor:
            return ((net(xt) - yt) ** 2).mean()

        history = minimise(net.parameters(), loss, self.schedule, checkpoint=checkpoint, config=repr(self))
        envelope = ChannelEnvelope.of(x, margin=self.margin)
        return FittedFieldModel(net=net.eval(), runtime=rt, envelope=envelope, inputs=sx, outputs=sy, history=history)


@dataclass(frozen=True, slots=True, eq=False)
class FittedFieldModel:
    net: UNet
    runtime: Runtime
    envelope: ChannelEnvelope
    inputs: Standardiser
    outputs: Standardiser
    history: list[float]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """``(n, c_out, H, W)`` for in-domain ``(n, c_in, H, W)``; a refusal names the first stranger."""
        x = self.envelope.require(x, 'a field-model query')
        with torch.no_grad():
            return self.outputs.back(numpy(self.net(self.runtime.tensor(self.inputs.forward(x)))))
