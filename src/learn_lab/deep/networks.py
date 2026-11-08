"""Network bodies shared by the families: a plain MLP and a monotone scalar map.

Every module a family persists records the ``arguments`` it was built with, so an artifact rebuilds the same
architecture before loading its weights (:mod:`learn_lab.deep.artifact`).
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Final

import torch
from torch import nn

from learn_lab.errors import refuse

__all__ = ['ACTIVATIONS', 'MLP', 'MonotoneScalar']

#: The activations a persisted MLP may name.
ACTIVATIONS: Final = {'tanh': nn.Tanh, 'silu': nn.SiLU, 'gelu': nn.GELU}


class MLP(nn.Sequential):
    """``widths[0] -> ... -> widths[-1]``, *activation* between layers, linear output. Smooth by default (PINN)."""

    def __init__(self, widths: Sequence[int], *, activation: str = 'tanh') -> None:
        if activation not in ACTIVATIONS:
            refuse(f'an MLP activation is one of {list(ACTIVATIONS)}, got {activation!r}')
        widths = tuple(int(w) for w in widths)
        layers: list[nn.Module] = []
        for i, (a, b) in enumerate(itertools.pairwise(widths)):
            layers.append(nn.Linear(a, b))
            if i < len(widths) - 2:
                layers.append(ACTIVATIONS[activation]())
        super().__init__(*layers)
        self.arguments = {'widths': widths, 'activation': activation}


class MonotoneScalar(nn.Module):
    """``f: R -> R`` strictly increasing BY CONSTRUCTION, able to bend both ways.

    ``f(x) = a x + sum_j c_j S(w_j x + b_j) + sum_j d_j R(v_j x + e_j)`` with ``S = tanh`` (saturating, concave
    knees) and ``R = softplus`` (ramps, convex knees); ``a, c, w, d, v > 0`` via softplus, so
    ``f'(x) >= a > 0`` everywhere -- a learned law whose derivative a Newton solver can trust.
    ``odd=True`` pairs every unit with its mirror (``S(wx+b) + S(wx-b)``, ``R(vx+e) - R(-vx+e)``), making
    ``f(-x) = -f(x)`` exactly (B-H through the origin). Inputs are expected scaled to about ``[-1, 1]``;
    the ramps' knees start spread over that interval.
    """

    def __init__(self, hidden: int = 16, *, odd: bool = True) -> None:
        super().__init__()
        self.arguments = {'hidden': hidden, 'odd': odd}
        self.odd = odd
        knots = torch.linspace(0.0, 1.0, hidden) if odd else torch.linspace(-1.0, 1.0, hidden)
        self.raw_slope = nn.Parameter(torch.tensor(-2.0))
        self.raw_s_weight = nn.Parameter(torch.full((hidden,), 1.0))
        self.raw_s_scale = nn.Parameter(torch.full((hidden,), -3.0))
        self.s_bias = nn.Parameter(-knots.clone())
        self.raw_r_weight = nn.Parameter(torch.full((hidden,), 2.0))
        self.raw_r_scale = nn.Parameter(torch.full((hidden,), -3.0))
        self.r_bias = nn.Parameter(-2.0 * knots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sp = nn.functional.softplus
        x1 = x.unsqueeze(-1)
        w, v = sp(self.raw_s_weight), sp(self.raw_r_weight)
        if self.odd:
            s = torch.tanh(w * x1 + self.s_bias) + torch.tanh(w * x1 - self.s_bias)
            r = sp(v * x1 + self.r_bias) - sp(-v * x1 + self.r_bias)
        else:
            s, r = torch.tanh(w * x1 + self.s_bias), sp(v * x1 + self.r_bias)
        return sp(self.raw_slope) * x + (sp(self.raw_s_scale) * s).sum(-1) + (sp(self.raw_r_scale) * r).sum(-1)
