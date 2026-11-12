"""A planted 2-D two-slab Poisson with a 1:4 jump, as an energy: -div(k grad u) = f on [0, 1]^2, u = 0 on all sides.

``k = 1`` for ``x < 1/2`` and ``4`` beyond, one output column per slab. The planted solution is manufactured from the
1-D jump: ``u = U(x) sin(pi y)``, ``U`` the piecewise quadratic of :mod:`._jump` (continuous, with continuous flux
``k U'``), so ``f = (8 + k pi^2 U(x)) sin(pi y)`` per slab and the interface conditions hold exactly.

``essential='penalty'``: the four sides and the interface continuity are penalised :class:`Term` s on Gauss-Legendre
lines. ``essential='hard'``: the ansatz ``u_s = d(x, y) (T + (x - 1/2) C_s)``, ``d = x(1-x) y(1-y)`` vanishing on the
boundary and the trace ``T`` shared by both slabs, so every essential condition holds by construction.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from learn_lab.deep.physics import Energy, Term, derivative, gauss_legendre
from learn_lab.validity import BoxEnvelope

from . import _jump

INTERFACE = _jump.INTERFACE
K = (_jump.K_LEFT, _jump.K_RIGHT)


def _profile(x: torch.Tensor) -> torch.Tensor:
    """The 1-D jump's ``U(x)`` as a tensor, per point."""
    a, c, d = _jump._coefficients()
    f = _jump.SOURCE
    return torch.where(x < INTERFACE, -f * x**2 / (2 * K[0]) + a * x, -f * x**2 / (2 * K[1]) + c * x + d)


def truth(x: np.ndarray) -> np.ndarray:
    return _jump.truth(x[:, :1]) * np.sin(np.pi * x[:, 1])


@dataclass(frozen=True)
class Slab2D:
    essential: str = 'penalty'  # or 'hard'
    order: int = 8
    cells: int = 2
    domain: BoxEnvelope = field(default_factory=lambda: BoxEnvelope(np.zeros(2), np.ones(2)))
    outputs: int = 2
    latent: int = 3  # hard: the shared trace and one correction per slab
    parameters: dict = field(default_factory=lambda: {'k_left': K[0], 'k_right': K[1]})
    observables: dict = field(default_factory=lambda: {'u': 2})

    def observe(self, name, field, points):
        return field(points)

    def owns(self, name, points):
        left = points[:, 0] <= INTERFACE
        return np.stack([left, ~left], axis=1)

    def ansatz(self, x, raw):
        d = x[:, 0] * (1 - x[:, 0]) * x[:, 1] * (1 - x[:, 1])
        s = x[:, 0] - INTERFACE
        return d[:, None] * (raw[:, :1] + s[:, None] * raw[:, 1:])

    def collocation(self, runtime):
        o, c = self.order, self.cells
        left, wl = gauss_legendre([0.0, 0.0], [INTERFACE, 1.0], o, runtime, cells=c)
        right, wr = gauss_legendre([INTERFACE, 0.0], [1.0, 1.0], o, runtime, cells=c)
        line = gauss_legendre([0.0], [1.0], o, runtime, cells=2 * c)[0][:, 0]

        def at(fixed, axis):
            p = torch.full((line.shape[0], 2), fixed, dtype=line.dtype, device=line.device)
            p[:, axis] = line
            return p

        sides = torch.cat([at(0.0, 1), at(1.0, 1), at(0.0, 0), at(1.0, 0)])
        return {
            'left': left,
            'right': right,
            'w_left': wl,
            'w_right': wr,
            'sides': sides,
            'interface': at(INTERFACE, 1),
        }

    def residuals(self, field, points, parameters):  # pragma: no cover - an energy provider
        raise NotImplementedError

    def energy(self, field, points, parameters):
        def part(column, name, k, kv):
            x = points[name]
            u = field(x)[:, column]
            g = derivative(u, x)
            f = (_jump.SOURCE + kv * np.pi**2 * _profile(x[:, 0])) * torch.sin(np.pi * x[:, 1])
            density = 0.5 * parameters[k] * (g**2).sum(1) - f.detach() * u
            return Energy(density, weights=points[f'w_{name}'].detach(), scale=_jump.SOURCE)

        parts = {'left': part(0, 'left', 'k_left', K[0]), 'right': part(1, 'right', 'k_right', K[1])}
        if self.essential == 'hard':
            return parts
        sides, at = field(points['sides']), field(points['interface'])
        owned = torch.where(points['sides'][:, 0] <= INTERFACE, sides[:, 0], sides[:, 1])
        return {**parts, 'sides': Term(owned, scale=1.0), 'continuity': Term(at[:, 0] - at[:, 1], scale=1.0)}
