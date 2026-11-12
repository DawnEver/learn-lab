"""A planted 1-D Poisson with a material jump, as an energy: -(k u')' = f on (0, 1/2) and (1/2, 1), u(0) = u(1) = 0.

One column per region (``u_left``, ``u_right``); continuity ``u_left = u_right`` at the interface and both ends are
essential :class:`Term` s, the flux balance ``k_left u_left' = k_right u_right'`` is the energy's own stationarity.
The exact piecewise solution is quadratic on each side (:func:`truth`). ``quadrature`` picks the rule: ``'monte_carlo'``
(uniform draws, ``measure * mean``) or ``'gauss'`` (tensor Gauss-Legendre per region, the interface a cell boundary).
``essential`` picks how the ends and continuity hold: ``'penalty'`` (the :class:`Term` s above) or ``'hard'`` (the
ansatz ``u_s = x (1 - x) (T + (x - 1/2) C_s)`` over ``latent`` raw columns ``T, C_left, C_right``: zero at both ends,
and the trace ``T`` shared by both columns so continuity is exact).
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from learn_lab.deep.physics import Energy, Term, derivative, gauss_legendre
from learn_lab.validity import BoxEnvelope

INTERFACE = 0.5
K_LEFT, K_RIGHT, SOURCE = 1.0, 4.0, 8.0


def _coefficients() -> tuple[float, float, float]:
    """``u_l = -f x^2 / 2k_l + a x``; ``u_r = -f x^2 / 2k_r + c x + d``, ``u_r(1) = 0``, continuity and flux at s."""
    s, kl, kr, f = INTERFACE, K_LEFT, K_RIGHT, SOURCE
    # unknowns (a, c): flux kl a = kr c; continuity -f s^2/2kl + a s = -f s^2/2kr + c (s - 1) + f/2kr
    m = np.array([[kl, -kr], [s, 1.0 - s]])
    rhs = np.array([0.0, f * s**2 / (2 * kl) - f * s**2 / (2 * kr) + f / (2 * kr)])
    a, c = np.linalg.solve(m, rhs)
    return a, c, f / (2 * kr) - c


def truth(x: np.ndarray) -> np.ndarray:
    a, c, d = _coefficients()
    x = x[:, 0]
    return np.where(x < INTERFACE, -SOURCE * x**2 / (2 * K_LEFT) + a * x, -SOURCE * x**2 / (2 * K_RIGHT) + c * x + d)


@dataclass(frozen=True)
class Jump:
    quadrature: str = 'monte_carlo'
    points: int = 256  # Monte Carlo draws per region, or Gauss points per region (order * cells)
    order: int = 16
    essential: str = 'penalty'  # or 'hard'
    latent: int = 3
    domain: BoxEnvelope = field(default_factory=lambda: BoxEnvelope(np.array([0.0]), np.array([1.0])))
    outputs: int = 2
    parameters: dict = field(default_factory=lambda: {'k_left': K_LEFT, 'k_right': K_RIGHT})
    observables: dict = field(default_factory=lambda: {'u': 2})

    def observe(self, name, field, points):
        return field(points)

    def owns(self, name, points):
        left = points[:, 0] <= INTERFACE
        return np.stack([left, ~left], axis=1)

    def ansatz(self, x, raw):
        d = x[:, :1] * (1.0 - x[:, :1])
        return d * (raw[:, :1] + (x[:, :1] - INTERFACE) * raw[:, 1:])

    def collocation(self, runtime):
        fixed = {
            'interface': runtime.tensor(np.array([[INTERFACE]])),
            'ends': runtime.tensor(np.array([[0.0], [1.0]])),
        }
        if self.quadrature == 'gauss':
            cells = self.points // self.order
            left, wl = gauss_legendre([0.0], [INTERFACE], self.order, runtime, cells=cells)
            right, wr = gauss_legendre([INTERFACE], [1.0], self.order, runtime, cells=cells)
            return {**fixed, 'left': left, 'right': right, 'w_left': wl, 'w_right': wr}
        r = torch.rand(self.points, 1, generator=runtime.generator(), device=runtime.torch_device).to(
            runtime.torch_dtype
        )
        return {**fixed, 'left': INTERFACE * r, 'right': INTERFACE + (1.0 - INTERFACE) * r}

    def residuals(self, field, points, parameters):  # pragma: no cover - an energy provider
        raise NotImplementedError

    def energy(self, field, points, parameters):
        def part(column, name, k):
            x = points[name]
            u = field(x)[:, column]
            density = 0.5 * parameters[k] * derivative(u, x)[:, 0] ** 2 - SOURCE * u
            if self.quadrature == 'gauss':
                return Energy(density, weights=points[f'w_{name}'].detach(), scale=SOURCE)
            return Energy(density, measure=0.5, scale=SOURCE)

        parts = {'left': part(0, 'left', 'k_left'), 'right': part(1, 'right', 'k_right')}
        if self.essential == 'hard':
            return parts
        ends, at = field(points['ends']), field(points['interface'])
        return {
            **parts,
            'ends': Term(torch.stack([ends[0, 0], ends[1, 1]]), scale=1.0),
            'continuity': Term(at[:, 0] - at[:, 1], scale=1.0),
        }
