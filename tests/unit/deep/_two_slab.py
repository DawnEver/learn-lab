"""A planted two-region 1-D problem: -(k u')' = 0 on (0, 1/2) and (1/2, 1), u(0) = 0, u(1) = 1.

One output column per region (``u_left``, ``u_right``) so a smooth network carries the jump in ``u'`` at the
interface: continuity ``u_left = u_right`` and flux ``k_left u_left' = k_right u_right'`` hold there. With
``k_left = 1`` and ``k_right = 4`` the solution is ``1.6 x`` on the left and ``1 - 0.4 (1 - x)`` on the right.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from learn_lab.deep.physics import Term, derivative
from learn_lab.validity import BoxEnvelope

INTERFACE = 0.5


def truth(x: np.ndarray, k_left: float, k_right: float) -> np.ndarray:
    """The planted ``u`` at ``(n, 1)`` points: the slopes ``a`` (left), ``b`` (right) obey continuity and flux."""
    b = 1.0 / (INTERFACE * k_right / k_left + (1.0 - INTERFACE))
    a = k_right * b / k_left
    return np.where(x[:, 0] < INTERFACE, a * x[:, 0], 1.0 - b * (1.0 - x[:, 0]))


@dataclass(frozen=True)
class TwoSlab:
    k_right_nominal: float = 1.0
    points: int = 64
    domain: BoxEnvelope = field(default_factory=lambda: BoxEnvelope(np.array([0.0]), np.array([1.0])))
    outputs: int = 2

    @property
    def parameters(self):
        return {'k_left': 1.0, 'k_right': self.k_right_nominal}

    def owns(self, points):
        """Column 0 owns the left slab, column 1 the right one (the interface point to the left)."""
        left = points[:, 0] <= INTERFACE
        return np.stack([left, ~left], axis=1)

    def collocation(self, runtime):
        r = torch.rand(self.points, 1, generator=runtime.generator(), device=runtime.torch_device)
        r = r.to(runtime.torch_dtype)
        return {
            'left': INTERFACE * r,
            'right': INTERFACE + (1.0 - INTERFACE) * r,
            'interface': runtime.tensor(np.array([[INTERFACE]])),
            'ends': runtime.tensor(np.array([[0.0], [1.0]])),
        }

    def residuals(self, field, points, parameters):
        def flux(column, x, k):
            return k * derivative(field(x)[:, column], x)[:, 0]

        xl, xr, xi = points['left'], points['right'], points['interface']
        ends = field(points['ends'])
        at = field(xi)
        return {
            'left': Term(derivative(flux(0, xl, parameters['k_left']), xl)[:, 0], scale=1.0),
            'right': Term(derivative(flux(1, xr, parameters['k_right']), xr)[:, 0], scale=1.0),
            'ends': Term(torch.stack([ends[0, 0], ends[1, 1] - 1.0]), scale=1.0),
            'continuity': Term(at[:, 0] - at[:, 1], scale=1.0),
            'flux': Term(flux(0, xi, parameters['k_left']) - flux(1, xi, parameters['k_right']), scale=1.0),
        }


@dataclass(frozen=True)
class EveryColumnEverywhere(TwoSlab):
    """The planted control: a provider claiming every column owns every point -- the defect's behaviour."""

    def owns(self, points):
        return np.ones((points.shape[0], self.outputs), bool)
