"""A planted 1-D problem: -k u'' = f on (0, 1), u(0) = u(1) = 0; u = sin(pi x) when f = k pi^2 sin(pi x)."""

from dataclasses import dataclass, field

import numpy as np
import torch

from learn_lab.deep.physics import Term, derivative
from learn_lab.validity import BoxEnvelope


@dataclass(frozen=True)
class Poisson:
    k_nominal: float = 1.0
    k_source: float = 1.0  # the k the source was built with: the truth an inverse fit must find
    points: int = 128
    domain: BoxEnvelope = field(default_factory=lambda: BoxEnvelope(np.array([0.0]), np.array([1.0])))
    outputs: int = 1

    @property
    def parameters(self):
        return {'k': self.k_nominal}

    @property
    def observables(self):
        return {'u': 1}

    def observe(self, name, field, points):
        return field(points)

    def owns(self, name, points):
        return np.ones((points.shape[0], 1), bool)

    def collocation(self, runtime):
        interior = torch.rand(self.points, 1, generator=runtime.generator(), device=runtime.torch_device)
        return {'interior': interior.to(runtime.torch_dtype), 'boundary': runtime.tensor(np.array([[0.0], [1.0]]))}

    def residuals(self, field, points, parameters):
        x = points['interior']
        u = field(x)[:, 0]
        u_xx = derivative(derivative(u, x)[:, 0], x)[:, 0]
        f = self.k_source * np.pi**2 * torch.sin(np.pi * x[:, 0])
        return {
            'pde': Term(-parameters['k'] * u_xx - f, scale=float(self.k_source * np.pi**2)),
            'boundary': Term(field(points['boundary'])[:, 0], scale=1.0),
        }
