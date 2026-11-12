"""A provider's ``features(x)`` replaces the normalised coordinates as the net's input.

Periodic features make every net output periodic by construction, the seam an exactly periodic hard ansatz needs.
The provider without features is the planted control: the same fit's outputs at the two ends differ.
"""

from dataclasses import dataclass

import numpy as np
import torch

from learn_lab.deep.fit import Schedule
from learn_lab.deep.physics import PhysicsInformedField
from learn_lab.deep.runtime import Runtime

from ._poisson import Poisson

CPU = Runtime(device='cpu')  # float64
SCHEDULE = Schedule(adam_steps=20, learning_rate=1e-3, lbfgs_steps=0)


@dataclass(frozen=True)
class PeriodicPoisson(Poisson):
    """The planted Poisson with features of period 1 = the box length."""

    def features(self, x: torch.Tensor) -> torch.Tensor:
        k = torch.arange(1, 4, dtype=x.dtype, device=x.device)
        phase = 2.0 * np.pi * x[:, :1] * k
        return torch.cat([torch.cos(phase), torch.sin(phase)], dim=1)


def _ends(provider) -> np.ndarray:
    fitted = PhysicsInformedField(hidden=(8, 8), schedule=SCHEDULE, runtime=CPU).fit(provider)
    return fitted.evaluate(np.array([[0.0], [1.0]]))[:, 0]


def test_periodic_features_make_the_net_outputs_periodic():
    ends = _ends(PeriodicPoisson())
    assert abs(ends[1] - ends[0]) <= 1e-12, ends


def test_without_features_the_ends_differ():
    ends = _ends(Poisson())
    assert abs(ends[1] - ends[0]) > 1e-6, ends
