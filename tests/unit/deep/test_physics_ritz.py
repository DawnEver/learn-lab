"""Deep Ritz: the field minimises the provider's ENERGY; learn-lab owns the quadrature sampling and the loop."""

import numpy as np
import pytest
import torch

from learn_lab.deep.fit import Schedule
from learn_lab.deep.physics import Balance, Energy, PhysicsInformedField, Ritz, Term, derivative
from learn_lab.errors import Refused

from ._poisson import Poisson

#: Adam only, on a quadrature redrawn every 10 steps: an L-BFGS polish on one fixed Monte Carlo draw fits that
#: draw's quadrature error (measured, 1024 points: 7.5e-2 redrawn then polished, 2.0e-2 never redrawn, 1.1e-2 Adam
#: only). Deep Ritz's floor sits above the PINN's (2e-3 on this problem): the Monte Carlo integral of the energy
#: and the penalised ends (u(0) ~ 3e-3 at penalty 100) bound it.
FAST = Schedule(adam_steps=3000, learning_rate=1e-3, lbfgs_steps=0)
BAR = 2e-2  # max |u - sin(pi x)|, unit of u (peak 1); measured 1.06e-2


class PoissonEnergy(Poisson):
    """``-k u'' = f`` as the minimiser of ``int k u'^2 / 2 - f u``; the ends ``u = 0`` are an essential condition."""

    sign = 1.0  # +1 the true energy; -1 the planted control (the load enters with the wrong sign)

    def energy(self, field, points, parameters):
        x = points['interior']
        u = field(x)[:, 0]
        du = derivative(u, x)[:, 0]
        f = self.k_source * np.pi**2 * torch.sin(np.pi * x[:, 0])
        scale = float(self.k_source * np.pi**2)  # the energy's unit: k (pi u)^2 at |u| ~ 1
        return {
            'energy': Energy(0.5 * parameters['k'] * du**2 - self.sign * f * u, measure=1.0, scale=scale),
            'boundary': Term(field(points['boundary'])[:, 0], scale=1.0),
        }


class WrongSign(PoissonEnergy):
    sign = -1.0


def _error(provider, ritz):
    fitted = PhysicsInformedField(hidden=(24, 24), schedule=FAST, ritz=ritz).fit(provider)
    x = np.linspace(0, 1, 101)[:, None]
    return float(np.max(np.abs(fitted.evaluate(x)[:, 0] - np.sin(np.pi * x[:, 0]))))


def test_deep_ritz_minimises_the_planted_poisson_energy_to_its_solution():
    error = _error(PoissonEnergy(points=1024), Ritz(every=10))
    assert error < BAR, f'max|u - sin(pi x)| = {error:.3e}'


def test_planted_control_the_wrong_energy_misses_the_solution():
    assert _error(WrongSign(points=1024), Ritz(every=10)) > 1.0  # it finds -sin(pi x): max error 2


def test_ritz_refuses_a_provider_without_energy_a_moving_balance_or_a_bad_declaration():
    with pytest.raises(Refused, match='declares no energy'):
        PhysicsInformedField(schedule=Schedule(adam_steps=1), ritz=Ritz()).fit(Poisson())
    with pytest.raises(Refused, match='balance'):
        PhysicsInformedField(ritz=Ritz(), balance=Balance('gradient_norm', 'energy')).fit(PoissonEnergy())
    with pytest.raises(Refused, match='penalty'):
        Ritz(penalty=0.0)
    with pytest.raises(Refused, match='measure'):
        Energy(torch.zeros(1), measure=0.0, scale=1.0)
