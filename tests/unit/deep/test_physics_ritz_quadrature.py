"""Deep Ritz on a DETERMINISTIC quadrature: Gauss-Legendre points and weights make the energy integral exact for
polynomials up to the rule's order, so an L-BFGS polish minimises the true energy instead of one draw's noise.
"""

import numpy as np
import pytest
import torch

from learn_lab.deep.fit import Schedule
from learn_lab.deep.physics import Energy, PhysicsInformedField, Ritz, gauss_legendre
from learn_lab.deep.runtime import Runtime
from learn_lab.errors import Refused

from ._jump import Jump, truth

ADAM = Schedule(adam_steps=3000, learning_rate=1e-3, lbfgs_steps=0)
POLISHED = Schedule(adam_steps=3000, learning_rate=1e-3, lbfgs_steps=500)
PENALTY = 1e3  # measured, Gauss + L-BFGS: 2.11e-3 at 1e3, 4.30e-3 at 1e4 (stiffer), 6.50e-3 at 1e2 (ends slack)
#: max |u - truth| in the unit of u (peak ~0.6). Measured, 64 points per region, 24x24 tanh, seed 0:
#: Gauss (order 16, 4 cells) + L-BFGS 2.11e-3; Monte Carlo on one fixed draw + L-BFGS 1.97e-2 (it fits the draw);
#: Monte Carlo redrawn every 10 steps, Adam only 2.75e-1 (3000 Adam steps do not converge this problem, so the
#: Adam-only schedule that Monte Carlo forces is the binding limit); 256 points redrawn then polished 3.94e-2.
GAUSS_BAR = 4e-3


def _error(provider, schedule, every):
    fit = PhysicsInformedField(hidden=(24, 24), schedule=schedule, ritz=Ritz(penalty=PENALTY, every=every))
    fitted = fit.fit(provider)
    x = np.linspace(0, 1, 201)[:, None]
    u = fitted.evaluate(x)
    owned = np.where(x[:, 0] <= 0.5, u[:, 0], u[:, 1])
    return float(np.max(np.abs(owned - truth(x))))


def test_gauss_quadrature_lets_lbfgs_polish_the_true_energy_where_monte_carlo_overfits_its_draw():
    gauss = _error(Jump('gauss', points=64, order=16), POLISHED, every=0)
    mc_polished = _error(Jump('monte_carlo', points=64), POLISHED, every=0)
    mc_adam = _error(Jump('monte_carlo', points=64), ADAM, every=10)
    report = f'Gauss + L-BFGS {gauss:.3e}, MC + L-BFGS {mc_polished:.3e}, MC Adam only {mc_adam:.3e}'
    assert gauss == pytest.approx(0.0, abs=GAUSS_BAR), report
    # planted control: the same points, schedule and network, only the rule differs -- the polish fits the draw
    assert mc_polished > 5 * gauss, report
    assert mc_adam > 5 * gauss, report


def test_gauss_legendre_integrates_polynomials_to_its_order_exactly():
    rt = Runtime()
    x, w = gauss_legendre([0.0, 1.0], [2.0, 4.0], 3, rt, cells=2)
    assert x.shape == (36, 2) and w.shape == (36,)
    p = x[:, 0] ** 5 * x[:, 1] ** 5  # degree 5 per coordinate: exact for order 3
    exact = (2.0**6 / 6) * ((4.0**6 - 1.0) / 6)
    assert float((w * p).sum()) == pytest.approx(exact, rel=1e-12, abs=1e-9)
    assert float(w.sum()) == pytest.approx(6.0, rel=1e-12, abs=1e-12)
    # control: degree 6 in x0 is beyond order 3 on one cell, so the same rule is NOT exact there
    y, v = gauss_legendre([0.0], [2.0], 3, rt)
    assert abs(float((v * y[:, 0] ** 6).sum()) - 2.0**7 / 7) > 1e-3


def test_an_energy_takes_a_measure_or_weights_never_both_nor_neither():
    d = torch.ones(3)
    assert float(Energy(d, weights=torch.full((3,), 0.5), scale=1.0).loss()) == pytest.approx(1.5, abs=1e-12)
    with pytest.raises(Refused, match='measure or quadrature weights'):
        Energy(d, measure=1.0, weights=torch.ones(3), scale=1.0)
    with pytest.raises(Refused, match='measure or quadrature weights'):
        Energy(d, scale=1.0)
    with pytest.raises(Refused, match='one weight per point'):
        Energy(d, weights=torch.ones(2), scale=1.0)
