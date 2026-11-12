"""Deep Ritz with essential conditions as a HARD ansatz instead of penalised terms.

The same Gauss-Legendre quadrature, schedule and network judge both; the penalised run is the planted control. A
penalised essential condition biases the minimiser by O(1 / penalty) and stiffens it; the ansatz holds the ends,
sides and interface continuity exactly, so only the energy is minimised.
"""

import numpy as np
import pytest
import torch

from learn_lab.deep.fit import Schedule
from learn_lab.deep.physics import PhysicsInformedField, Ritz
from learn_lab.deep.runtime import Runtime
from learn_lab.errors import Refused

from . import _jump, _slab2d

SCHEDULE = Schedule(adam_steps=3000, learning_rate=1e-3, lbfgs_steps=500)
PENALTY = 1e3  # the penalised control at its best measured penalty (test_physics_ritz_quadrature.py)
#: max |u - truth| in the unit of u (peak ~0.6). Measured, seed 0, Gauss + L-BFGS, same net per problem:
#: 1-D jump (64 points per region, 24x24 tanh): hard 6.14e-5, penalised 2.47e-3 (40x);
#: 2-D two-slab (order 8, 2 cells per axis per slab, 16x16 tanh): hard 9.59e-5, penalised 8.77e-2 (914x).
BAR_1D = 2e-4
BAR_2D = 3e-4
CPU = Runtime(device='cpu')  # float64
CONTROL = 10.0  # the penalised error is at least this many times the hard one


def _error(provider, hidden, ansatz, grid, truth):
    fit = PhysicsInformedField(hidden=hidden, schedule=SCHEDULE, runtime=CPU, ritz=Ritz(penalty=PENALTY), ansatz=ansatz)
    u = fit.fit(provider).evaluate(grid)
    owned = np.where(grid[:, 0] <= _jump.INTERFACE, u[:, 0], u[:, 1])
    return float(np.max(np.abs(owned - truth(grid))))


def test_a_hard_ansatz_beats_the_penalised_essential_conditions_on_the_1d_jump():
    x = np.linspace(0, 1, 201)[:, None]
    hard = _error(_jump.Jump('gauss', points=64, order=16, essential='hard'), (24, 24), True, x, _jump.truth)
    penalised = _error(_jump.Jump('gauss', points=64, order=16), (24, 24), False, x, _jump.truth)
    report = f'1-D hard {hard:.3e}, penalised {penalised:.3e}'
    assert hard == pytest.approx(0.0, abs=BAR_1D), report
    assert penalised > CONTROL * hard, report


def test_a_hard_ansatz_beats_the_penalised_essential_conditions_on_the_2d_two_slab():
    g = np.linspace(0, 1, 41)
    xy = np.stack([a.ravel() for a in np.meshgrid(g, g, indexing='ij')], axis=1)
    hard = _error(_slab2d.Slab2D('hard'), (16, 16), True, xy, _slab2d.truth)
    penalised = _error(_slab2d.Slab2D('penalty'), (16, 16), False, xy, _slab2d.truth)
    report = f'2-D hard {hard:.3e}, penalised {penalised:.3e}'
    assert hard == pytest.approx(0.0, abs=BAR_2D), report
    assert penalised > CONTROL * hard, report


def test_the_ansatz_holds_the_essential_conditions_exactly_whatever_the_net():
    rt = CPU
    p = _slab2d.Slab2D('hard')
    raw = torch.randn(4, 3, dtype=rt.torch_dtype)
    edge = rt.tensor(np.array([[0.0, 0.3], [1.0, 0.7], [0.2, 0.0], [0.8, 1.0]]))
    assert float(p.ansatz(edge, raw).abs().max()) == 0.0
    at = rt.tensor(np.array([[0.5, 0.1], [0.5, 0.4], [0.5, 0.6], [0.5, 0.9]]))
    u = p.ansatz(at, raw)
    assert float((u[:, 0] - u[:, 1]).abs().max()) == 0.0
    # control: off the interface the two columns differ, so the continuity check could fail
    off = p.ansatz(at + rt.tensor(np.array([0.1, 0.0])), raw)
    assert float((off[:, 0] - off[:, 1]).abs().max()) > 1e-6


def test_a_hard_ansatz_fit_refuses_a_provider_without_one():
    fit = PhysicsInformedField(hidden=(4,), schedule=Schedule(adam_steps=1, lbfgs_steps=0), runtime=CPU, ansatz=True)
    with pytest.raises(Refused, match='ansatz'):
        fit.fit(_jump.Jump('gauss', latent=0))
