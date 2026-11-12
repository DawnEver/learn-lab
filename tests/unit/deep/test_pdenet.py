"""PDE-Net recovers a planted linear PDE ``u_t = a u_xx + b u_x`` on a periodic line from snapshot pairs.

The data is exact (each Fourier mode evolves in closed form). The moment constraint fixes each stencil's leading
moments, so the stencil IS a derivative and its coefficient is the PDE's; without it a coefficient and its kernel's
scale trade off and the coefficient is not identified. Pinned to CPU float64.
"""

import numpy as np
import pytest

from learn_lab.deep.fit import Schedule
from learn_lab.deep.pdenet import PDENet
from learn_lab.deep.runtime import Runtime
from learn_lab.errors import Refused

ALPHA, BETA = 0.5, 1.0  # m^2/s, m/s
POINTS, LENGTH, DT = 64, 2 * np.pi, 1e-3
BAR = 0.01  # relative error of each recovered coefficient, dimensionless
# Measured 2026-11-12 (CPU float64): constrained 2.7e-4 (u_x), 1.2e-4 (u_xx); unconstrained 0.78 (u_x), 0.30 (u_xx).
CPU64 = Runtime(device='cpu', dtype='float64')
SCHEDULE = Schedule(adam_steps=1500, learning_rate=1e-2)


def _pairs(count, seed):
    """``u(0)`` from random modes k <= 4; ``u(DT)`` from the exact mode evolution ``exp((-a k^2 + i b k) DT)``."""
    rng = np.random.default_rng(seed)
    k = np.fft.rfftfreq(POINTS, LENGTH / POINTS) * 2 * np.pi
    spectrum = np.zeros((count, k.size), complex)
    spectrum[:, 1:5] = (rng.standard_normal((count, 4)) + 1j * rng.standard_normal((count, 4))) * POINTS / 4
    spectrum[:, 0] = rng.standard_normal(count) * POINTS / 4
    before = np.fft.irfft(spectrum, POINTS)
    after = np.fft.irfft(spectrum * np.exp((-ALPHA * k**2 + 1j * BETA * k) * DT), POINTS)
    return before, after


@pytest.fixture(scope='module')
def pairs():
    return _pairs(64, 0)


def _relative(terms):
    return abs(terms.get('u_xx', 0.0) / ALPHA - 1), abs(terms.get('u_x', 0.0) / BETA - 1)


def test_the_constrained_stencils_recover_the_coefficients(pairs):
    fitted = PDENet(schedule=SCHEDULE, runtime=CPU64).fit(*pairs, dt=DT, dx=LENGTH / POINTS)
    terms = fitted.terms()
    assert max(_relative(terms)) < BAR
    assert set(terms) == {'u_x', 'u_xx'}  # the sparse response keeps only the planted terms


def test_unconstrained_kernels_miss_the_coefficients(pairs):
    """PLANTED CONTROL: the same fit with every moment free does not identify the coefficients."""
    fitted = PDENet(constrained=False, schedule=SCHEDULE, runtime=CPU64).fit(*pairs, dt=DT, dx=LENGTH / POINTS)
    terms = fitted.terms()
    assert max(_relative(terms)) > BAR


def test_the_learned_rate_advances_a_held_out_state(pairs):
    fitted = PDENet(schedule=SCHEDULE, runtime=CPU64).fit(*pairs, dt=DT, dx=LENGTH / POINTS)
    before, after = _pairs(4, 1)
    predicted = fitted.advance(before, DT)
    assert np.linalg.norm(predicted - after) < 0.05 * np.linalg.norm(after - before)
    stencils = fitted.stencils()
    assert stencils.shape == (3, 5)


def test_a_state_outside_the_envelope_is_refused(pairs):
    fitted = PDENet(schedule=Schedule(adam_steps=3, learning_rate=1e-2), runtime=CPU64).fit(*pairs, dt=DT, dx=0.1)
    fitted.rate(pairs[0][:1])  # PLANTED CONTROL: inside, it answers
    with pytest.raises(Refused):
        fitted.rate(pairs[0][:1] + 100.0)


def test_a_stencil_too_short_for_its_order_is_refused():
    with pytest.raises(Refused):
        PDENet(order=2, kernel=3, accuracy=2)
