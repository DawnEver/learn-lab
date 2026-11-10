"""PDE-FIND: the same sparse regression recovers Burgers' equation from a planted numerical solution."""

import numpy as np
import pytest

from learn_lab.equation import SparseRegression, pde_data, pde_library
from learn_lab.errors import Refused

NU = 0.1
TRUE = {'u': {'u*u_x': -1.0, 'u_xx': NU}}
BAR = 0.01  # relative error of each recovered coefficient (dimensionless); measured 9.9e-4 (u*u_x), 2.2e-3 (u_xx)


def _burgers(points=256, length=16.0, t_end=10.0, snapshots=101, dt=1e-3):
    """``u_t = -u u_x + nu u_xx`` on a periodic line, pseudo-spectral in x, RK4 in t: ``(t, x, u (T, N))``."""
    x = np.arange(points) * length / points - length / 2
    k = 2 * np.pi * np.fft.rfftfreq(points, length / points)

    def rate(v):
        u = np.fft.irfft(v, points)
        return -0.5j * k * np.fft.rfft(u * u) - NU * k**2 * v

    v, out = np.fft.rfft(np.exp(-((x + 2.0) ** 2))), []
    every = round(t_end / (snapshots - 1) / dt)
    for i in range((snapshots - 1) * every + 1):
        if i % every == 0:
            out.append(np.fft.irfft(v, points))
        k1 = rate(v)
        k2 = rate(v + dt / 2 * k1)
        k3 = rate(v + dt / 2 * k2)
        k4 = rate(v + dt * k3)
        v = v + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.linspace(0, t_end, snapshots), x, np.array(out)


@pytest.fixture(scope='module')
def burgers():
    return _burgers()


def _identify(t, x, u):
    states, rates = pde_data(u, t[1] - t[0], x[1] - x[0], order=3)
    library = pde_library(order=3, degree=2)
    return SparseRegression(threshold=0.05).fit(library, states, rates, targets=('u',))


def test_burgers_is_recovered_with_its_exact_support(burgers):
    law = _identify(*burgers)
    assert {name: set(terms) for name, terms in law.terms().items()} == {k: set(v) for k, v in TRUE.items()}
    for term, value in TRUE['u'].items():
        assert law.terms()['u'][term] == pytest.approx(value, rel=BAR, abs=0)


def test_noisy_data_fails_the_support_check(burgers):
    """PLANTED CONTROL: 1 % noise ruins finite-difference derivatives; the support check must say no."""
    t, x, u = burgers
    noisy = u + 0.01 * np.abs(u).max() * np.random.default_rng(0).standard_normal(u.shape)
    law = _identify(t, x, noisy)
    assert {name: set(terms) for name, terms in law.terms().items()} != {k: set(v) for k, v in TRUE.items()}


def test_the_library_names_its_terms():
    assert pde_library(order=2, degree=1).names == ('1', 'u', 'u_x', 'u*u_x', 'u_xx', 'u*u_xx')


def test_a_grid_too_small_for_the_stencil_is_refused():
    with pytest.raises(Refused, match='stencil'):
        pde_data(np.zeros((10, 4)), 0.1, 0.1, order=3)


def test_targets_name_known_variables():
    with pytest.raises(Refused, match='targets'):
        SparseRegression().fit(pde_library(order=1, degree=1), np.zeros((9, 2)), np.zeros((9, 1)), targets=('v',))
