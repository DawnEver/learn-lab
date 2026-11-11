"""A learned monotone law, and the HYBRID: a classical Newton solve handed the law as two callables."""

import numpy as np
import pytest

from learn_lab.deep.constitutive import MonotoneLaw
from learn_lab.deep.fit import Schedule
from learn_lab.errors import Refused


def _law(b):
    """A saturating H(B) shape (A/m vs T): linear plus an exponential knee, odd and monotone."""
    return 400.0 * b + 30.0 * np.sinh(3.0 * b)


def _true_slope(b):
    return 400.0 + 90.0 * np.cosh(3.0 * b)


@pytest.fixture(scope='module')
def law():
    b = np.linspace(-1.8, 1.8, 61)
    return MonotoneLaw().fit(b, _law(b))


def test_the_law_reproduces_held_out_pairs_and_is_odd(law):
    b = np.linspace(-1.75, 1.75, 50)
    h = law.value(b)
    assert np.max(np.abs(h - _law(b))) < 0.01 * np.max(np.abs(_law(b)))  # A/m: 1 % of the range
    np.testing.assert_allclose(law.value(-b), -h, rtol=0, atol=1e-9)  # A/m: exact oddness


def test_the_derivative_is_positive_everywhere_and_matches_finite_differences(law):
    b = np.linspace(-1.7, 1.7, 41)
    d = law.derivative(b)
    assert np.all(d > 0)
    fd = (law.value(b + 1e-5) - law.value(b - 1e-5)) / 2e-5
    np.testing.assert_allclose(d, fd, rtol=1e-5, atol=1e-6)  # (A/m)/T


def test_a_query_beyond_the_measured_range_is_refused(law):
    with pytest.raises(Refused, match='out_of_domain'):
        law.value(np.array([2.5]))


N = 40


def _gradients(u):
    return np.diff(np.concatenate([[0.0], u, [0.0]])) * N


def _newton(flux, slope, source=3000.0, reach=1.75):
    """-(d/dx) q(du/dx) = s on (0, 1), u(0) = u(1) = 0 by finite volumes; q and dq/dg are callables.

    A step that would carry a gradient beyond ``reach`` (inside the law's measured range) is halved: the
    classical solver keeps its iterates where the law it was handed is valid.
    """
    h = 1.0 / N
    u = np.zeros(N - 1)
    for _ in range(100):
        g = _gradients(u)
        q, dq = flux(g), slope(g)
        r = -(q[1:] - q[:-1]) / h - source
        jac = (np.diag(dq[1:] + dq[:-1]) - np.diag(dq[1:-1], 1) - np.diag(dq[1:-1], -1)) / h**2
        step = np.linalg.solve(jac, -r)
        while np.max(np.abs(_gradients(u + step))) > reach:
            step = step / 2
        u += step
        if np.max(np.abs(step)) < 1e-12:
            return u
    msg = 'Newton did not converge'
    raise AssertionError(msg)


def test_a_classical_newton_solve_with_the_learned_law_matches_the_true_law(law):
    exact = _newton(_law, _true_slope)
    hybrid = _newton(law.value, law.derivative)
    assert np.max(np.abs(_gradients(exact))) > 1.0  # T: the solve reaches the knee, not the linear region
    assert np.max(np.abs(hybrid - exact)) < 0.01 * np.max(np.abs(exact))  # unit of u


def _steel(h):
    """A B(H) shape spanning four decades of H (A/m -> T): odd, monotone, saturating."""
    return 1.6 * np.tanh(h / 300.0) + 0.05 * np.arcsinh(h / 50.0) + 4e-6 * h


DECADES = np.logspace(1.0, 5.0, 60)
LOW = np.logspace(1.02, 2.0, 12)  # A/m: the low-field decade, held out
POLISHED = Schedule(adam_steps=3000, learning_rate=3e-3, lbfgs_steps=3000)


def _low_field_error(loss):
    x = np.concatenate([-DECADES[::-1], DECADES])
    law = MonotoneLaw(loss=loss, schedule=POLISHED).fit(x, _steel(x))
    return np.max(np.abs(law.value(LOW) - _steel(LOW)) / _steel(LOW)), law


def test_a_relative_loss_fits_a_decades_spanning_law_uniformly_where_mse_misses_the_low_end():
    mse, _ = _low_field_error('mse')
    relative, law = _low_field_error('relative')
    assert mse > 0.02  # PLANTED CONTROL: plain MSE misses the low-field decade by more than 2 %
    assert relative < 0.01  # dimensionless: the relative loss holds the low end within 1 %
    assert np.all(law.derivative(LOW) > 0)
    np.testing.assert_allclose(law.value(-LOW), -law.value(LOW), rtol=0, atol=1e-12)  # T: exact oddness


def test_mse_is_the_default_loss_and_an_unknown_loss_is_refused():
    assert MonotoneLaw().loss == 'mse'
    with pytest.raises(Refused, match='loss'):
        MonotoneLaw(loss='huber').fit(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
