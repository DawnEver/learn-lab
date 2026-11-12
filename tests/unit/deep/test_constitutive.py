"""A learned monotone law, and the HYBRID: a classical Newton solve handed the law as two callables."""

import numpy as np
import pytest

from learn_lab.deep.constitutive import MonotoneLaw
from learn_lab.deep.fit import Schedule
from learn_lab.deep.runtime import Runtime
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
#: Pinned so every host fits the same arithmetic: 'auto' is CUDA on a GPU box and CPU on the forge.
CPU64 = Runtime(device='cpu', dtype='float64')


def _low_field_error(loss):
    x = np.concatenate([-DECADES[::-1], DECADES])
    law = MonotoneLaw(loss=loss, input_warp='log', schedule=POLISHED, runtime=CPU64).fit(x, _steel(x))
    return np.max(np.abs(law.value(LOW) - _steel(LOW)) / _steel(LOW)), law


def _worst_low_field(value) -> float:
    """Max relative error of *value* over the held-out low-field decade (dimensionless)."""
    return float(np.max(np.abs(value(LOW) - _steel(LOW)) / _steel(LOW)))


def test_a_relative_loss_holds_a_decades_spanning_law_within_one_percent_at_the_low_end():
    """``loss='relative'`` holds the low-field decade of a four-decade B(H) within 1 %.

    PLANTED CONTROL (deterministic): the same check refuses the fitted law with a 2 % error injected at the low end.
    The input is log-warped (``input_warp='log'``): unwarped, the low decade sits at ``x / max|x|`` in
    [1e-4, 1e-3] and where Adam stops decides the minimum L-BFGS polishes -- 0.21..1.76 % across Adam
    2500..3500 steps and learning rate 2.7e-3..3.3e-3 on Windows CPU, over 1 % on the Linux forge; warped,
    0.03..0.25 % over the same sweep.
    Measured 2026-11-12 (CPU float64, unwarped): plain MSE 2.3 % on Windows CPU but under 2 % and within
    4x of the relative fit on the Linux forge. An MSE fit is odd and monotone, hence linear near H = 0, and learns
    that slope from the mid range, so its low-end miss is small and platform-dependent: it cannot be a control.
    """
    _, law = _low_field_error('relative')
    assert _worst_low_field(law.value) < 0.01  # dimensionless: within 1 %
    assert _worst_low_field(lambda h: 1.02 * law.value(h)) > 0.01  # PLANTED: a 2 % low-end error is refused
    assert np.all(law.derivative(LOW) > 0)
    np.testing.assert_allclose(law.value(-LOW), -law.value(LOW), rtol=0, atol=1e-12)  # T: exact oddness
    fd = (law.value(LOW * (1 + 1e-6)) - law.value(LOW * (1 - 1e-6))) / (2e-6 * LOW)
    np.testing.assert_allclose(law.derivative(LOW), fd, rtol=1e-5, atol=0)  # T/(A/m): through the warp


def test_mse_is_the_default_loss_and_an_unknown_loss_is_refused():
    assert MonotoneLaw().loss == 'mse'
    with pytest.raises(Refused, match='loss'):
        MonotoneLaw(loss='huber').fit(np.array([1.0, 2.0]), np.array([1.0, 2.0]))


def test_linear_is_the_default_input_warp_and_an_unknown_warp_is_refused():
    assert MonotoneLaw().input_warp == 'linear'
    with pytest.raises(Refused, match='input_warp'):
        MonotoneLaw(input_warp='sqrt').fit(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
