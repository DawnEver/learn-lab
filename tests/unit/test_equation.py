"""Sparse identification recovers a planted law's support exactly and its coefficients within a bar."""

import numpy as np
import pytest

from learn_lab.equation import SparseRegression, polynomial_library, time_derivative
from learn_lab.errors import Refused

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0
TRUE = {
    'x': {'x': -SIGMA, 'y': SIGMA},
    'y': {'x': RHO, 'y': -1.0, 'x*z': -1.0},
    'z': {'x*y': 1.0, 'z': -BETA},
}


def _lorenz(s):
    x, y, z = s
    return np.array([SIGMA * (y - x), x * (RHO - z) - y, x * y - BETA * z])


def _trajectory(dt=1e-3, steps=10_000):
    s, out = np.array([-8.0, 7.0, 27.0]), []
    for _ in range(steps):
        out.append(s)
        k1 = _lorenz(s)
        k2 = _lorenz(s + dt / 2 * k1)
        k3 = _lorenz(s + dt / 2 * k2)
        k4 = _lorenz(s + dt * k3)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.array(out), dt


@pytest.fixture(scope='module')
def data():
    states, dt = _trajectory()
    return states, time_derivative(states, dt)


def test_the_lorenz_law_is_recovered_with_its_exact_support(data):
    states, rates = data
    law = SparseRegression(threshold=0.5).fit(polynomial_library(('x', 'y', 'z'), degree=2), states, rates)
    assert {name: set(terms) for name, terms in law.terms().items()} == {k: set(v) for k, v in TRUE.items()}
    for name, terms in TRUE.items():
        for term, value in terms.items():
            assert law.terms()[name][term] == pytest.approx(
                value, abs=1e-3 * abs(value)
            )  # unit of the coefficient; measured 3.7e-4 relative
    # PLANTED CONTROL: without thresholding, the same data gives a dense law -- the support test can say no.
    dense = SparseRegression(threshold=0.0).fit(polynomial_library(('x', 'y', 'z'), degree=2), states, rates)
    assert sum(len(t) for t in dense.terms().values()) > sum(len(t) for t in TRUE.values())


def test_the_identified_rate_matches_and_refuses_a_state_outside_the_data(data):
    states, rates = data
    law = SparseRegression(threshold=0.5).fit(polynomial_library(('x', 'y', 'z'), degree=2), states, rates)
    probe = states[::997]
    np.testing.assert_allclose(law.rate(probe), np.array([_lorenz(s) for s in probe]), rtol=0, atol=0.05)  # 1/time
    with pytest.raises(Refused, match='out_of_domain'):
        law.rate(np.array([[100.0, 0.0, 0.0]]))


def test_the_library_names_its_terms_and_a_mismatched_width_is_refused():
    library = polynomial_library(('a', 'b'), degree=2)
    assert library.names == ('1', 'a', 'b', 'a^2', 'a*b', 'b^2')
    np.testing.assert_array_equal(library.evaluate(np.array([[2.0, 3.0]])), [[1, 2, 3, 4, 6, 9]])
    with pytest.raises(Refused, match='variables'):
        library.evaluate(np.zeros((1, 3)))


def test_time_derivative_is_second_order_accurate():
    t = np.linspace(0, 1, 201)
    np.testing.assert_allclose(time_derivative(np.sin(t)[:, None], t[1] - t[0])[:, 0], np.cos(t), atol=1e-4)  # 1/s
