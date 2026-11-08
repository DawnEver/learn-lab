"""A grid -> grid field model learns a planted linear PDE solution operator: 2-D Poisson, zero Dirichlet."""

import numpy as np
import pytest

from learn_lab.deep.field import FieldModel
from learn_lab.deep.fit import Schedule
from learn_lab.errors import Refused

N = 16
BAR = 0.05  # relative L2 over the held-out set, dimensionless


def _laplacian(n):
    """The 5-point -Laplacian on the n x n interior of the unit square (zero Dirichlet), h = 1/(n+1)."""
    t = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    return (np.kron(t, np.eye(n)) + np.kron(np.eye(n), t)) * (n + 1) ** 2


def _sources(count, seed):
    """Smooth random sources: 4 x 4 sine modes with standard-normal amplitudes."""
    rng = np.random.default_rng(seed)
    s = np.arange(1, N + 1) / (N + 1)
    modes = np.sin(np.pi * np.arange(1, 5)[:, None] * s[None, :])  # (4, N)
    a = rng.standard_normal((count, 4, 4))
    return np.einsum('nkl,ki,lj->nij', a, modes, modes)[:, None]


def _solve(f):
    u = np.linalg.solve(_laplacian(N), f.reshape(f.shape[0], -1).T).T
    return u.reshape(f.shape)


def _relative_l2(p, y):
    return float(np.linalg.norm(p - y) / np.linalg.norm(y))


@pytest.fixture(scope='module')
def data():
    f_train, f_test = _sources(512, 0), 0.9 * _sources(64, 1)
    return f_train, _solve(f_train), f_test, _solve(f_test)


def test_the_field_model_learns_the_poisson_operator_on_held_out_sources(data):
    f, u, f_test, u_test = data
    fitted = FieldModel(schedule=Schedule(adam_steps=600, learning_rate=3e-3)).fit(f, u)
    inside = fitted.envelope.classify(f_test) == 'in_domain'
    assert inside.sum() >= 48  # the held-out set is mostly in the envelope; the rest would be refused
    error = _relative_l2(fitted.predict(f_test[inside]), u_test[inside])
    assert error < BAR
    # PLANTED CONTROL: a model trained one step must fail the same bar, or the bar measures nothing.
    untrained = FieldModel(schedule=Schedule(adam_steps=1)).fit(f, u)
    assert _relative_l2(untrained.predict(f_test[inside]), u_test[inside]) > BAR


def test_another_grid_shape_is_invalid_and_a_stronger_source_out_of_domain(data):
    f, u, _, _ = data
    fitted = FieldModel(schedule=Schedule(adam_steps=1)).fit(f[:8], u[:8])
    with pytest.raises(Refused, match='invalid'):
        fitted.predict(np.zeros((1, 1, 8, 8)))
    with pytest.raises(Refused, match='out_of_domain'):
        fitted.predict(10 * f[:1])


def test_a_grid_the_levels_cannot_halve_is_refused():
    with pytest.raises(Refused, match='divisible'):
        FieldModel(levels=2, schedule=Schedule(adam_steps=1)).fit(np.zeros((2, 1, 10, 10)), np.zeros((2, 1, 10, 10)))
