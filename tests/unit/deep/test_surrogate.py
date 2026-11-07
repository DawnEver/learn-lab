import numpy as np
import pytest

from learn_lab.deep.fit import Schedule
from learn_lab.deep.surrogate import DeepEnsemble
from learn_lab.errors import Refused


def _truth(x):
    return np.stack([np.sin(2 * x[:, 0]) + x[:, 1] ** 2], axis=1)


def _grad(x):
    return np.stack([2 * np.cos(2 * x[:, 0]), 2 * x[:, 1]], axis=1)[:, None, :]


@pytest.fixture(scope='module')
def fitted():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (400, 2))
    return DeepEnsemble(members=4, schedule=Schedule(adam_steps=2500, learning_rate=3e-3)).fit(x, _truth(x))


def test_the_ensemble_predicts_held_out_points_mean_and_gradient(fitted):
    x = np.random.default_rng(1).uniform(-0.9, 0.9, (200, 2))
    p = fitted.predict(x, variance=True, gradient=True)
    y = _truth(x)
    assert np.sqrt(np.mean((p.mean - y) ** 2)) < 0.02 * np.ptp(y)  # rms error, unit of y
    assert np.all(p.variance >= 0)
    assert np.any(p.variance > 0)
    assert np.sqrt(np.mean((p.gradient - _grad(x)) ** 2)) < 0.1  # unit of dy/dx; |grad| ~ 2


def test_a_query_outside_the_training_envelope_is_refused(fitted):
    with pytest.raises(Refused, match='out_of_domain'):
        fitted.predict(np.array([[1.5, 0.0]]))


def test_a_single_member_gives_no_variance():
    x = np.linspace(0, 1, 8)[:, None]
    one = DeepEnsemble(members=1, schedule=Schedule(adam_steps=5)).fit(x, x)
    with pytest.raises(Refused, match='gives no variance'):
        one.predict(x, variance=True)
