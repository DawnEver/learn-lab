"""Neural operators learn the planted 1-D Poisson solution operator -u'' = f, u(0) = u(1) = 0, on held-out f."""

import numpy as np
import pytest

from learn_lab.deep.fit import Schedule
from learn_lab.deep.operator import FNO, DeepONet
from learn_lab.errors import Refused

MODES = np.arange(1, 6)
DEEPONET_BAR = 0.02  # relative L2 over the held-out set, dimensionless
FNO_BAR = 0.01  # relative L2 at the training resolution, dimensionless
FNO_FINER_BAR = 0.03  # relative L2 at twice the training resolution, dimensionless


def _amplitudes(count, seed):
    return np.random.default_rng(seed).standard_normal((count, MODES.size))


def _source(a, x):
    return a @ np.sin(np.pi * MODES[:, None] * x[None, :])


def _solution(a, x):
    return (a / (np.pi * MODES) ** 2) @ np.sin(np.pi * MODES[:, None] * x[None, :])


def _relative_l2(p, y):
    return float(np.linalg.norm(p - y) / np.linalg.norm(y))


TRAIN, TEST = _amplitudes(512, 0), 0.8 * _amplitudes(64, 1)


def test_a_deeponet_maps_sensor_values_to_the_solution_at_any_point():
    sensors, points = np.linspace(0, 1, 32), np.linspace(0, 1, 50)
    model = DeepONet(schedule=Schedule(adam_steps=1500, learning_rate=2e-3))
    fitted = model.fit(_source(TRAIN, sensors), points[:, None], _solution(TRAIN, points))
    query = np.random.default_rng(2).uniform(0, 1, 77)  # points never trained on
    error = _relative_l2(fitted.predict(_source(TEST, sensors), query[:, None]), _solution(TEST, query))
    assert error < DEEPONET_BAR
    # PLANTED CONTROL: one training step cannot pass the bar.
    untrained = DeepONet(schedule=Schedule(adam_steps=1)).fit(
        _source(TRAIN, sensors), points[:, None], _solution(TRAIN, points)
    )
    assert (
        _relative_l2(untrained.predict(_source(TEST, sensors), query[:, None]), _solution(TEST, query)) > DEEPONET_BAR
    )
    with pytest.raises(Refused, match='out_of_domain'):
        fitted.predict(_source(TEST, sensors), np.array([[1.5]]))
    with pytest.raises(Refused, match='invalid'):
        fitted.predict(_source(TEST, np.linspace(0, 1, 31)), query[:, None])


def test_an_fno_learns_the_operator_and_keeps_it_at_twice_the_resolution():
    coarse, fine = np.linspace(0, 1, 64), np.linspace(0, 1, 128)
    fitted = FNO(modes=(12,), schedule=Schedule(adam_steps=1000, learning_rate=3e-3)).fit(
        _source(TRAIN, coarse)[:, None], _solution(TRAIN, coarse)[:, None]
    )
    at_coarse = _relative_l2(fitted.predict(_source(TEST, coarse)[:, None])[:, 0], _solution(TEST, coarse))
    at_fine = _relative_l2(fitted.predict(_source(TEST, fine)[:, None])[:, 0], _solution(TEST, fine))
    assert at_coarse < FNO_BAR
    assert at_fine < FNO_FINER_BAR
    # PLANTED CONTROL: one training step cannot pass the bar.
    untrained = FNO(modes=(12,), schedule=Schedule(adam_steps=1)).fit(
        _source(TRAIN, coarse)[:, None], _solution(TRAIN, coarse)[:, None]
    )
    assert _relative_l2(untrained.predict(_source(TEST, coarse)[:, None])[:, 0], _solution(TEST, coarse)) > FNO_BAR


def test_a_2d_fno_runs_and_refuses_a_grid_too_coarse_for_its_modes():
    x = np.random.default_rng(0).standard_normal((4, 2, 16, 16))
    fitted = FNO(modes=(4, 4), schedule=Schedule(adam_steps=2)).fit(x, x[:, :1])
    assert fitted.predict(x).shape == (4, 1, 16, 16)
    with pytest.raises(Refused, match='modes'):
        fitted.predict(x[:, :, ::4, ::4])
    with pytest.raises(Refused, match='invalid'):
        fitted.predict(x[:, :, 0])
