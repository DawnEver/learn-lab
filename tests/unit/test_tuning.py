"""Hyper-parameter tuning against an analytic quadratic (ported from optimi-lab, bars unchanged).

Tuning the polynomial degree over ``(1, 2, 3)`` must choose 2 (2 and 3 are both exact on
``y = 1 + x1 - 2 x2 + x1 x2``; a tie keeps grid order) and score degree 1 lower -- the planted control
that the score can tell a wrong setting.
"""

import numpy as np
import pytest

from learn_lab.errors import Refused
from learn_lab.models.response import Bagging, Polynomial
from learn_lab.tuning import tune
from learn_lab.validity import BoxEnvelope

RNG = np.random.default_rng(0)
X = RNG.uniform(-1.0, 1.0, size=(60, 2))


def quadratic(x: np.ndarray) -> np.ndarray:
    return (1.0 + x[:, 0] - 2.0 * x[:, 1] + x[:, 0] * x[:, 1])[:, None]


def test_tuning_chooses_the_degree_that_reproduces_the_quadratic():
    tuning = tune('polynomial', {'degree': (1, 2, 3)}, X, quadratic(X), k=5, seed=2)
    assert tuning.best.settings == {'degree': 2}
    assert tuning.best.score == pytest.approx(1.0, abs=1e-9)
    assert [t.settings['degree'] for t in tuning.trials] == [1, 2, 3]
    assert tuning.trials[0].score < 0.99
    assert tuning.surrogate == Polynomial(degree=2)


def test_tuning_minimises_an_error_score_over_a_two_setting_grid():
    grid = {'degree': (1, 2), 'penalty': (None,)}
    tuning = tune('polynomial', grid, X, quadratic(X), metric='rmse', seed=2)
    assert len(tuning.trials) == 2
    assert tuning.best.settings == {'degree': 2, 'penalty': None}
    assert tuning.best.score == pytest.approx(0.0, abs=1e-9)


def test_tuning_grids_a_bags_member_in_a_declared_domain():
    cube = BoxEnvelope(np.full(2, -1.0), np.full(2, 1.0))
    grid = {'member': (Polynomial(degree=1), Polynomial(degree=2)), 'n_members': (3,), 'seed': (4,)}
    tuning = tune('bagging', grid, X, quadratic(X), seed=2, domain=cube)
    assert tuning.best.settings['member'] == Polynomial(degree=2)
    assert tuning.best.score == pytest.approx(1.0, abs=1e-9)
    assert tuning.surrogate == Bagging(Polynomial(degree=2), n_members=3, seed=4)


def test_tuning_refuses_strangers():
    with pytest.raises(Refused, match='score is one of'):
        tune('polynomial', {'degree': (1,)}, X, quadratic(X), metric='nope')
    with pytest.raises(Refused, match='at least one candidate'):
        tune('polynomial', {'degree': ()}, X, quadratic(X))
    with pytest.raises(Refused, match='surrogate is one of'):
        tune('nope', {}, X, quadratic(X))
    with pytest.raises(Refused, match='holds every training point'):
        tune('polynomial', {'degree': (1,)}, X, quadratic(X), domain=BoxEnvelope(np.zeros(2), np.ones(2)))
