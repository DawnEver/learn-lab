"""What is tuned and how a setting is scored; the search itself is optimi-lab's.

The planted controls: ``y = 1 + x1 - 2 x2 + x1 x2`` is exact at polynomial degree 2 and not at 1, so the
objective must rank degree 2 below (better than) degree 1; a setting outside the declared space, an
unknown setting and an ill-formed :class:`Space` are each refused; a fit that fails numerically is
``inf`` on record, never an exception.
"""

import math

import numpy as np
import pytest

from learn_lab.errors import Refused
from learn_lab.models.response import Bagging, Polynomial
from learn_lab.registry import SURROGATES
from learn_lab.tuning import Choice, Integer, Objective, Real, Space, objective
from learn_lab.validity import BoxEnvelope

RNG = np.random.default_rng(0)
X = RNG.uniform(-1.0, 1.0, size=(60, 2))


def quadratic(x: np.ndarray) -> np.ndarray:
    return (1.0 + x[:, 0] - 2.0 * x[:, 1] + x[:, 0] * x[:, 1])[:, None]


def test_the_objective_ranks_the_exact_degree_above_the_wrong_one():
    f = objective('polynomial', X, quadratic(X), k=5, seed=2)
    good, bad = f({'degree': 2}), f({'degree': 1})
    assert good == pytest.approx(-1.0, abs=1e-9)  # r2 is maximised, so its negation is minimised
    assert bad > -0.99
    assert good < bad
    assert [(e.settings, e.value) for e in f.evaluations] == [({'degree': 2}, good), ({'degree': 1}, bad)]


def test_an_error_metric_is_minimised_as_it_stands():
    f = objective('polynomial', X, quadratic(X), metric='rmse', seed=2)
    assert f({'degree': 2}) == pytest.approx(0.0, abs=1e-9)
    assert f({'degree': 1}) > 0.1


def test_every_call_sees_the_same_folds():
    f = objective('polynomial', X, quadratic(X), seed=2)
    assert f({'degree': 1}) == f({'degree': 1})


def test_fixed_settings_hold_for_every_call_in_a_declared_domain():
    cube = BoxEnvelope(-np.ones(2), np.ones(2))
    f = objective('bagging', X, quadratic(X), seed=2, domain=cube, fixed={'member': Polynomial(degree=2), 'seed': 4})
    assert f({'n_members': 3}) == pytest.approx(-1.0, abs=1e-9)
    assert f.build({'n_members': 3}) == Bagging(Polynomial(degree=2), n_members=3, seed=4)


def test_a_setting_outside_the_declared_space_is_refused():
    f = objective('polynomial', X, quadratic(X))
    hi = f.space['degree'].hi
    with pytest.raises(Refused, match='degree'):
        f({'degree': hi + 1})
    with pytest.raises(Refused, match='degree'):
        f({'degree': 2.5})
    with pytest.raises(Refused, match='nope'):
        f({'nope': 1})
    assert f.evaluations == ()


def test_a_fixed_setting_may_not_also_be_tuned():
    with pytest.raises(Refused, match='degree'):
        objective('polynomial', X, quadratic(X), fixed={'degree': 2})


def test_a_numerically_failing_fit_is_inf_on_record():
    y = quadratic(X)
    f = objective('radial_basis', X[:4], y[:4], k=2, fixed={'degree': 3})  # a cubic tail on 2 points is singular
    value = f({'kernel': 'cubic', 'epsilon': 1.0, 'smoothing': 0.0})
    assert value == math.inf
    assert f.evaluations[-1].value == math.inf
    assert f.evaluations[-1].failure


def test_strangers_are_refused():
    with pytest.raises(Refused):
        objective('polynomial', X, quadratic(X), metric='nope')
    with pytest.raises(Refused):
        objective('nope', X, quadratic(X))
    with pytest.raises(Refused):
        objective('polynomial', X, quadratic(X), domain=BoxEnvelope(np.zeros(2), np.ones(2)))


@pytest.mark.parametrize(
    'make',
    [
        lambda: Real(1.0, 1.0),
        lambda: Real(2.0, 1.0),
        lambda: Real(0.0, 1.0, log=True),
        lambda: Real(-1.0, 1.0, log=True),
        lambda: Integer(3, 3),
        lambda: Integer(1.5, 3),
        lambda: Choice(()),
        lambda: Space({'a': (1, 2)}),
        lambda: Space({'': Integer(1, 2)}),
    ],
)
def test_a_bad_space_is_refused(make):
    with pytest.raises(Refused):
        make()


def test_a_space_is_a_frozen_mapping_that_says_what_it_holds():
    space = Space({'rate': Real(1e-4, 1e-1, log=True), 'n': Integer(1, 4), 'kind': Choice(('a', 'b'))})
    assert set(space) == {'rate', 'n', 'kind'} and len(space) == 3
    assert space.holds({'rate': 1e-2, 'n': 4, 'kind': 'b'})
    assert not space.holds({'rate': 1.0})
    assert not space.holds({'n': True})
    with pytest.raises(TypeError):
        space['n'] = Integer(1, 2)


#: Families with nothing to search: every setting is fitted, is physics the caller pins, or is a numerical
#: tolerance of the fit (the interpolant's gradient estimate) that moves no score a search should chase.
UNTUNED = {'clough_tocher', 'power_law'}


def test_every_response_family_declares_a_space_its_constructor_accepts():
    for name, family in SURROGATES.items():
        assert bool(len(family.space)) is (name not in UNTUNED), name
        family.build(**family.space.lowest())  # the corner of the space is a valid declaration
        assert isinstance(objective(name, X, quadratic(X)), Objective)
