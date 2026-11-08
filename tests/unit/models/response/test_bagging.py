"""Bagging against an analytic quadratic (ported from optimi-lab's ``test_bagging_and_tuning``, bars unchanged).

``y = 1 + x1 - 2 x2 + x1 x2`` is reproduced EXACTLY by a degree-2 polynomial fitted on any resample
holding enough distinct points, so a bagged degree-2 polynomial predicts it to round-off with zero
spread; a degree-1 member cannot, and its bag shows a positive spread (the planted control).
"""

import numpy as np
import pytest

from learn_lab.errors import Refused
from learn_lab.models.response import Bagging, Polynomial, RegressionTree
from learn_lab.protocol import Give
from learn_lab.validity import BoxEnvelope

RNG = np.random.default_rng(0)
X = RNG.uniform(-1.0, 1.0, size=(60, 2))
QUERY = RNG.uniform(-1.0, 1.0, size=(15, 2))


def quadratic(x: np.ndarray) -> np.ndarray:
    return (1.0 + x[:, 0] - 2.0 * x[:, 1] + x[:, 0] * x[:, 1])[:, None]


def test_a_bag_of_exact_members_is_exact_with_zero_spread():
    fitted = Bagging(Polynomial(degree=2), n_members=8, seed=1).fit(X, quadratic(X))
    p = fitted.predict(QUERY, variance=True, gradient=True)
    assert p.mean == pytest.approx(quadratic(QUERY), abs=1e-9)
    assert p.variance == pytest.approx(np.zeros_like(p.variance), abs=1e-12)
    expected_slope = np.column_stack([1.0 + QUERY[:, 1], -2.0 + QUERY[:, 0]])
    assert p.gradient[:, 0, :] == pytest.approx(expected_slope, abs=1e-8)


def test_a_bag_of_biased_members_shows_spread_and_is_reproducible():
    bag = Bagging(Polynomial(degree=1), n_members=8, seed=1)
    first = bag.fit(X, quadratic(X)).predict(QUERY, variance=True)
    again = bag.fit(X, quadratic(X)).predict(QUERY, variance=True)
    assert np.all(first.variance > 0)
    assert np.array_equal(first.mean, again.mean)


def test_a_bag_gives_variance_always_and_gradient_only_through_its_member():
    assert Bagging(Polynomial()).gives == {Give.VARIANCE, Give.GRADIENT}
    assert Bagging(RegressionTree()).gives == {Give.VARIANCE}
    assert Bagging().member == RegressionTree()


def test_a_bag_refuses_degenerate_settings():
    with pytest.raises(Refused, match='at least two'):
        Bagging(Polynomial(), n_members=1)
    with pytest.raises(Refused, match='share of the rows'):
        Bagging(Polynomial(), max_samples=0.0)


def test_every_member_answers_in_the_bags_one_envelope():
    """A resample that misses the extreme rows still answers over the whole declared domain."""
    cube = BoxEnvelope(np.full(2, -2.0), np.full(2, 2.0))
    fitted = Bagging(Polynomial(degree=2), n_members=4, max_samples=0.2, seed=3).fit(X, quadratic(X), domain=cube)
    assert all(m.envelope is cube for m in fitted.members)
    corner = np.array([[1.9, -1.9]])
    assert fitted.predict(corner).mean == pytest.approx(quadratic(corner), abs=1e-9)
    with pytest.raises(Refused, match='out_of_domain'):
        fitted.predict(np.array([[2.1, 0.0]]))
