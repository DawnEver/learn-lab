"""D56 ensemble, D57 cross-validation and calibration, against scikit-learn / scipy.stats (tests only).

Scores and folds must agree with sklearn EXACTLY (absolute, in the score's own unit). Calibration is
checked on data whose calibration is known by construction: draws from the predicted distribution
cover at the nominal rate, and a predicted spread that is planted too narrow is read as over-confident.
"""

import numpy as np
import pytest
from scipy.stats import norm
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold

from learn_lab.errors import Refused
from learn_lab.models.response import Ensemble, GaussianProcess, NearestNeighbours, Polynomial
from learn_lab.protocol import Prediction
from learn_lab.validation import SCORES, calibration, cross_validate, folds, score
from learn_lab.validity import BoxEnvelope

_SKLEARN = {
    'r2': r2_score,
    'mse': mean_squared_error,
    'rmse': root_mean_squared_error,
    'mae': mean_absolute_error,
    'median_absolute_error': median_absolute_error,
}


def _cube_of(x) -> BoxEnvelope:
    """The unit cube in the width of *x*: the declared domain of data drawn on [0, 1)."""
    return BoxEnvelope(np.zeros(np.shape(x)[1]), np.ones(np.shape(x)[1]))


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(16)
    x = rng.random((40, 2))
    y = np.column_stack([np.sin(4 * x[:, 0]) + x[:, 1], x[:, 0] * x[:, 1]])
    return x, y


@pytest.mark.parametrize('metric', SCORES)
def test_every_score_equals_sklearns(metric) -> None:
    rng = np.random.default_rng(17)
    measured, predicted = rng.random((30, 2)), rng.random((30, 2))
    assert abs(score(metric, measured, predicted) - _SKLEARN[metric](measured, predicted)) <= 1e-12


def test_the_score_set_is_exactly_the_oracle_set() -> None:
    assert set(SCORES) == set(_SKLEARN)
    with pytest.raises(Refused, match='r2'):
        score('accuracy', np.zeros((2, 1)), np.zeros((2, 1)))


@pytest.mark.parametrize(('n', 'k'), [(10, 3), (40, 5), (7, 7)])
def test_unshuffled_folds_equal_sklearns(n, k) -> None:
    ours = folds(n, k)
    theirs = [test for _, test in KFold(n_splits=k).split(np.zeros((n, 1)))]
    assert len(ours) == len(theirs)
    for a, b in zip(ours, theirs, strict=True):
        assert np.array_equal(a, b)


def test_shuffled_folds_partition_the_points_and_the_seed_fixes_them() -> None:
    a = folds(23, 4, seed=1)
    assert np.array_equal(np.sort(np.concatenate(a)), np.arange(23))
    assert all(np.array_equal(p, q) for p, q in zip(a, folds(23, 4, seed=1), strict=True))
    assert not all(np.array_equal(p, q) for p, q in zip(a, folds(23, 4, seed=2), strict=True))


def test_cross_validation_predicts_every_point_from_the_other_folds(data) -> None:
    x, y = data
    held_out = cross_validate(Polynomial(degree=2), x, y, k=5)
    for test in folds(len(x), 5):
        train = np.setdiff1d(np.arange(len(x)), test)
        expected = Polynomial(degree=2).fit(x[train], y[train], domain=_cube_of(x[train])).predict(x[test]).mean
        assert np.max(np.abs(held_out.mean[test] - expected)) <= 1e-12


def test_draws_from_the_predicted_distribution_are_calibrated() -> None:
    rng = np.random.default_rng(18)
    mean = rng.standard_normal((4000, 1))
    sigma = 0.5 + rng.random((4000, 1))
    measured = mean + sigma * rng.standard_normal((4000, 1))
    report = calibration(measured, Prediction(mean, sigma**2))
    # Absolute, dimensionless coverage fractions: four binomial standard errors at n = 4000.
    for nominal, observed in report.coverage.items():
        assert abs(observed - nominal) <= 4 * np.sqrt(nominal * (1 - nominal) / 4000)
    assert abs(report.standardised_variance - 1.0) <= 0.1
    expected = -np.mean(norm.logpdf(measured, mean, sigma))
    assert abs(report.negative_log_density - expected) <= 1e-12 * abs(expected)
    # Planted control: a spread three times too narrow must read as over-confident.
    narrow = calibration(measured, Prediction(mean, (sigma / 3) ** 2))
    assert narrow.coverage[0.9] < 0.5
    assert narrow.standardised_variance > 5


def test_calibration_needs_a_variance() -> None:
    with pytest.raises(Refused, match='variance'):
        calibration(np.zeros((3, 1)), Prediction(np.zeros((3, 1))))


def test_the_ensemble_is_the_score_weighted_mean_of_its_members(data) -> None:
    x, y = data
    members = (Polynomial(degree=2), NearestNeighbours(k=3))
    fitted = Ensemble(members, k=5).fit(x, y, domain=_cube_of(x))
    r2 = np.array([score('r2', y, cross_validate(m, x, y, k=5).mean) for m in members])
    weights = np.maximum(r2, 0) / np.maximum(r2, 0).sum()
    assert np.allclose(fitted.weights, weights, rtol=0, atol=1e-12)
    q = np.random.default_rng(19).random((5, 2))
    expected = sum(w * m.fit(x, y, domain=_cube_of(x)).predict(q).mean for w, m in zip(weights, members, strict=True))
    assert np.max(np.abs(fitted.predict(q).mean - expected)) <= 1e-12
    # Planted control: the weighting is live -- the combination is not the plain mean of the members.
    plain = np.mean([m.fit(x, y, domain=_cube_of(x)).predict(q).mean for m in members], axis=0)
    assert np.max(np.abs(fitted.predict(q).mean - plain)) > 1e-3
    best = Ensemble(members, weighting='best', k=5).fit(x, y, domain=_cube_of(x))
    assert np.array_equal(best.weights, np.eye(2)[int(np.argmax(r2))])


def test_the_ensemble_gives_only_what_every_member_gives(data) -> None:
    x, y = data
    assert Ensemble((Polynomial(), NearestNeighbours())).gives == frozenset()
    both = Ensemble((GaussianProcess(), GaussianProcess('matern52'))).fit(x, y, domain=_cube_of(x))
    p = both.predict(x[:3], variance=True, gradient=True)
    assert p.variance.shape == (3, 2)
    assert np.all(p.variance >= 0)


def test_an_ensemble_whose_every_member_scores_nothing_is_refused() -> None:
    rng = np.random.default_rng(20)
    x, y = rng.random((30, 1)), rng.standard_normal((30, 1))
    with pytest.raises(Refused, match='r2'):
        Ensemble((NearestNeighbours(k=1), Polynomial(degree=1))).fit(x, y, domain=_cube_of(x))
