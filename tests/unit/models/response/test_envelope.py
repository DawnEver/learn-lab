"""Every fitted response model answers only inside its envelope: the declared domain, else the training span.

The search case is the reason for the declared domain: an optimiser fits on the points it has and asks
about candidates anywhere in its design space, so its envelope is that space, never the samples' span.
"""

from __future__ import annotations

import numpy as np
import pytest

from learn_lab.errors import Refused
from learn_lab.models.response import (
    Bagging,
    ChaosExpansion,
    CoKriging,
    Ensemble,
    Forest,
    GaussianProcess,
    GradientEnhancedKriging,
    MultilayerPerceptron,
    MultiOutputKriging,
    NearestNeighbours,
    Polynomial,
    RadialBasis,
    RegressionTree,
)
from learn_lab.registry import SURROGATES
from learn_lab.validation import cross_validate
from learn_lab.validity import BoxEnvelope, Validity

X = np.random.default_rng(4).uniform(0.2, 0.8, (24, 2))
Y = np.column_stack([np.sin(3 * X[:, 0]) + X[:, 1], X[:, 0] * X[:, 1]])
CUBE = BoxEnvelope(np.zeros(2), np.ones(2))
#: Inside the cube, outside the samples' span (which lies within [0.2, 0.8]^2).
CANDIDATE = np.array([[0.95, 0.05]])

FITS = {
    'polynomial': lambda d: Polynomial(degree=2).fit(X, Y, domain=d),
    'radial_basis': lambda d: RadialBasis().fit(X, Y, domain=d),
    'gaussian_process': lambda d: GaussianProcess(restarts=0).fit(X, Y, domain=d),
    'gradient_enhanced_kriging': lambda d: GradientEnhancedKriging(restarts=0).fit(
        X, Y[:, :1], np.zeros((24, 1, 2)), domain=d
    ),
    'multi_output_kriging': lambda d: MultiOutputKriging(restarts=0).fit(X, Y, domain=d),
    'co_kriging': lambda d: CoKriging(restarts=0).fit(X, Y, np.arange(24) % 2, domain=d),
    'nearest_neighbours': lambda d: NearestNeighbours(k=3).fit(X, Y, domain=d),
    'regression_tree': lambda d: RegressionTree(max_depth=3).fit(X, Y, domain=d),
    'forest': lambda d: Forest(n_trees=3).fit(X, Y, domain=d),
    'multilayer_perceptron': lambda d: MultilayerPerceptron((4,), max_iterations=20).fit(X, Y, domain=d),
    'chaos_expansion': lambda d: ChaosExpansion(degree=2).fit(X, Y, domain=d),
    'ensemble': lambda d: Ensemble((Polynomial(degree=1), NearestNeighbours(k=3)), k=3).fit(X, Y, domain=d),
    'bagging': lambda d: Bagging(Polynomial(degree=1), n_members=3, seed=1).fit(X, Y, domain=d),
}


def test_the_table_covers_every_registered_surrogate() -> None:
    assert set(SURROGATES) <= set(FITS)


@pytest.mark.parametrize('family', list(FITS))
def test_a_declared_domain_answers_a_candidate_outside_the_samples_span(family) -> None:
    fitted = FITS[family](CUBE)
    assert fitted.envelope is CUBE
    assert np.all(np.isfinite(fitted.predict(CANDIDATE).mean))
    with pytest.raises(Refused, match='out_of_domain'):
        fitted.predict(np.array([[1.01, 0.5]]))


@pytest.mark.parametrize('family', [f for f in FITS if f != 'chaos_expansion'])
def test_without_a_domain_the_envelope_is_the_span_and_a_stranger_is_refused(family) -> None:
    """PLANTED CONTROL for the test above: the same candidate, no declared domain, is refused."""
    fitted = FITS[family](None)
    np.testing.assert_array_equal(fitted.envelope.lower, X.min(axis=0))
    np.testing.assert_array_equal(fitted.envelope.upper, X.max(axis=0))
    assert fitted.envelope.classify(CANDIDATE)[0] == Validity.OUT_OF_DOMAIN
    with pytest.raises(Refused, match='out_of_domain'):
        fitted.predict(CANDIDATE)
    assert fitted.predict(X).mean.shape[0] == len(X)


def test_a_chaos_expansion_answers_on_its_unit_cube_by_default() -> None:
    assert np.all(np.isfinite(FITS['chaos_expansion'](None).predict(CANDIDATE).mean))
    with pytest.raises(Refused, match='unit cube'):
        ChaosExpansion(degree=2).fit(X, Y, domain=BoxEnvelope(np.zeros(2), np.full(2, 2.0)))


def test_a_gp_conditions_only_on_points_inside_its_envelope() -> None:
    fitted = FITS['gaussian_process'](CUBE)
    assert fitted.condition(CANDIDATE, np.zeros((1, 2))).envelope is CUBE
    with pytest.raises(Refused, match='out_of_domain'):
        fitted.condition(np.array([[1.5, 0.5]]), np.zeros((1, 2)))


def test_cross_validation_fits_every_fold_in_the_one_envelope() -> None:
    """The extreme row of a fold is outside that fold's own span, yet inside the data's: it is predicted."""
    prediction = cross_validate(Polynomial(degree=2), X, Y, k=6, seed=1)
    assert np.all(np.isfinite(prediction.mean))
