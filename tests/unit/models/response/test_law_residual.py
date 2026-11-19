"""A law as the mean and a learned residual on top: ``y = law(x) + residual(x)``.

The planted data are a power law ``y = 2 x0^1.5 x1^-1`` (dimensionless) plus a smooth bump the law cannot
express. Errors are relative to the data's own magnitude, so every bar is dimensionless.
"""

import numpy as np
import pytest

from learn_lab.errors import Refused
from learn_lab.models.response import GaussianProcess, LawResidual, PowerLaw, RegressionTree
from learn_lab.protocol import Give

RNG = np.random.default_rng(3)
X = RNG.uniform(1.0, 3.0, (60, 2))
QUERY = np.random.default_rng(4).uniform(1.1, 2.9, (25, 2))


def law(x: np.ndarray) -> np.ndarray:
    return (2.0 * x[:, 0] ** 1.5 / x[:, 1])[:, None]


def bump(x: np.ndarray) -> np.ndarray:
    return (0.8 * np.sin(2.0 * x[:, 0]) * np.cos(1.5 * x[:, 1]))[:, None]


MODEL = LawResidual(law=PowerLaw(fixed=((1.5, -1.0),)), base=GaussianProcess(restarts=1))


def _relative(predicted: np.ndarray, truth: np.ndarray) -> float:
    return float(np.max(np.abs(predicted - truth)) / np.max(np.abs(truth)))


def test_the_residual_learns_what_the_law_leaves_and_the_law_alone_cannot():
    y = law(X) + bump(X)
    truth = law(QUERY) + bump(QUERY)
    both = _relative(MODEL.fit(X, y).predict(QUERY).mean, truth)
    alone = _relative(PowerLaw(fixed=((1.5, -1.0),)).fit(X, y).predict(QUERY).mean, truth)
    assert both < 2e-3, (alone, both)  # measured 3.6e-4 against the law alone 8.7e-2 (seed 3, 60 rows)
    assert alone > 5 * both, (alone, both)  # planted control: the law alone misses the bump


def test_on_exact_law_data_the_mean_is_the_law():
    fitted = MODEL.fit(X, law(X))
    assert _relative(fitted.predict(QUERY).mean, law(QUERY)) < 1e-6


def test_the_gradient_is_the_laws_plus_the_residuals_chained_through_the_unit_box():
    fitted = MODEL.fit(X, law(X) + bump(X))
    got = fitted.predict(QUERY, gradient=True).gradient
    step = 1e-6
    for i in range(2):
        shift = np.zeros(2)
        shift[i] = step
        central = (fitted.predict(QUERY + shift).mean - fitted.predict(QUERY - shift).mean) / (2 * step)
        np.testing.assert_allclose(got[:, :, i], central, rtol=1e-5, atol=1e-6)


def test_the_variance_is_the_residuals_and_a_gradient_needs_both_parts():
    fitted = MODEL.fit(X, law(X) + bump(X))
    assert fitted.gives == frozenset({Give.VARIANCE, Give.GRADIENT})
    assert (fitted.predict(QUERY, variance=True).variance >= 0).all()
    stepwise = LawResidual(law=RegressionTree(max_depth=2), base=GaussianProcess(restarts=0))
    assert Give.GRADIENT not in stepwise.gives
    with pytest.raises(Refused, match='gives no gradient'):
        stepwise.fit(X, law(X)).predict(QUERY, gradient=True)
