"""D43 -- polynomial response surfaces, ordinary / ridge / lasso, against scikit-learn (test oracle only).

Tolerances are ABSOLUTE in the target's units (the targets here are dimensionless numbers of order 1).
"""

import numpy as np
import pytest
from sklearn.linear_model import Lasso as SkLasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge as SkRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from learn_lab.errors import Refused
from learn_lab.models.response import Lasso, Polynomial, Ridge
from learn_lab.validity import BoxEnvelope

#: Absolute, target units: a direct least-squares solve agrees with sklearn's to rounding of a
#: condition number ~1e4 basis.
EXACT = 1e-9
#: Absolute, target units: both lasso solvers stop at a coordinate-descent tolerance of 1e-12.
LASSO = 1e-6


def _cube_of(x) -> BoxEnvelope:
    """The unit cube in the width of *x*: the declared domain of data drawn on [0, 1)."""
    return BoxEnvelope(np.zeros(np.shape(x)[1]), np.ones(np.shape(x)[1]))


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(3)
    x = rng.random((40, 3))
    y = np.column_stack([np.sin(3 * x[:, 0]) + x[:, 1] * x[:, 2], x[:, 0] ** 2 - x[:, 2]])
    return x, y, rng.random((15, 3))


def _sklearn(degree, regressor):
    return make_pipeline(PolynomialFeatures(degree, include_bias=False), regressor)


@pytest.mark.parametrize('degree', [1, 2, 3])
def test_ordinary_least_squares_agrees_with_sklearn(data, degree) -> None:
    x, y, q = data
    ours = Polynomial(degree=degree).fit(x, y, domain=_cube_of(x)).predict(q).mean
    theirs = _sklearn(degree, LinearRegression()).fit(x, y).predict(q)
    assert np.max(np.abs(ours - theirs)) <= EXACT


@pytest.mark.parametrize('alpha', [1e-3, 0.1, 10.0])
def test_ridge_agrees_with_sklearn(data, alpha) -> None:
    x, y, q = data
    ours = Polynomial(degree=2, penalty=Ridge(alpha)).fit(x, y, domain=_cube_of(x)).predict(q).mean
    theirs = _sklearn(2, SkRidge(alpha=alpha)).fit(x, y).predict(q)
    assert np.max(np.abs(ours - theirs)) <= EXACT


@pytest.mark.parametrize('alpha', [1e-3, 1e-2, 0.1])
def test_lasso_agrees_with_sklearn(data, alpha) -> None:
    x, y, q = data
    ours = Polynomial(degree=2, penalty=Lasso(alpha)).fit(x, y, domain=_cube_of(x)).predict(q).mean
    theirs = _sklearn(2, SkLasso(alpha=alpha, tol=1e-14, max_iter=1_000_000)).fit(x, y).predict(q)
    assert np.max(np.abs(ours - theirs)) <= LASSO


def test_lasso_zeroes_coefficients_and_ridge_does_not(data) -> None:
    """The planted control for the penalty: an l1 penalty makes exact zeros, an l2 penalty only shrinks."""
    x, y, _ = data
    lasso = Polynomial(degree=3, penalty=Lasso(0.05)).fit(x, y[:, :1], domain=_cube_of(x))
    ridge = Polynomial(degree=3, penalty=Ridge(0.05)).fit(x, y[:, :1], domain=_cube_of(x))
    assert np.sum(lasso.coefficients == 0.0) > 5
    assert np.sum(ridge.coefficients == 0.0) == 0


def test_the_gradient_is_the_derivative_of_the_mean(data) -> None:
    x, y, q = data
    fitted = Polynomial(degree=3).fit(x, y, domain=_cube_of(x))
    analytic = fitted.predict(q, gradient=True).gradient
    h = 1e-6
    for j in range(3):
        step = np.zeros(3)
        step[j] = h
        central = (fitted.predict(q + step).mean - fitted.predict(q - step).mean) / (2 * h)
        # Absolute, target units per unit input: central differences at h = 1e-6 carry ~1e-9 error.
        assert np.max(np.abs(analytic[:, :, j] - central)) <= 1e-7


def test_a_degree_below_one_or_a_negative_penalty_is_refused() -> None:
    with pytest.raises(Refused):
        Polynomial(degree=0)
    with pytest.raises(Refused):
        Ridge(-1.0)
    with pytest.raises(Refused):
        Lasso(0.0)
