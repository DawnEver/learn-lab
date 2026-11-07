"""D49 gradient-enhanced Kriging, D50 ICM multi-output, D51 AR1 co-Kriging.

No library ships these with a comparable parametrisation, so each is checked against what it must
be by construction: its log-likelihood is a multivariate normal log-density of the stacked data
(``scipy.stats.multivariate_normal``, test oracle only), its likelihood gradient is the derivative
of that likelihood, and each beats its plain-GP counterpart on the problem it exists for -- with a
planted control showing the gain comes from the extra information, not from the fit.
"""

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from learn_lab.errors import Refused
from learn_lab.models.response import CoKriging, GaussianProcess, GradientEnhancedKriging, MultiOutputKriging

#: Relative: the same log-density computed two ways (Cholesky here, an eigendecomposition in scipy).
DENSITY = 1e-9
#: Relative: the same, for the value-and-gradient covariance, whose condition number reaches ~1e8 at
#: these points, so two factorisations agree to ~1e-8 of the log-density.
GEK_DENSITY = 1e-7


def _branin_like(x):
    value = np.sin(3 * x[:, 0]) * np.cos(2 * x[:, 1]) + x[:, 0] ** 2
    grad = np.column_stack([
        3 * np.cos(3 * x[:, 0]) * np.cos(2 * x[:, 1]) + 2 * x[:, 0],
        -2 * np.sin(3 * x[:, 0]) * np.sin(2 * x[:, 1]),
    ])
    return value[:, None], grad[:, None, :]


def _rmse(prediction, truth):
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


# -- D49 -------------------------------------------------------------------------------------------


def test_gek_likelihood_is_the_density_of_values_and_gradients() -> None:
    rng = np.random.default_rng(1)
    x = rng.random((6, 2))
    y, g = _branin_like(x)
    model = GradientEnhancedKriging()
    theta = np.log([1.3, 0.4, 0.7])
    value, _ = model.log_likelihood(theta, x, y[:, 0], g[:, 0, :])
    covariance = model.covariance(theta, x)
    stacked = np.concatenate([y[:, 0], g[:, 0, :].reshape(-1)])
    assert abs(value - multivariate_normal(np.zeros(stacked.size), covariance).logpdf(stacked)) <= GEK_DENSITY * abs(
        value
    )


def test_gek_likelihood_gradient_is_its_derivative() -> None:
    rng = np.random.default_rng(2)
    x = rng.random((6, 2))
    y, g = _branin_like(x)
    model = GradientEnhancedKriging()
    theta = np.log([1.3, 0.4, 0.7])
    _, analytic = model.log_likelihood(theta, x, y[:, 0], g[:, 0, :])
    h = 1e-6
    for i in range(theta.size):
        step = np.zeros_like(theta)
        step[i] = h
        central = (
            model.log_likelihood(theta + step, x, y[:, 0], g[:, 0, :])[0]
            - model.log_likelihood(theta - step, x, y[:, 0], g[:, 0, :])[0]
        ) / (2 * h)
        assert abs(analytic[i] - central) <= 1e-5 * max(1.0, abs(central))  # relative, dimensionless


def test_gek_interpolates_values_and_gradients_and_beats_a_gp_with_the_same_points() -> None:
    rng = np.random.default_rng(3)
    x = rng.random((8, 2))
    y, g = _branin_like(x)
    fitted = GradientEnhancedKriging().fit(x, y, g)
    at_data = fitted.predict(x, gradient=True)
    # Absolute, target units (and per unit input): the interpolation conditions, up to the nugget.
    assert np.max(np.abs(at_data.mean - y)) <= 1e-4
    assert np.max(np.abs(at_data.gradient - g)) <= 1e-3
    q = rng.random((200, 2))
    truth, _ = _branin_like(q)
    gek = _rmse(fitted.predict(q).mean, truth)
    gp = _rmse(GaussianProcess().fit(x, y).predict(q).mean, truth)
    assert gek < 0.5 * gp
    # Planted control: the same fit handed WRONG (zero) gradients must lose its advantage.
    wrong = _rmse(GradientEnhancedKriging().fit(x, y, np.zeros_like(g)).predict(q).mean, truth)
    assert wrong > 2 * gek


# -- D50 -------------------------------------------------------------------------------------------


def test_icm_likelihood_is_the_density_of_the_stacked_outputs() -> None:
    rng = np.random.default_rng(4)
    x = rng.random((7, 2))
    y = np.column_stack([np.sin(3 * x[:, 0]), np.cos(2 * x[:, 1])])
    model = MultiOutputKriging(rank=1)
    theta = model.initial(x.shape[1], 2) + 0.1 * rng.standard_normal(model.initial(x.shape[1], 2).size)
    value, _ = model.log_likelihood(theta, x, y)
    stacked = y.T.reshape(-1)  # output-major, the order of kron(B, R)
    density = multivariate_normal(np.zeros(stacked.size), model.covariance(theta, x)).logpdf(stacked)
    assert abs(value - density) <= DENSITY * abs(value)


def test_icm_likelihood_gradient_is_its_derivative() -> None:
    rng = np.random.default_rng(5)
    x = rng.random((7, 2))
    y = np.column_stack([np.sin(3 * x[:, 0]), np.cos(2 * x[:, 1])])
    model = MultiOutputKriging(rank=1)
    theta = model.initial(2, 2) + 0.1 * rng.standard_normal(model.initial(2, 2).size)
    _, analytic = model.log_likelihood(theta, x, y)
    h = 1e-6
    for i in range(theta.size):
        step = np.zeros_like(theta)
        step[i] = h
        central = (model.log_likelihood(theta + step, x, y)[0] - model.log_likelihood(theta - step, x, y)[0]) / (2 * h)
        assert abs(analytic[i] - central) <= 1e-5 * max(1.0, abs(central))


def test_icm_learns_that_related_outputs_are_correlated_and_unrelated_ones_are_not() -> None:
    rng = np.random.default_rng(6)
    x = rng.random((20, 1))
    base = np.sin(6 * x[:, 0])
    related = MultiOutputKriging(rank=1).fit(x, np.column_stack([base, -2 * base + 0.01 * rng.standard_normal(20)]))
    assert related.correlation[0, 1] < -0.95
    # Planted control: a second output ORTHOGONAL to the first on [0, 1] must not be read as correlated.
    grid = np.linspace(0, 1, 20, endpoint=False)[:, None]
    unrelated = MultiOutputKriging(rank=2).fit(
        grid, np.column_stack([np.sin(2 * np.pi * grid[:, 0]), np.cos(2 * np.pi * grid[:, 0])])
    )
    assert abs(unrelated.correlation[0, 1]) < 0.5


def test_icm_mean_variance_and_gradient_have_the_one_shape() -> None:
    rng = np.random.default_rng(7)
    x = rng.random((10, 2))
    fitted = MultiOutputKriging().fit(x, np.column_stack([x[:, 0], x[:, 1] ** 2, x.sum(1)]))
    p = fitted.predict(rng.random((4, 2)), variance=True, gradient=True)
    assert p.mean.shape == (4, 3)
    assert p.variance.shape == (4, 3)
    assert p.gradient.shape == (4, 3, 2)


# -- D51 -------------------------------------------------------------------------------------------


def _forrester(x):
    high = (6 * x - 2) ** 2 * np.sin(12 * x - 4)
    return high, 0.5 * high + 10 * (x - 0.5) - 5


def test_co_kriging_beats_a_high_fidelity_gp_on_forrester_and_finds_the_scale() -> None:
    x_low = np.linspace(0, 1, 11)[:, None]
    x_high = np.array([[0.0], [0.4], [0.6], [1.0]])
    _, y_low = _forrester(x_low)
    y_high, _ = _forrester(x_high)
    x = np.vstack([x_low, x_high])
    y = np.vstack([y_low, y_high])
    level = np.array([0] * 11 + [1] * 4)
    fitted = CoKriging().fit(x, y, level)
    q = np.linspace(0, 1, 201)[:, None]
    truth, _ = _forrester(q)
    co = _rmse(fitted.predict(q).mean, truth)
    gp = _rmse(GaussianProcess().fit(x_high, y_high).predict(q).mean, truth)
    assert co < 0.2 * gp
    # f_high = 2 f_low + a linear discrepancy: the scale is 2 (dimensionless, ratio of the levels).
    assert abs(fitted.scales[0] - 2.0) <= 0.3


def test_co_kriging_variance_adds_up_through_the_levels() -> None:
    x_low = np.linspace(0, 1, 11)[:, None]
    x_high = np.array([[0.0], [0.4], [0.6], [1.0]])
    _, y_low = _forrester(x_low)
    y_high, _ = _forrester(x_high)
    fitted = CoKriging().fit(np.vstack([x_low, x_high]), np.vstack([y_low, y_high]), np.array([0] * 11 + [1] * 4))
    p = fitted.predict(np.array([[0.2], [0.8]]), variance=True, gradient=True)
    assert np.all(p.variance >= 0)
    assert p.gradient.shape == (2, 1, 1)
    # Planted control: a low level unrelated to the high one must lose the gain -- it came from the information.
    rng = np.random.default_rng(8)
    noise = rng.standard_normal((11, 1))
    unrelated = CoKriging().fit(np.vstack([x_low, x_high]), np.vstack([noise, y_high]), np.array([0] * 11 + [1] * 4))
    q = np.linspace(0, 1, 201)[:, None]
    truth, _ = _forrester(q)
    assert _rmse(unrelated.predict(q).mean, truth) > 5 * _rmse(fitted.predict(q).mean, truth)


def test_co_kriging_refuses_a_level_without_data() -> None:
    with pytest.raises(Refused, match='level'):
        CoKriging().fit(np.zeros((3, 1)), np.zeros((3, 1)), np.array([0, 0, 2]))
