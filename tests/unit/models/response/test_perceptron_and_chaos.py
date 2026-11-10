"""D54 multilayer perceptron (against scikit-learn's ``MLPRegressor``, tests only), D55 polynomial chaos.

The perceptron's training starts from its own seeded weights, so it is compared where comparison is
exact -- the forward pass and the loss at sklearn's fitted weights -- and by held-out error after
training. The chaos expansion is compared against CLOSED FORMS: an exact polynomial is recovered
exactly, and Ishigami's mean, variance and first-order Sobol indices are known analytically.
"""

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from sklearn.neural_network import MLPRegressor

from learn_lab.errors import Refused
from learn_lab.models.response import ChaosExpansion, MultilayerPerceptron, Network, legendre_table
from learn_lab.validity import BoxEnvelope


def _cube_of(x) -> BoxEnvelope:
    """The unit cube in the width of *x*: the declared domain of data drawn on [0, 1)."""
    return BoxEnvelope(np.zeros(np.shape(x)[1]), np.ones(np.shape(x)[1]))


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(12)
    x = rng.random((120, 2))
    y = np.column_stack([np.sin(3 * x[:, 0]) * x[:, 1], np.cos(2 * x[:, 1])])
    q = rng.random((200, 2))
    truth = np.column_stack([np.sin(3 * q[:, 0]) * q[:, 1], np.cos(2 * q[:, 1])])
    return x, y, q, truth


@pytest.fixture(scope='module')
def sklearn_net(data):
    x, y, _, _ = data
    return MLPRegressor(
        hidden_layer_sizes=(16, 16), activation='tanh', solver='lbfgs', alpha=1e-4, max_iter=3000, random_state=0
    ).fit(x, y)


@pytest.mark.parametrize('activation', ['tanh', 'relu', 'logistic', 'identity'])
def test_the_forward_pass_with_sklearns_weights_is_sklearns_prediction(data, activation) -> None:
    x, y, q, _ = data
    sk = MLPRegressor(hidden_layer_sizes=(8,), activation=activation, solver='lbfgs', max_iter=200, random_state=0).fit(
        x, y
    )
    net = Network(_cube_of(x), tuple(sk.coefs_), tuple(sk.intercepts_), activation)
    # Absolute, target units: the same matrix products.
    assert np.max(np.abs(net.predict(q).mean - sk.predict(q))) <= 1e-12


def test_the_loss_at_sklearns_weights_is_sklearns_loss(data, sklearn_net) -> None:
    x, y, _, _ = data
    net = Network(_cube_of(x), tuple(sklearn_net.coefs_), tuple(sklearn_net.intercepts_), 'tanh')
    loss, _ = net.loss(x, y, alpha=1e-4)
    assert abs(loss - sklearn_net.loss_) <= 1e-9 * abs(sklearn_net.loss_)  # relative, dimensionless


def test_the_loss_gradient_is_its_derivative(data) -> None:
    x, y, _, _ = data
    net = MultilayerPerceptron((5,), seed=3).initial(_cube_of(x), 2)
    _, gradient = net.loss(x, y, alpha=1e-3)
    flat = net.flat
    h = 1e-6
    for i in range(0, flat.size, 3):
        step = np.zeros_like(flat)
        step[i] = h
        plus = net.with_flat(flat + step).loss(x, y, alpha=1e-3)[0]
        minus = net.with_flat(flat - step).loss(x, y, alpha=1e-3)[0]
        assert abs(gradient[i] - (plus - minus) / (2 * h)) <= 1e-7  # absolute, loss units per weight


def test_training_is_as_accurate_as_sklearns(data, sklearn_net) -> None:
    x, y, q, truth = data
    ours = (
        MultilayerPerceptron((16, 16), alpha=1e-4, max_iterations=3000, seed=0)
        .fit(x, y, domain=_cube_of(x))
        .predict(q)
        .mean
    )
    mine = float(np.sqrt(np.mean((ours - truth) ** 2)))
    reference = float(np.sqrt(np.mean((sklearn_net.predict(q) - truth) ** 2)))
    # Relative to sklearn's held-out RMSE (dimensionless), with an absolute floor in target units.
    assert mine <= 2.0 * reference + 1e-3


def test_the_input_gradient_is_the_derivative_of_the_mean(data) -> None:
    x, y, q, _ = data
    fitted = MultilayerPerceptron((8,), max_iterations=200, seed=1).fit(x, y, domain=_cube_of(x))
    analytic = fitted.predict(q, gradient=True).gradient
    h = 1e-6
    for j in range(2):
        step = np.zeros(2)
        step[j] = h
        central = (fitted.predict(q + step).mean - fitted.predict(q - step).mean) / (2 * h)
        assert np.max(np.abs(analytic[:, :, j] - central)) <= 1e-7


# -- D55 -------------------------------------------------------------------------------------------


def test_the_legendre_basis_is_orthonormal_on_the_unit_interval() -> None:
    nodes, weights = leggauss(20)
    values, _ = legendre_table((nodes[:, None] + 1) / 2, 6)
    gram = np.einsum('i,ia,ib->ab', weights / 2, values[:, 0, :], values[:, 0, :])
    assert np.max(np.abs(gram - np.eye(7))) <= 1e-13  # absolute, dimensionless


def test_an_exact_polynomial_is_recovered_exactly() -> None:
    rng = np.random.default_rng(13)
    x = rng.random((60, 2))
    y = (1 + 2 * x[:, 0] - x[:, 0] * x[:, 1] ** 2)[:, None]
    fitted = ChaosExpansion(degree=3).fit(x, y, domain=_cube_of(x))
    q = rng.random((30, 2))
    assert np.max(np.abs(fitted.predict(q).mean[:, 0] - (1 + 2 * q[:, 0] - q[:, 0] * q[:, 1] ** 2))) <= 1e-12
    # Planted control: one degree short cannot represent the cubic term.
    short = ChaosExpansion(degree=2).fit(x, y, domain=_cube_of(x))
    assert np.max(np.abs(short.predict(q).mean - (1 + 2 * q[:, 0] - q[:, 0] * q[:, 1] ** 2)[:, None])) > 1e-3


def test_ishigami_moments_and_sobol_indices_match_the_closed_form() -> None:
    a, b = 7.0, 0.1
    rng = np.random.default_rng(14)
    x = rng.random((3000, 3))
    z = np.pi * (2 * x - 1)
    y = (np.sin(z[:, 0]) + a * np.sin(z[:, 1]) ** 2 + b * z[:, 2] ** 4 * np.sin(z[:, 0]))[:, None]
    fitted = ChaosExpansion(degree=12).fit(x, y, domain=_cube_of(x))
    mean = a / 2
    variance = a**2 / 8 + b * np.pi**4 / 5 + b**2 * np.pi**8 / 18 + 0.5
    first = np.array([(b * np.pi**4 / 5 + b**2 * np.pi**8 / 50 + 0.5) / variance, a**2 / 8 / variance, 0.0])
    # Absolute, in the output's units (mean), its square (variance), and dimensionless (indices).
    assert abs(fitted.mean[0] - mean) <= 1e-2
    assert abs(fitted.variance[0] - variance) <= 5e-2
    assert np.max(np.abs(fitted.first_order[:, 0] - first)) <= 1e-2
    assert fitted.total_order[2, 0] > 0.2  # x3 acts only through its interaction with x1


def test_the_chaos_gradient_is_the_derivative_of_the_mean() -> None:
    rng = np.random.default_rng(15)
    x = rng.random((80, 2))
    fitted = ChaosExpansion(degree=4).fit(x, np.exp(x[:, :1]) * x[:, 1:], domain=_cube_of(x))
    q = rng.random((10, 2))
    analytic = fitted.predict(q, gradient=True).gradient
    h = 1e-6
    for j in range(2):
        step = np.zeros(2)
        step[j] = h
        central = (fitted.predict(q + step).mean - fitted.predict(q - step).mean) / (2 * h)
        assert np.max(np.abs(analytic[:, :, j] - central)) <= 1e-7


def test_too_few_points_for_the_basis_is_refused() -> None:
    with pytest.raises(Refused, match='terms'):
        ChaosExpansion(degree=5).fit(
            np.random.default_rng(0).random((10, 3)),
            np.zeros((10, 1)),
            domain=_cube_of(np.random.default_rng(0).random((10, 3))),
        )


def test_training_moves_the_network_from_its_start(data) -> None:
    """Planted control: the untrained Glorot start must be far worse than the trained network."""
    x, y, q, truth = data
    model = MultilayerPerceptron((16,), max_iterations=500, seed=0)
    start = model.initial(_cube_of(x), 2).predict(q).mean
    trained = model.fit(x, y, domain=_cube_of(x)).predict(q).mean
    assert np.sqrt(np.mean((trained - truth) ** 2)) < 0.2 * np.sqrt(np.mean((start - truth) ** 2))
