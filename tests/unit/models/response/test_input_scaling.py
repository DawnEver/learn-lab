"""Every Kriging family scales its inputs onto the unit box of its envelope before the kernel.

Physical inputs differ by orders of magnitude (speed ~1e3 rad/s against flux ~1e-2 Wb); a kernel on the
raw coordinates cannot place one length scale per input inside its bounds and learns near-nothing. Each
family is fitted on a smooth function of x1 in [1e3, 3e3] and x2 in [1e-2, 3e-2] and must predict held-out
points within :data:`ABSOLUTE` (output units) -- and, the planted control against a near-constant
predictor, the spread of its held-out predictions must track the truth's within :data:`SPREAD`.
"""

import numpy as np
import pytest

from learn_lab.models.response import CoKriging, GaussianProcess, GradientEnhancedKriging, MultiOutputKriging
from learn_lab.validity import BoxEnvelope

LOWER, UPPER = np.array([1e3, 1e-2]), np.array([3e3, 3e-2])
#: Absolute, output units (the truth spans about 2.5): worst held-out error.
ABSOLUTE = 5e-2
#: Dimensionless: |std(prediction) / std(truth) - 1|.
SPREAD = 5e-2


def _truth(x: np.ndarray) -> np.ndarray:
    u = (x - LOWER) / (UPPER - LOWER)
    return np.sin(3 * u[:, 0]) * (1 + u[:, 1]) + u[:, 1] ** 2


def _slopes(x: np.ndarray) -> np.ndarray:
    u = (x - LOWER) / (UPPER - LOWER)
    du = np.column_stack([3 * np.cos(3 * u[:, 0]) * (1 + u[:, 1]), np.sin(3 * u[:, 0]) + 2 * u[:, 1]])
    return du / (UPPER - LOWER)


def _points(n: int, seed: int) -> np.ndarray:
    return LOWER + (UPPER - LOWER) * np.random.default_rng(seed).random((n, 2))


DOMAIN = BoxEnvelope(LOWER, UPPER)
TRAIN, HELD = _points(30, 0), _points(200, 1)


def _judge(predicted: np.ndarray, truth: np.ndarray) -> None:
    assert np.max(np.abs(predicted - truth)) < ABSOLUTE
    assert abs(np.std(predicted) / np.std(truth) - 1) < SPREAD


@pytest.mark.parametrize('restarts', [0, 4])
@pytest.mark.parametrize('domain', [DOMAIN, None], ids=['declared', 'fitted'])
def test_a_gp_fits_inputs_of_wildly_different_scales(domain, restarts) -> None:
    """``restarts=0`` pins the declared start (unit lengths), which is meaningful on the unit box only."""
    fitted = GaussianProcess(restarts=restarts).fit(TRAIN, _truth(TRAIN)[:, None], domain=domain)
    inside = HELD[fitted.envelope.classify(HELD) == fitted.envelope.classify(HELD[:1])[0]]
    _judge(fitted.predict(inside).mean[:, 0], _truth(inside))


def test_a_scaled_gp_gradient_is_in_the_raw_inputs_units() -> None:
    fitted = GaussianProcess().fit(TRAIN, _truth(TRAIN)[:, None], domain=DOMAIN)
    q = HELD[:20]
    gradient = fitted.predict(q, gradient=True).gradient[:, 0, :]
    step = 1e-6 * (UPPER - LOWER)
    numeric = np.column_stack([
        (fitted.predict(q + step * e).mean[:, 0] - fitted.predict(q - step * e).mean[:, 0]) / (2 * step[j])
        for j, e in enumerate(np.eye(2))
    ])
    assert np.allclose(gradient, numeric, rtol=1e-5, atol=1e-6 / (UPPER - LOWER))
    assert np.allclose(gradient, _slopes(q), rtol=0, atol=0.1 / (UPPER - LOWER))


def test_an_icm_fits_inputs_of_wildly_different_scales() -> None:
    y = np.column_stack([_truth(TRAIN), 2 * _truth(TRAIN) + 1])
    fitted = MultiOutputKriging(restarts=0).fit(TRAIN, y, domain=DOMAIN)
    _judge(fitted.predict(HELD).mean[:, 0], _truth(HELD))


def test_co_kriging_fits_inputs_of_wildly_different_scales() -> None:
    cheap = _points(40, 2)
    x = np.vstack([cheap, TRAIN[:15]])
    y = np.concatenate([0.5 * _truth(cheap) + 0.2, _truth(TRAIN[:15])])[:, None]
    level = np.r_[np.zeros(40, int), np.ones(15, int)]
    fitted = CoKriging(restarts=0).fit(x, y, level, domain=DOMAIN)
    _judge(fitted.predict(HELD).mean[:, 0], _truth(HELD))


def test_gek_fits_inputs_of_wildly_different_scales() -> None:
    x = TRAIN[:15]
    fitted = GradientEnhancedKriging(restarts=0).fit(x, _truth(x)[:, None], _slopes(x)[:, None, :], domain=DOMAIN)
    p = fitted.predict(HELD, gradient=True)
    _judge(p.mean[:, 0], _truth(HELD))
    assert np.allclose(p.gradient[:, 0, :], _slopes(HELD), rtol=0, atol=0.1 / (UPPER - LOWER))
