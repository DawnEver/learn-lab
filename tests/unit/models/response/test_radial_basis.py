"""D44 -- radial basis interpolation against ``scipy.interpolate.RBFInterpolator`` (test oracle only).

Tolerances are ABSOLUTE in the target's units (dimensionless, order 1).
"""

import numpy as np
import pytest
from scipy.interpolate import RBFInterpolator

from learn_lab.errors import Refused
from learn_lab.models.response import RADIAL_KERNELS, RadialBasis

#: Absolute, target units: two direct solves of the same saddle system (condition ~1e6 at most here).
AGREE = 1e-8


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(11)
    x = rng.random((30, 2))
    y = np.column_stack([np.sin(4 * x[:, 0]) * np.cos(3 * x[:, 1]), x[:, 0] * x[:, 1]])
    return x, y, rng.random((20, 2))


@pytest.mark.parametrize('kernel', RADIAL_KERNELS)
@pytest.mark.parametrize('smoothing', [0.0, 1e-3])
def test_every_kernel_agrees_with_scipy(data, kernel, smoothing) -> None:
    x, y, q = data
    epsilon = 2.0
    degree = 1 if kernel != 'quintic' else 2
    ours = RadialBasis(kernel, epsilon=epsilon, smoothing=smoothing, degree=degree).fit(x, y).predict(q).mean
    theirs = RBFInterpolator(x, y, kernel=kernel, epsilon=epsilon, smoothing=smoothing, degree=degree)(q)
    assert np.max(np.abs(ours - theirs)) <= AGREE


def test_without_smoothing_the_surface_interpolates(data) -> None:
    x, y, _ = data
    fitted = RadialBasis('thin_plate_spline').fit(x, y)
    assert np.max(np.abs(fitted.predict(x).mean - y)) <= AGREE
    smoothed = RadialBasis('thin_plate_spline', smoothing=1.0).fit(x, y)
    assert np.max(np.abs(smoothed.predict(x).mean - y)) > 1e-3  # planted: smoothing must leave a residual


@pytest.mark.parametrize('kernel', RADIAL_KERNELS)
def test_the_gradient_is_the_derivative_of_the_mean(data, kernel) -> None:
    x, y, q = data
    fitted = RadialBasis(kernel, epsilon=2.0, degree=2).fit(x, y)
    analytic = fitted.predict(q, gradient=True).gradient
    h = 1e-6
    for j in range(2):
        step = np.zeros(2)
        step[j] = h
        central = (fitted.predict(q + step).mean - fitted.predict(q - step).mean) / (2 * h)
        # Absolute, target units per unit input; the kinked kernels (linear) are smooth away from centres.
        assert np.max(np.abs(analytic[:, :, j] - central)) <= 1e-5


def test_a_degree_below_the_kernels_minimum_is_refused_and_a_stranger_kernel_names_the_set() -> None:
    with pytest.raises(Refused, match='thin_plate_spline'):
        RadialBasis('thin_plate_spline', degree=0)
    with pytest.raises(Refused, match='gaussian'):
        RadialBasis('no_such_kernel')
