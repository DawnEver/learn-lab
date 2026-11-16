"""A conditional diffusion model learns a planted Gaussian random field whose mean depends on the condition.

``y = c * A * phi + S * (a1 * psi1 + a2 * psi2)``, ``a ~ N(0, 1)``, in tesla: the mean is ``c * A * phi`` and the
per-cell variance ``S^2 (psi1^2 + psi2^2)`` -- both closed form. Pinned to CPU float64.
"""

import numpy as np
import pytest

from learn_lab.deep.diffusion import DiffusionField
from learn_lab.deep.fit import Schedule
from learn_lab.deep.runtime import Runtime
from learn_lab.errors import Refused

N, A, S = 8, 1.0, 0.3  # grid side; mean amplitude and noise amplitude, T
MEAN_BAR = 0.06  # T: RMS over the grid of (sample mean - closed-form mean)
VARIANCE_BAR = 0.3  # relative error of the grid-averaged sample variance, dimensionless
# The rate is cosine-annealed: at a constant 5e-3 the loss's fresh noise keeps the weights jittering, and the sample
# mean of whichever snapshot the last step lands on read 0.027..0.405 T over steps 500..1000 and 0.041..0.057 T at
# step 1000 across seeds 0..2 on Windows -- a knife edge the Linux forge fell off. Annealed, the same sweep reads
# 0.008..0.036 T over steps 800..1000 and 0.012..0.030 T at step 1000.
# The variance was the under-fit net, not the sampler: width 8 x 1000 steps read -0.409 / -0.330 relative (seeds 0 / 1,
# posterior sampler; 'beta' -0.358 / -0.276), 200 noise steps -0.399 / -0.361. Width 16 x 3000 steps reads
# -0.168 / -0.187 (mean 0.003 / 0.040 T); 'beta' -0.125 / -0.156, so the default sampler stays 'posterior'.
QUERY, COUNT = 0.8, 128
CPU64 = Runtime(device='cpu', dtype='float64')
MODEL = DiffusionField(
    width=16,
    levels=1,
    noise_steps=50,
    schedule=Schedule(adam_steps=3000, learning_rate=5e-3, decay='cosine'),
    runtime=CPU64,
    chunk=None,
)

_s = (np.arange(N) + 0.5) / N
PHI = np.outer(np.sin(np.pi * _s), np.sin(np.pi * _s))
PSI1, PSI2 = np.outer(np.cos(np.pi * _s), np.ones(N)), np.outer(np.ones(N), np.cos(np.pi * _s))
VARIANCE = S**2 * (PSI1**2 + PSI2**2)


def _planted(count, seed):
    rng = np.random.default_rng(seed)
    c = rng.uniform(-1, 1, (count, 1))
    a = rng.standard_normal((count, 2))
    y = c[:, :, None] * A * PHI + S * (a[:, :1, None] * PSI1 + a[:, 1:, None] * PSI2)
    return c, y[:, None]


def _rms(a):
    return float(np.sqrt(np.mean(a**2)))


@pytest.fixture(scope='module')
def planted():
    return _planted(1024, 0)


@pytest.fixture(scope='module')
def fitted(planted):
    return MODEL.fit(*planted)


def test_the_sample_mean_matches_the_closed_form(fitted):
    mean, _ = fitted.moments(np.array([[QUERY]]), COUNT)
    assert _rms(mean[0, 0] - QUERY * A * PHI) < MEAN_BAR


def test_the_sample_variance_matches_the_closed_form(fitted):
    _, variance = fitted.moments(np.array([[QUERY]]), COUNT)
    assert abs(variance[0, 0].mean() / VARIANCE.mean() - 1) < VARIANCE_BAR


def test_the_beta_sampler_disperses_more_and_an_unknown_one_is_refused(fitted):
    query = np.array([[QUERY]])
    posterior = fitted.moments(query, COUNT)[1].mean()
    assert fitted.moments(query, COUNT, variance='beta')[1].mean() > posterior
    with pytest.raises(Refused):
        fitted.sample(query, 2, variance='learned')


def test_shuffled_conditions_miss_the_mean(planted):
    """PLANTED CONTROL: the same model on shuffled conditions learns the population, not the conditional mean."""
    c, y = planted
    control = MODEL.fit(np.random.default_rng(1).permutation(c), y)
    mean, _ = control.moments(np.array([[QUERY]]), COUNT)
    error = _rms(mean[0, 0] - QUERY * A * PHI)
    assert error > MEAN_BAR


def test_samples_are_shaped_and_reproducible(fitted):
    first = fitted.sample(np.array([[0.1], [-0.4]]), 3, seed=7)
    assert first.shape == (2, 3, 1, N, N)
    assert np.array_equal(first, fitted.sample(np.array([[0.1], [-0.4]]), 3, seed=7))
    assert not np.array_equal(first, fitted.sample(np.array([[0.1], [-0.4]]), 3, seed=8))


def test_a_condition_outside_the_envelope_is_refused(fitted):
    fitted.sample(np.array([[0.9]]), 1)  # PLANTED CONTROL: inside, it samples
    with pytest.raises(Refused):
        fitted.sample(np.array([[1.5]]), 1)
    with pytest.raises(Refused):
        fitted.sample(np.array([[0.1, 0.2]]), 1)


def test_a_stochastic_loss_refuses_an_lbfgs_polish():
    with pytest.raises(Refused):
        DiffusionField(schedule=Schedule(adam_steps=10, lbfgs_steps=5))
