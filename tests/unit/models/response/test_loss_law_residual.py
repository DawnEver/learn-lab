"""A loss scaling law as the mean, a learned residual on top: ``P = law(x) + residual(x)``.

The planted law is the iron Bertotti split plus the magnet eddy loss, each ``P = c f^a D^b`` with the textbook
exponents; the drivers are named columns (speed and pole pairs -> frequency, psi_d/psi_q -> flux proxy,
i_d/i_q -> current). Losses are in watts, so every tolerance is an absolute one in watts.
"""

import numpy as np
import pytest

from learn_lab import artifact
from learn_lab.errors import Refused
from learn_lab.models.response import GaussianProcess, LossLawResidual
from learn_lab.validity import BoxEnvelope

INPUTS = ('speed', 'pole_pairs', 'psi_d', 'psi_q', 'i_d', 'i_q')
TERMS = ('hysteresis', 'classical', 'excess', 'magnet_eddy')
LEVELS = np.array([2e-2, 4e-5, 1.5e-3, 1e-5])  # W at f = 1 Hz, D = 1 in the drivers' units
TEXTBOOK = {'hysteresis': (1.0, 2.0), 'classical': (2.0, 2.0), 'excess': (1.5, 1.5), 'magnet_eddy': (2.0, 2.0)}


def drivers(n: int, speed: tuple[float, float], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.uniform(*speed, n),  # rpm
        np.full(n, 4.0),
        rng.uniform(0.03, 0.08, n),  # Wb
        rng.uniform(0.02, 0.10, n),
        rng.uniform(-200.0, -20.0, n),  # A
        rng.uniform(50.0, 300.0, n),
    ])


def law(x: np.ndarray) -> np.ndarray:
    f = x[:, 0] * x[:, 1] / 60.0
    flux, current = np.hypot(x[:, 2], x[:, 3]), np.hypot(x[:, 4], x[:, 5])
    cols = []
    for term, c in zip(TERMS, LEVELS, strict=True):
        a, b = TEXTBOOK[term]
        cols.append(c * f**a * (current if term == 'magnet_eddy' else flux) ** b)
    return np.column_stack(cols)


TRAIN = drivers(30, (1000.0, 6000.0), seed=1)
DOMAIN = BoxEnvelope(
    np.array([500.0, 4.0, 0.03, 0.02, -200.0, 50.0]), np.array([10000.0, 4.0, 0.08, 0.10, -20.0, 300.0])
)
BEYOND = drivers(20, (8000.0, 10000.0), seed=2)  # above the training speed range, inside the declared domain
MODEL = LossLawResidual(inputs=INPUTS, terms=TERMS, base=GaussianProcess(restarts=1))

#: Rounding through a log-space level and a GP on a ~1e-12 W residual: far below a milliwatt.
ON_THE_LAW = 1e-6  # W


def test_textbook_exponents_are_the_default_and_the_declaration_resolves_the_drivers() -> None:
    assert MODEL.exponents == tuple(TEXTBOOK[t] for t in TERMS)


def test_data_exactly_on_the_law_leaves_no_residual_and_extrapolates_on_the_law() -> None:
    fitted = MODEL.fit(TRAIN, law(TRAIN), domain=DOMAIN)
    np.testing.assert_allclose(fitted.law(TRAIN), law(TRAIN), rtol=0, atol=ON_THE_LAW)
    np.testing.assert_allclose(fitted.predict(TRAIN).mean, law(TRAIN), rtol=0, atol=ON_THE_LAW)
    np.testing.assert_allclose(fitted.predict(BEYOND).mean, law(BEYOND), rtol=0, atol=ON_THE_LAW)


def test_a_plain_gp_leaves_the_law_beyond_the_training_speeds() -> None:
    """PLANTED CONTROL for the extrapolation above: without the law the same GP misses by watts."""
    plain = GaussianProcess(restarts=1).fit(TRAIN, law(TRAIN), domain=DOMAIN)
    miss = np.max(np.abs(plain.predict(BEYOND).mean - law(BEYOND)))
    assert miss > 1.0  # W


def bent(x: np.ndarray) -> np.ndarray:
    """The law plus a smooth bend in speed the law cannot hold: 20 % of each term's mean training loss."""
    return law(x) + 0.2 * np.mean(law(TRAIN), axis=0) * np.sin(x[:, :1] / 1500.0)


def rms(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The RMS error per loss term, W."""
    return np.sqrt(np.mean((a - b) ** 2, axis=0))


def test_a_bent_law_is_beaten_by_the_family_against_the_law_and_the_gp_alone() -> None:
    """The GP alone is given the same unit-box inputs the family's residual sees, so only the law differs."""
    train, held = drivers(40, (1000.0, 6000.0), seed=3), drivers(40, (1200.0, 5800.0), seed=4)
    lower, upper = np.array([1000.0, 4.0, 0.03, 0.02, -200.0, 50.0]), np.array([6000.0, 4.0, 0.08, 0.10, -20.0, 300.0])
    fitted = MODEL.fit(train, bent(train), domain=BoxEnvelope(lower, upper))
    family = rms(fitted.predict(held).mean, bent(held))
    law_alone = rms(fitted.law(held), bent(held))
    span = np.where(upper > lower, upper - lower, 1.0)
    gp = GaussianProcess(restarts=1).fit(
        (train - lower) / span, bent(train), domain=BoxEnvelope(np.zeros(6), np.ones(6))
    )
    gp_alone = rms(gp.predict((held - lower) / span).mean, bent(held))
    assert np.all(family < 0.5 * law_alone), (family, law_alone)
    assert np.all(family < 0.5 * gp_alone), (family, gp_alone)


def test_free_exponents_are_fitted_in_log_space() -> None:
    off = np.array([1.3, 2.0, 1.5, 2.0])
    y = law(TRAIN) * (TRAIN[:, :1] * 4.0 / 60.0) ** (off - np.array([1.0, 2.0, 1.5, 2.0]))
    fitted = LossLawResidual(inputs=INPUTS, terms=TERMS, fit_exponents=True, base=GaussianProcess(restarts=0)).fit(
        TRAIN, y
    )
    np.testing.assert_allclose(fitted.exponents[:, 0], off, rtol=0, atol=1e-9)  # dimensionless
    np.testing.assert_allclose(fitted.law(TRAIN), y, rtol=0, atol=ON_THE_LAW)


def test_variance_and_gradient_pass_through_the_residual() -> None:
    fitted = MODEL.fit(TRAIN, law(TRAIN))
    p = fitted.predict(TRAIN[:3], variance=True, gradient=True)
    assert p.variance.shape == (3, 4) and p.gradient.shape == (3, 4, 6)
    step = np.zeros(6)
    step[0] = 1e-3
    numeric = (fitted.predict(TRAIN[:3] + step).mean - fitted.predict(TRAIN[:3] - step).mean) / 2e-3
    np.testing.assert_allclose(p.gradient[:, :, 0], numeric, rtol=0, atol=1e-6)  # W/rpm


def test_frequency_and_current_columns_declare_the_drivers_directly() -> None:
    f = TRAIN[:, 0] * TRAIN[:, 1] / 60.0
    x = np.column_stack([f, np.hypot(TRAIN[:, 2], TRAIN[:, 3]), np.hypot(TRAIN[:, 4], TRAIN[:, 5])])
    fitted = LossLawResidual(inputs=('frequency', 'flux', 'current'), terms=TERMS, base=GaussianProcess(restarts=0))
    np.testing.assert_allclose(fitted.fit(x, law(TRAIN)).law(x), law(TRAIN), rtol=0, atol=ON_THE_LAW)


@pytest.mark.parametrize(
    ('inputs', 'terms', 'match'),
    [
        (('frequency', 'speed', 'flux'), ('classical',), 'frequency'),
        (('frequency', 'psi_d'), ('classical',), 'psi_q'),
        (('frequency', 'flux'), ('magnet_eddy',), 'current'),
        (('speed', 'flux'), ('eddy',), 'term'),
        (('speed', 'speed', 'flux'), ('classical',), 'once'),
    ],
)
def test_an_ill_declared_law_is_refused(inputs, terms, match) -> None:
    with pytest.raises(Refused, match=match):
        LossLawResidual(inputs=inputs, terms=terms)


def test_a_negligible_reference_is_refused_never_answered() -> None:
    y = law(TRAIN)
    y[5, 1] = y[:, 1].max() * 1e-12
    with pytest.raises(Refused, match='negligible'):
        MODEL.fit(TRAIN, y)


def test_the_artifact_reloads_the_fit_bit_for_bit(tmp_path) -> None:
    fitted = MODEL.fit(TRAIN, law(TRAIN))
    artifact.save(fitted, tmp_path / 'm', declaration=MODEL, data_identity='sha256:planted')
    again = artifact.load(tmp_path / 'm')
    assert np.array_equal(again.predict(TRAIN).mean, fitted.predict(TRAIN).mean)
    assert artifact.manifest(tmp_path / 'm')['family'] == 'loss_law_residual'
