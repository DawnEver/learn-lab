"""Conformal intervals cover what they claim on a planted small-n heteroscedastic problem; the GP's own does not.

The problem: ``y = sin 6 x1 + x2^2 / 2 + (0.02 + 0.4 x1^2) e`` on the unit square, a GP fitted on 24 rows -- few
rows, and noise whose sd varies 20-fold, which a homoscedastic GP cannot represent.

THE BAR, and its sample size. Split-conformal with ``n = 99`` calibration rows at ``c = 0.9`` takes the
``ceil(100 * 0.9) = 90``-th score, so its coverage of a fresh row is EXACTLY 90 / 100 in expectation over
calibration draws. One repetition's coverage (of 2000 test rows) has sd ``sqrt(0.9 * 0.1 / 101 + 0.09 / 2000)
~ 0.030``; the mean over 200 seeded repetitions has sd ~ 0.0021. The band ``[0.88, 0.92]`` is +-9 of those sds:
a calibrated interval cannot miss it by chance, and an interval 2 points off would fail. Measured
(2026-11-08): the GP's Gaussian 90 % observation interval covers 0.843; split-conformal 0.898; the GP
leave-one-out jackknife 0.884; the planted in-sample control 0.780.
"""

from __future__ import annotations

import numpy as np
import pytest

from learn_lab.calibration import conformal, jackknife, split_conformal
from learn_lab.errors import Refused
from learn_lab.models.response import GaussianProcess, NearestNeighbours
from learn_lab.protocol import Prediction
from learn_lab.validity import BoxEnvelope

CUBE = BoxEnvelope(np.zeros(2), np.ones(2))
Z90 = 1.6448536269514722
TRAIN, CALIBRATE, TEST, REPEATS = 24, 99, 2000, 200


def _draw(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    x = rng.uniform(0, 1, (n, 2))
    noise = (0.02 + 0.4 * x[:, 0] ** 2) * rng.standard_normal(n)
    return x, (np.sin(6 * x[:, 0]) + 0.5 * x[:, 1] ** 2 + noise)[:, None]


def _observed(fitted, x) -> Prediction:
    """The GP's predictive distribution of a new OBSERVATION: latent variance plus the fitted noise."""
    p = fitted.predict(x, variance=True)
    return Prediction(p.mean, p.variance + fitted.noise)


@pytest.fixture(scope='module')
def coverages() -> dict[str, float]:
    found: dict[str, list[float]] = {'gaussian': [], 'split': [], 'jackknife': [], 'in_sample': []}
    for r in range(REPEATS):
        rng = np.random.default_rng(r)
        x, y = _draw(rng, TRAIN + CALIBRATE)
        xs, ys = _draw(rng, TEST)
        fitted, split = split_conformal(
            GaussianProcess(restarts=1), x, y, coverage=0.9, share=CALIBRATE / (TRAIN + CALIBRATE), seed=r, domain=CUBE
        )
        test = _observed(fitted, xs)
        found['gaussian'].append(float(np.mean(np.abs(ys - test.mean) <= Z90 * np.sqrt(test.variance))))
        found['split'].append(float(split.report(ys, fitted.predict(xs, variance=True)).empirical[0]))
        found['jackknife'].append(float(jackknife(fitted, coverage=0.9).report(ys, test).empirical[0]))
        trained_on = np.column_stack([c.level + c.scale * c.values for c in fitted.columns])
        in_sample = conformal(trained_on, _observed(fitted, fitted.x), coverage=0.9)
        found['in_sample'].append(float(in_sample.report(ys, test).empirical[0]))
    return {name: float(np.mean(values)) for name, values in found.items()}


def test_split_conformal_covers_ninety_percent(coverages) -> None:
    assert 0.88 <= coverages['split'] <= 0.92, coverages


def test_the_gps_own_interval_under_covers(coverages) -> None:
    """What the calibration is FOR: without it the 90 % interval covers well under 90 % here."""
    assert coverages['gaussian'] < 0.87, coverages


def test_the_jackknife_spends_no_data_and_lands_near_the_target(coverages) -> None:
    """Approximate (the hyperparameters saw every row): a wider band, still well above the raw GP."""
    assert 0.86 <= coverages['jackknife'] <= 0.92, coverages
    assert coverages['jackknife'] > coverages['gaussian'] + 0.02, coverages


def test_planted_a_calibrator_scored_on_its_own_training_rows_under_covers(coverages) -> None:
    """PLANTED CONTROL: in-sample residuals are shrunk by the fit, so their quantile is too narrow."""
    assert coverages['in_sample'] < 0.85, coverages


def test_the_report_states_coverage_and_sharpness_in_the_values_units() -> None:
    measured = np.array([[0.0], [1.0], [2.0], [20.0]])
    interval = conformal(np.arange(19.0)[:, None], Prediction(np.zeros((19, 1))), coverage=0.9)
    assert interval.quantile[0] == 17.0  # ceil(20 * 0.9) = 18th smallest of 0..18
    report = interval.report(measured, Prediction(np.zeros((4, 1))))
    assert report.empirical[0] == 0.75
    assert report.width[0] == 34.0
    assert report.count == 4


def test_widening_is_the_gaussian_sd_factor_that_reproduces_the_interval() -> None:
    scores = np.linspace(0.0, 3.0, 199)[:, None]
    interval = conformal(scores, Prediction(np.zeros_like(scores), np.ones_like(scores)), coverage=0.9)
    assert interval.widening[0] == pytest.approx(interval.quantile[0] / Z90, rel=1e-12, abs=1e-12)


def test_too_few_rows_a_variance_mismatch_and_a_bad_coverage_are_refused() -> None:
    with pytest.raises(Refused, match='at least 9 rows'):
        conformal(np.zeros((8, 1)), Prediction(np.zeros((8, 1))), coverage=0.9)
    with pytest.raises(Refused, match='strictly between'):
        conformal(np.zeros((8, 1)), Prediction(np.zeros((8, 1))), coverage=1.0)
    interval = conformal(np.zeros((9, 1)), Prediction(np.zeros((9, 1))), coverage=0.9)
    with pytest.raises(Refused, match='this calibration used none'):
        interval.interval(Prediction(np.zeros((2, 1)), np.ones((2, 1))))


def test_split_conformal_takes_a_model_without_a_variance_as_a_constant_band() -> None:
    x, y = _draw(np.random.default_rng(0), 60)
    fitted, interval = split_conformal(NearestNeighbours(k=3), x, y, coverage=0.8, seed=2, domain=CUBE)
    lower, upper = interval.interval(fitted.predict(x))
    assert not interval.scaled
    np.testing.assert_allclose(upper - lower, 2 * interval.quantile[0], rtol=0, atol=1e-12)
