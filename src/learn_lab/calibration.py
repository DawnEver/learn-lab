"""Calibrated predictive intervals: conformal prediction, MEASURED on data the fit did not see.

A model's own predictive spread is a claim; a conformal interval is that claim rescaled until it covers held-out
residuals at the requested rate. The conformity score of a held-out row is ``|y - mean| / sd`` per output (``sd``
the prediction's own standard deviation; ``1`` for a model that gives no variance, which makes the interval a
constant band), and the interval at a new point is ``mean +- q sd`` with ``q`` the ``ceil((n + 1) c)``-th smallest
of the ``n`` scores for coverage ``c``. For exchangeable rows that interval covers a new row with probability in
``[c, c + 1 / (n + 1))`` over the draw of the calibration set -- the guarantee, and the reason the scores must
come from rows the model was NOT fitted on: in-sample residuals are shrunk by the fit itself, so a quantile of
them under-covers (the planted control in the tests).

* :func:`conformal` -- the interval calibrated on the caller's held-out ``(measured, prediction)``;
* :func:`split_conformal` -- fit on one seeded part of the data, calibrate on the rest: the split is made here,
  so the held-out rule cannot be broken by accident;
* :func:`jackknife` -- for a Gaussian process, no data spent: the scores are its closed-form leave-one-out
  residuals (each row predicted without itself) standardised by the held-out OBSERVATION sd. Its interval is
  applied to the observation sd ``sqrt(latent variance + noise)``. The hyperparameters still saw every row, so
  this is the jackknife's approximate guarantee, not split-conformal's exact one;
* :meth:`ConformalInterval.report` -- the calibration report on a further held-out set: empirical coverage
  against the target, and sharpness (the mean interval width, in the values' units).

The Gaussian-z view: :attr:`ConformalInterval.widening` is ``q / z_c`` -- the factor a Gaussian sd must be widened
by so its ``c`` interval is the calibrated one. It is measured per fit; it replaces any fixed factor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri

from learn_lab.errors import refuse
from learn_lab.models.response.gaussian_process import FittedGaussianProcess
from learn_lab.protocol import Fitted, Give, Prediction, Surrogate, training_data
from learn_lab.validity import BoxEnvelope

__all__ = ['ConformalInterval', 'IntervalReport', 'conformal', 'jackknife', 'split_conformal']

#: Dimensionless slack on the rank ``(n + 1) c``: ``0.9 * 10`` is ``9.000000000000002`` in floating point, and
#: a rank that rounds up past an exact integer would demand one more row than the definition does.
_ROUNDING = 1e-9


def _coverage(coverage: float) -> float:
    if not 0 < coverage < 1:
        refuse(f'a coverage is a probability strictly between 0 and 1, got {coverage!r}')
    return float(coverage)


def _matrix(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[:, None] if array.ndim == 1 else array


def _spread(prediction: Prediction) -> np.ndarray:
    """The prediction's sd per row and output, or ones when it carries no variance."""
    if prediction.variance is None:
        return np.ones_like(prediction.mean)
    if np.any(prediction.variance <= 0):
        refuse('a conformal score divides by the predicted sd; it needs a variance > 0 at every point')
    return np.sqrt(prediction.variance)


@dataclass(frozen=True, slots=True)
class IntervalReport:
    """Target and empirical coverage per output, mean width per output (values' units), and the row count."""

    coverage: float
    empirical: np.ndarray
    width: np.ndarray
    count: int


@dataclass(frozen=True, slots=True, eq=False)
class ConformalInterval:
    """The calibrated score quantile ``(k,)`` at *coverage*, from *count* held-out rows, scaled by sd or not."""

    coverage: float
    quantile: np.ndarray
    count: int
    scaled: bool

    def interval(self, prediction: Prediction) -> tuple[np.ndarray, np.ndarray]:
        """``(lower, upper)`` ``(m, k)`` around *prediction* -- which carries a variance iff calibration's did."""
        if (prediction.variance is not None) != self.scaled:
            refuse(
                'the interval scales the sd the calibration scored with: '
                f'{"a variance is needed" if self.scaled else "this calibration used none"}'
            )
        if prediction.mean.shape[1] != self.quantile.size:
            refuse(f'calibrated for {self.quantile.size} output(s), the prediction has {prediction.mean.shape[1]}')
        half = self.quantile * _spread(prediction)
        return prediction.mean - half, prediction.mean + half

    @property
    def widening(self) -> np.ndarray:
        """``q / z_c`` per output: the factor a Gaussian sd is widened by to give this interval (1 when calibrated)."""
        return self.quantile / ndtri(0.5 + self.coverage / 2)

    def report(self, measured: object, prediction: Prediction) -> IntervalReport:
        """Empirical coverage and sharpness on *measured* -- rows held out of the fit AND of the calibration."""
        m = _matrix(measured)
        if m.shape != prediction.mean.shape:
            refuse(f'measured {m.shape} and predicted {prediction.mean.shape} values must have one shape')
        lower, upper = self.interval(prediction)
        inside = (m >= lower) & (m <= upper)
        return IntervalReport(self.coverage, inside.mean(axis=0), (upper - lower).mean(axis=0), len(m))


def conformal(measured: object, prediction: Prediction, *, coverage: float) -> ConformalInterval:
    """The interval at *coverage* calibrated on rows the model was NOT fitted on (see the module docstring)."""
    coverage = _coverage(coverage)
    m = _matrix(measured)
    if m.shape != prediction.mean.shape:
        refuse(f'measured {m.shape} and predicted {prediction.mean.shape} values must have one shape')
    n = len(m)
    rank = int(np.ceil((n + 1) * coverage - _ROUNDING))
    if rank > n:
        least = int(np.ceil(coverage / (1 - coverage) - _ROUNDING))
        refuse(f'a {coverage} conformal interval needs at least {least} rows, got {n}')
    scores = np.sort(np.abs(m - prediction.mean) / _spread(prediction), axis=0)
    return ConformalInterval(coverage, scores[rank - 1], n, prediction.variance is not None)


def split_conformal(
    surrogate: Surrogate,
    x: object,
    y: object,
    *,
    coverage: float,
    share: float = 0.25,
    seed: int = 0,
    domain: BoxEnvelope | None = None,
) -> tuple[Fitted, ConformalInterval]:
    """*surrogate* fitted on a seeded ``1 - share`` of the rows, calibrated at *coverage* on the other *share*.

    The fit's envelope is *domain*, else the box ALL rows span, so no calibration row is out of domain. The
    scores are scaled by the fit's own ``predict(variance=)`` (latent, for a GP), so the interval is applied to
    exactly that prediction.
    """
    x, y, envelope = training_data(x, y, domain)
    if not 0 < share < 1:
        refuse(f'the calibration share is strictly between 0 and 1, got {share!r}')
    order = np.random.default_rng(seed).permutation(len(x))
    held = round(share * len(x))
    if held < 1 or held == len(x):
        refuse(f'a share of {share} of {len(x)} rows leaves no rows to fit on or none to calibrate on')
    calibrate, train = order[:held], order[held:]
    fitted = surrogate.fit(x[train], y[train], domain=envelope)
    prediction = fitted.predict(x[calibrate], variance=Give.VARIANCE in fitted.gives)
    return fitted, conformal(y[calibrate], prediction, coverage=coverage)


def jackknife(fitted: FittedGaussianProcess, *, coverage: float) -> ConformalInterval:
    """The interval at *coverage* from the GP's leave-one-out residuals (observation sd; see the module docstring)."""
    loo = fitted.leave_one_out()
    measured = np.column_stack([c.level + c.scale * c.values for c in fitted.columns])
    return conformal(measured, loo, coverage=coverage)
