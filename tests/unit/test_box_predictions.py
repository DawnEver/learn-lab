"""The box-wide prediction sample: a fit's predictions on seeded uniform draws over its rows' box, and its R^2."""

import numpy as np
import pytest

from learn_lab.models.response import Polynomial
from learn_lab.validation import box_predictions


def test_box_predictions_sample_the_rows_box_and_report_the_fit_r2():
    x = np.random.default_rng(1).random((20, 2))
    y = np.column_stack([x[:, 0] + 2 * x[:, 1], x[:, 0] * x[:, 1]])
    predicted, r2 = box_predictions(Polynomial(degree=2), x, y, samples=100, seed=5)
    sample = np.random.default_rng(5).uniform(x.min(axis=0), x.max(axis=0), (100, 2))
    np.testing.assert_allclose(predicted, np.column_stack([sample[:, 0] + 2 * sample[:, 1], sample.prod(axis=1)]))
    assert r2 == pytest.approx(1.0)
