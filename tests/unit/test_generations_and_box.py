"""Per-generation surrogate refits and the box-wide prediction sample.

A quadratic response surface reproduces a linear objective exactly, so every cross-validated error
is zero and every prediction is the objective itself -- closed forms, not captured runs.
"""

import numpy as np
import pytest

from learn_lab.errors import Refused
from learn_lab.generations import RunHistory, generation_errors, generation_predictions
from learn_lab.models.response import Polynomial
from learn_lab.validation import box_predictions


def _history():
    x = np.random.default_rng(0).random((24, 2)) * [2.0, 5.0] + [1.0, -1.0]
    objectives = {'a': 3.0 + 2.0 * x[:, 0] - x[:, 1], 'b': 10.0 + x[:, 1]}
    return RunHistory(axes=('p', 'q'), x=x, objectives=objectives, generation_rows=(12, 18, 24))


def test_row_generation_follows_the_cuts():
    assert _history().row_generation().tolist() == [0] * 12 + [1] * 6 + [2] * 6


def test_run_history_refuses_inconsistent_shapes_and_cuts():
    x = np.zeros((4, 2))
    with pytest.raises(Refused):
        RunHistory(('p',), x, {'a': np.zeros(4)}, (4,))
    with pytest.raises(Refused):
        RunHistory(('p', 'q'), x, {'a': np.zeros(3)}, (4,))
    with pytest.raises(Refused):
        RunHistory(('p', 'q'), x, {'a': np.zeros(4)}, (3, 3))


def test_run_history_refuses_a_first_generation_too_small_to_cross_validate():
    x = np.zeros((6, 2))
    with pytest.raises(Refused, match='first generation'):
        RunHistory(('p', 'q'), x, {'a': np.zeros(6)}, (1, 6))


def test_generation_errors_vanish_on_an_exactly_representable_objective():
    errors = generation_errors(_history(), key='polynomial')
    assert set(errors) == {'a', 'b'} and all(len(v) == 3 for v in errors.values())
    assert max(max(v) for v in errors.values()) < 1e-8


def test_generation_predictions_are_the_objective_on_fresh_seeded_samples():
    history = _history()
    points, predictions = generation_predictions(history, key='polynomial', samples=50, seed=7)
    lo, span = history.x.min(axis=0), np.ptp(history.x, axis=0)
    for k, (here, predicted) in enumerate(zip(points, predictions, strict=True)):
        np.testing.assert_allclose(here, lo + span * np.random.default_rng(7 + k).random((50, 2)))
        np.testing.assert_allclose(predicted['a'], 3.0 + 2.0 * here[:, 0] - here[:, 1], atol=1e-8)
        np.testing.assert_allclose(predicted['b'], 10.0 + here[:, 1], atol=1e-8)


def test_box_predictions_sample_the_rows_box_and_report_the_fit_r2():
    x = np.random.default_rng(1).random((20, 2))
    y = np.column_stack([x[:, 0] + 2 * x[:, 1], x[:, 0] * x[:, 1]])
    predicted, r2 = box_predictions(Polynomial(degree=2), x, y, samples=100, seed=5)
    sample = np.random.default_rng(5).uniform(x.min(axis=0), x.max(axis=0), (100, 2))
    np.testing.assert_allclose(predicted, np.column_stack([sample[:, 0] + 2 * sample[:, 1], sample.prod(axis=1)]))
    assert r2 == pytest.approx(1.0)
