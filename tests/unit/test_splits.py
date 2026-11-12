"""k-fold and region-held-out splits partition the data as declared, and their per-split scores expose an overfit.

PLANTED CONTROL: a 1-nearest-neighbour model reproduces every training row (in-sample error zero), so it scores
worse on every held-out fold and on the held-out region than in sample; a fold set that dropped or repeated a
row, or a region that leaked a row across its boundary, fails the partition checks.
"""

import numpy as np
import pytest

from learn_lab.errors import Refused
from learn_lab.models.response.neighbours import NearestNeighbours
from learn_lab.validation import kfold, region, split_scores

RNG = np.random.default_rng(0)
X = RNG.uniform(0, 1, (60, 2))
Y = (np.sin(4 * X[:, 0]) + X[:, 1] + 0.1 * RNG.standard_normal(60))[:, None]
#: The noise is 0.1 in y's unit; a held-out RMS below this would mean the noise was memorised.
NOISE = 0.1


@pytest.mark.parametrize('seed', [None, 3])
def test_k_folds_partition_the_rows_exactly(seed) -> None:
    splits = kfold(len(X), 5, seed=seed)
    tests = np.concatenate([test for _, test in splits])
    assert np.array_equal(np.sort(tests), np.arange(len(X)))
    for train, test in splits:
        assert len(np.intersect1d(train, test)) == 0
        assert np.array_equal(np.sort(np.concatenate([train, test])), np.arange(len(X)))


def test_the_region_split_holds_out_exactly_the_declared_box() -> None:
    lower, upper = np.array([0.2, 0.5]), np.array([0.6, 1.0])
    train, test = region(X, lower, upper)
    inside = np.all((lower <= X) & (upper >= X), axis=1)
    assert np.array_equal(test, np.flatnonzero(inside))
    assert np.array_equal(train, np.flatnonzero(~inside))
    assert 0 < len(test) < len(X)


def test_a_region_holding_every_row_or_none_is_refused() -> None:
    with pytest.raises(Refused, match='hold some rows'):
        region(X, [0, 0], [1, 1])
    with pytest.raises(Refused, match='hold some rows'):
        region(X, [2, 2], [3, 3])


def test_an_overfit_model_scores_worse_held_out_than_in_sample() -> None:
    overfit = NearestNeighbours(k=1)
    folds = split_scores(overfit, X, Y, kfold(len(X), 5, seed=1))
    assert len(folds) == 5
    for fold in folds:
        assert fold.in_sample == pytest.approx(0.0, abs=1e-12)  # y's unit: it memorised its rows
        assert fold.held_out > NOISE
    (held,) = split_scores(overfit, X, Y, [region(X, [0.2, 0.5], [0.6, 1.0])])
    assert held.held_out > held.in_sample + NOISE


def test_a_split_that_tests_on_its_training_rows_is_refused() -> None:
    rows = np.arange(len(X))
    with pytest.raises(Refused, match='disjoint'):
        split_scores(NearestNeighbours(k=1), X, Y, [(rows, rows[:5])])
