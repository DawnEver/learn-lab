"""The equivalence check admits a faithful candidate and refuses a degraded one on the model's own statistic."""

import numpy as np
import pytest

from learn_lab.equivalence import Statistic, Tolerance, equivalence
from learn_lab.errors import Refused

RNG = np.random.default_rng(0)
X = RNG.uniform(-1, 1, (64, 3))
W = np.array([[1.0], [-2.0], [0.5]])


def _truth(x):
    return x @ W


RELATIVE_L2 = Statistic(
    'relative_l2', 'dimensionless', lambda p, x: float(np.linalg.norm(p - _truth(x)) / np.linalg.norm(_truth(x)))
)
BAR = Tolerance(1e-3, 'dimensionless')


def test_a_faithful_candidate_is_reported_beside_its_reference():
    report = equivalence(_truth, lambda x: _truth(x) + 1e-6, X, statistic=RELATIVE_L2, tolerance=BAR)
    assert report.reference == pytest.approx(0.0, abs=1e-12)
    assert 0 < report.candidate < BAR.value
    assert report.max_deviation == pytest.approx(1e-6, rel=1e-6, abs=1e-12)


def test_a_candidate_with_perturbed_weights_is_refused():
    corrupted = W + 0.05 * RNG.standard_normal(W.shape)  # PLANTED: the check must be able to say no
    with pytest.raises(Refused, match='worse'):
        equivalence(_truth, lambda x: x @ corrupted, X, statistic=RELATIVE_L2, tolerance=BAR)


def test_a_tolerance_in_another_unit_or_without_one_is_refused():
    with pytest.raises(Refused, match='cannot bound'):
        equivalence(_truth, _truth, X, statistic=RELATIVE_L2, tolerance=Tolerance(1e-3, 'T'))
    with pytest.raises(Refused, match='unit'):
        Tolerance(1e-3, '')
