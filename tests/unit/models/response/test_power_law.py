"""Power laws ``y = exp(c) prod x_i ** a_i``, fitted in log space.

The planted law is a Steinmetz-like loss ``P = 0.02 f^1.4 B^2.1`` (W, Hz, T) beside a second output
``Q = 3 f^-0.5 B^1``. Exponents are dimensionless; levels are compared in log space (dimensionless too).
"""

import numpy as np
import pytest

from learn_lab import artifact
from learn_lab.errors import Refused
from learn_lab.models.response import PowerLaw
from learn_lab.validation import cross_validate

#: Dimensionless (exponents, log levels): a least-squares solve on exact data recovers them to rounding.
EXACT = 1e-9

RNG = np.random.default_rng(11)
X = np.column_stack([RNG.uniform(50.0, 1000.0, 40), RNG.uniform(0.1, 1.8, 40)])  # f in Hz, B in T
A = np.array([[1.4, -0.5], [2.1, 1.0]])  # (inputs, outputs)
C = np.log([0.02, 3.0])


def law(x: np.ndarray) -> np.ndarray:
    return np.exp(C + np.log(x) @ A)


def test_planted_exponents_and_levels_are_recovered_on_exact_data() -> None:
    fitted = PowerLaw().fit(X, law(X))
    np.testing.assert_allclose(fitted.exponents, A, rtol=0, atol=EXACT)
    np.testing.assert_allclose(fitted.level, C, rtol=0, atol=EXACT)


def test_fixed_exponents_are_honoured_and_only_the_level_is_fitted() -> None:
    noisy = law(X) * np.exp(RNG.normal(0.0, 0.05, (40, 2)))
    pinned = PowerLaw(fixed=((1.5, 2.0), (None, 1.0))).fit(X, noisy)
    assert pinned.exponents[:, 0].tolist() == [1.5, 2.0]
    assert pinned.exponents[1, 1] == 1.0
    # every exponent of output 0 pinned: its level is the mean log residual, nothing else moves
    expected = np.mean(np.log(noisy[:, 0]) - np.log(X) @ np.array([1.5, 2.0]))
    assert pinned.level[0] == pytest.approx(expected, abs=EXACT)


def test_a_fixed_table_of_the_wrong_shape_is_refused() -> None:
    with pytest.raises(Refused, match=r'\(2, 2\) table'):
        PowerLaw(fixed=((1.0,), (2.0,))).fit(X, law(X))
    with pytest.raises(Refused, match='finite number or None'):
        PowerLaw(fixed=((float('nan'), None),))


@pytest.mark.parametrize(('which', 'column', 'row'), [('inputs', 1, 7), ('outputs', 0, 3)])
def test_non_positive_data_is_refused_naming_column_and_row(which, column, row) -> None:
    x, y = X.copy(), law(X)
    (x if which == 'inputs' else y)[row, column] = 0.0
    with pytest.raises(Refused, match=f'positive {which}: column {column}, row {row}'):
        PowerLaw().fit(x, y)


def test_the_gradient_matches_central_differences() -> None:
    fitted = PowerLaw().fit(X, law(X))
    q = np.array([[300.0, 1.0], [700.0, 0.5]])
    g = fitted.predict(q, gradient=True).gradient
    for i, h in enumerate((1e-3, 1e-6)):  # step in Hz, T
        step = np.zeros(2)
        step[i] = h
        fd = (fitted.predict(q + step).mean - fitted.predict(q - step).mean) / (2 * h)
        # absolute in output units per input unit; the slopes here are O(1e-2..1e2)
        np.testing.assert_allclose(g[:, :, i], fd, rtol=1e-6, atol=1e-8)


def test_the_artifact_round_trips(tmp_path) -> None:
    fitted = PowerLaw(fixed=((None, 2.1), (None, None))).fit(X, law(X))
    artifact.save(
        fitted,
        tmp_path / 'law',
        declaration=PowerLaw(fixed=((None, 2.1), (None, None))),
        data_identity='sha256:planted',
    )
    back = artifact.load(tmp_path / 'law')
    np.testing.assert_array_equal(back.exponents, fitted.exponents)
    np.testing.assert_array_equal(back.predict(X[:5]).mean, fitted.predict(X[:5]).mean)


def test_planted_control_a_non_power_law_cross_validates_clearly_worse() -> None:
    """The instrument can say no: ``y = 1 + f/100 + B`` is not a power law and misses by far more."""
    power = law(X)[:, :1]
    other = (1.0 + X[:, 0] / 100.0 + X[:, 1])[:, None]
    relative = [
        np.sqrt(np.mean((cross_validate(PowerLaw(), X, y, k=5, seed=1).mean / y - 1.0) ** 2)) for y in (power, other)
    ]
    assert relative[0] < 1e-9
    assert relative[1] > 1e-2
