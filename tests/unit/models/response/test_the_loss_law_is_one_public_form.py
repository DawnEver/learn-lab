"""The loss law's form ``P = p_ref f_ratio^a d_ratio^b`` is public, once: consumers scale through it, and the fit's law calls it."""

import numpy as np
import pytest

from learn_lab.models.response import GaussianProcess
from learn_lab.models.response.loss_law import DRIVER, TEXTBOOK_EXPONENTS, LossLawResidual, scale, scale_terms


def test_scale_is_bit_identical_to_the_explicit_expression() -> None:
    p, f, d = 3.7, np.array([0.3, 1.0, 2.9]), np.array([0.7, 1.1, 1.6])
    assert np.array_equal(scale(p, f_ratio=f, d_ratio=d, exponents=(1.5, 1.5)), p * f**1.5 * d**1.5)
    assert scale(2.0, f_ratio=3.0, d_ratio=0.5, exponents=(1.0, 2.0)) == 2.0 * 3.0**1.0 * 0.5**2.0


def test_scale_broadcasts_a_scalar_anchor_over_ratio_grids() -> None:
    f, d = np.linspace(0.5, 2.0, 4)[:, None], np.linspace(0.8, 1.2, 3)[None, :]
    assert scale(1.0, f_ratio=f, d_ratio=d, exponents=(2.0, 2.0)).shape == (4, 3)


def test_scale_terms_uses_each_terms_textbook_exponents_by_default() -> None:
    p_ref = {term: 1.0 + k for k, term in enumerate(TEXTBOOK_EXPONENTS)}
    got = scale_terms(p_ref, f_ratio=2.0, d_ratio=3.0)
    assert got == {t: p * 2.0 ** TEXTBOOK_EXPONENTS[t][0] * 3.0 ** TEXTBOOK_EXPONENTS[t][1] for t, p in p_ref.items()}


def test_planted_control_a_different_exponent_table_changes_the_answer() -> None:
    other = {t: (a + 0.5, b) for t, (a, b) in TEXTBOOK_EXPONENTS.items()}
    textbook = scale_terms({'classical': 1.0}, f_ratio=2.0, d_ratio=1.0)
    assert scale_terms({'classical': 1.0}, f_ratio=2.0, d_ratio=1.0, exponents=other) != textbook


def test_each_term_names_the_amplitude_it_scales_with() -> None:
    assert DRIVER == {'hysteresis': 'flux', 'classical': 'flux', 'excess': 'flux', 'magnet_eddy': 'current'}
    assert set(DRIVER) == set(TEXTBOOK_EXPONENTS)


def test_the_fitted_law_is_scale_from_its_level() -> None:
    rng = np.random.default_rng(0)
    x = np.column_stack([rng.uniform(50.0, 500.0, 12), rng.uniform(0.05, 0.2, 12)])
    y = 1e-3 * x[:, :1] ** 2 * x[:, 1:] ** 2
    fit = LossLawResidual(base=GaussianProcess()).fit(x, y)
    expected = scale(np.exp(fit.level[0]), f_ratio=x[:, 0], d_ratio=x[:, 1], exponents=tuple(fit.exponents[0]))
    assert np.array_equal(fit.law(x)[:, 0], expected)
    assert fit.law(x)[:, 0] == pytest.approx(y[:, 0], rel=1e-9, abs=1e-12)
