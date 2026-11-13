"""The Clough-Tocher interpolant: C1, exact at its nodes, and asked only inside the hull of its nodes."""

from __future__ import annotations

import numpy as np
import pytest

from learn_lab.errors import Refused
from learn_lab.models.response import CloughTocher
from learn_lab.validity import BoxEnvelope, Validity

#: The bar on the worst absolute error against a smooth function sampled on a 15 x 15 grid of the unit square,
#: in the function's own units (it spans [-1, 1]). MEASURED 2026-11-13: 3.96e-3 at 500 uniform queries; the
#: planted 4 x 4 grid measures 0.108, so the bar can say no.
WORST_ABS_ERROR = 1e-2

QUERY = np.random.default_rng(1).uniform(0.0, 1.0, (500, 2))


def _smooth(p: np.ndarray) -> np.ndarray:
    return np.column_stack([np.sin(np.pi * p[:, 0]) * np.cos(np.pi * p[:, 1]), p[:, 0] * p[:, 1]])


def _grid(n: int) -> np.ndarray:
    g = np.linspace(0.0, 1.0, n)
    return np.array([(a, b) for a in g for b in g])


def _worst(n: int) -> float:
    nodes = _grid(n)
    fitted = CloughTocher().fit(nodes, _smooth(nodes))
    return float(np.max(np.abs(fitted.predict(QUERY).mean - _smooth(QUERY))[:, 0]))


def test_it_reproduces_a_smooth_function_within_the_bar() -> None:
    assert _worst(15) <= WORST_ABS_ERROR


def test_a_coarse_grid_breaks_the_bar() -> None:
    """PLANTED CONTROL: the same check on a 4 x 4 grid fails, so the bar above is a measurement."""
    assert _worst(4) > WORST_ABS_ERROR


def test_it_is_exact_at_its_nodes() -> None:
    nodes = np.random.default_rng(2).uniform(0.0, 1.0, (40, 2))
    values = _smooth(nodes)
    np.testing.assert_allclose(CloughTocher().fit(nodes, values).predict(nodes).mean, values, rtol=0, atol=1e-12)


#: A plane's worst absolute error inside the hull, in its own units, once the nodal-gradient estimate is driven to
#: ``tolerance=1e-10``. MEASURED 2026-11-13: 1.96e-12 there, 4.0e-8 at the default ``1e-6``.
PLANE_ABS_ERROR = 1e-10


def _plane_error(tolerance: float) -> float:
    nodes = np.random.default_rng(3).uniform(0.0, 1.0, (30, 2))
    slope = np.array([[2.0], [-3.0]])
    fitted = CloughTocher(tolerance=tolerance, max_iterations=10_000).fit(nodes, nodes @ slope + 1.0)
    inside = QUERY[fitted.envelope.classify(QUERY) == Validity.IN_DOMAIN]
    assert len(inside) > 100
    return float(np.max(np.abs(fitted.predict(inside).mean - (inside @ slope + 1.0))))


def test_it_reproduces_a_plane_everywhere_in_its_hull() -> None:
    assert _plane_error(1e-10) <= PLANE_ABS_ERROR


def test_the_gradient_tolerance_is_what_buys_the_plane() -> None:
    """PLANTED CONTROL: at the default tolerance the same plane misses the bar, so the setting is live."""
    assert _plane_error(1e-6) > PLANE_ABS_ERROR


def test_its_envelope_is_the_hull_and_a_point_just_outside_is_refused() -> None:
    triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.2, 0.3]])
    fitted = CloughTocher().fit(triangle, triangle[:, :1])
    assert np.isfinite(fitted.predict(np.array([[0.5 - 1e-6, 0.5 - 1e-6]])).mean).all()
    with pytest.raises(Refused, match='out_of_domain'):
        fitted.predict(np.array([[0.5 + 1e-6, 0.5 + 1e-6]]))


def test_a_declared_domain_is_refused_since_it_would_be_extrapolated() -> None:
    nodes = _grid(4)
    with pytest.raises(Refused, match='hull of its nodes'):
        CloughTocher().fit(nodes, _smooth(nodes), domain=BoxEnvelope(np.full(2, -1.0), np.full(2, 2.0)))


@pytest.mark.parametrize(
    ('nodes', 'match'),
    [
        (np.random.default_rng(4).uniform(0.0, 1.0, (6, 3)), 'two inputs'),
        (np.column_stack([np.linspace(0, 1, 5), np.linspace(0, 1, 5)]), 'collinear'),
    ],
)
def test_it_refuses_what_it_cannot_triangulate(nodes, match) -> None:
    with pytest.raises(Refused, match=match):
        CloughTocher().fit(nodes, nodes[:, :1])


def test_it_gives_no_gradient_or_variance() -> None:
    nodes = _grid(4)
    fitted = CloughTocher().fit(nodes, _smooth(nodes))
    assert fitted.gives == frozenset()
    with pytest.raises(Refused, match='gradient'):
        fitted.predict(nodes, gradient=True)
