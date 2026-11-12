"""A simple polygon's quadrature (``simplex_rule``) is exact to total degree ``2 order - 1``.

The rule ear-clips the polygon into triangles and maps a collapsed (Duffy) Gauss-Legendre rule onto each. The
reference integral of ``x^p y^q`` is CLOSED FORM and independent of any triangulation: Green's theorem turns it into
``oint x^(p+1) y^q / (p+1) dy``, each straight side a polynomial in its parameter integrated exactly.

Planted controls -- each a way the bar could read a constant:
- one degree past the claim the same rule MISSES (the exactness bar can say no);
- a Monte Carlo draw of the same integral misses it by more than the rule's own error.
"""

import numpy as np
import pytest

from learn_lab.deep.physics import simplex_rule
from learn_lab.deep.runtime import Runtime
from learn_lab.errors import Refused

_CPU = Runtime(device='cpu')
#: A quadrilateral, no two sides parallel (m).
_SLOPED = ((0.0, 0.0), (0.3, 0.05), (0.22, 0.2), (0.04, 0.12))
#: An L: non-convex, so an ear clip must refuse the reflex corner (m).
_L = ((0.0, 0.0), (0.2, 0.0), (0.2, 0.08), (0.08, 0.08), (0.08, 0.2), (0.0, 0.2))
#: Relative bar on an exact rule: float64 rounding.
_EXACT = 1e-11


def _green(polygon, p: int, q: int) -> float:
    """``int x^p y^q dA`` (m^(p+q+2)) over a counter-clockwise polygon, by Green's theorem on each side."""
    v = np.asarray(polygon, dtype=float)
    total = 0.0
    for a, b in zip(v, np.roll(v, -1, axis=0), strict=True):
        x = np.polynomial.Polynomial([a[0], b[0] - a[0]])
        y = np.polynomial.Polynomial([a[1], b[1] - a[1]])
        anti = ((x ** (p + 1)) * (y**q) * y.deriv() / (p + 1)).integ()
        total += anti(1.0) - anti(0.0)
    return float(total)


def _inside(polygon, xy: np.ndarray) -> np.ndarray:
    """Strictly inside the polygon, by ray casting (boundary points fall either way; the rule has none)."""
    v = np.asarray(polygon, dtype=float)
    inside = np.zeros(len(xy), dtype=bool)
    for p, q in zip(v, np.roll(v, -1, axis=0), strict=True):
        crosses = (p[1] > xy[:, 1]) != (q[1] > xy[:, 1])
        with np.errstate(divide='ignore', invalid='ignore'):
            x_at = p[0] + (xy[:, 1] - p[1]) * (q[0] - p[0]) / (q[1] - p[1])
        inside ^= crosses & (xy[:, 0] < x_at)
    return inside


def _rule(polygon, order: int, cells: int = 1):
    pts, w = simplex_rule(np.asarray(polygon, dtype=float), order, _CPU, cells=cells)
    return pts.numpy(), w.numpy()


def _monomials(degree: int):
    return [(p, d - p) for d in range(degree + 1) for p in range(d + 1)]


@pytest.mark.parametrize('polygon', [_SLOPED, _L], ids=['sloped-quad', 'L'])
@pytest.mark.parametrize('order', [1, 2, 3, 4])
def test_the_rule_is_exact_to_degree_two_order_minus_one(polygon, order):
    pts, w = _rule(polygon, order)
    for p, q in _monomials(2 * order - 1):
        got = float(w @ (pts[:, 0] ** p * pts[:, 1] ** q))
        assert got == pytest.approx(_green(polygon, p, q), rel=_EXACT, abs=0.0), (p, q)


@pytest.mark.parametrize('polygon', [_SLOPED, _L], ids=['sloped-quad', 'L'])
def test_the_weights_are_positive_sum_to_the_area_and_the_points_lie_inside(polygon):
    pts, w = _rule(polygon, 3, cells=2)
    assert (w > 0).all()
    assert float(w.sum()) == pytest.approx(_green(polygon, 0, 0), rel=_EXACT, abs=0.0)  # m^2
    assert _inside(polygon, pts).all()


def test_a_clockwise_polygon_gives_the_same_positive_rule():
    pts, w = _rule(_L[::-1], 2)
    assert (w > 0).all()
    assert float(w @ pts[:, 0] ** 3) == pytest.approx(_green(_L, 3, 0), rel=_EXACT, abs=0.0)


def test_cells_refine_without_losing_exactness():
    pts, w = _rule(_L, 2, cells=3)
    assert len(w) == 4 * 9 * 2 * 3  # 4 triangles x 9 sub-triangles x order * (order + 1) points
    assert float(w @ pts[:, 0] ** 3) == pytest.approx(_green(_L, 3, 0), rel=_EXACT, abs=0.0)


def test_planted_one_degree_past_the_claim_the_rule_misses():
    pts, w = _rule(_SLOPED, 2)
    misses = [abs(float(w @ (pts[:, 0] ** p * pts[:, 1] ** q)) / _green(_SLOPED, p, q) - 1.0) for p, q in _monomials(4)]
    assert max(misses) > 1e-4


def test_planted_monte_carlo_misses_the_exact_integral():
    rng = np.random.default_rng(7)
    lo, hi = np.min(_L, axis=0), np.max(_L, axis=0)
    draw = lo + (hi - lo) * rng.random((4000, 2))
    inside = draw[_inside(_L, draw)]
    mc = _green(_L, 0, 0) * float(np.mean(inside[:, 0] ** 2 * inside[:, 1]))
    assert abs(mc / _green(_L, 2, 1) - 1.0) > 1e3 * _EXACT


@pytest.mark.parametrize(
    ('vertices', 'order', 'cells'),
    [
        (_SLOPED, 0, 1),
        (_SLOPED, 2, 0),
        (((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)), 2, 1),  # no area
        (((0.0, 0.0), (1.0, 0.0)), 2, 1),  # not a polygon
        (((0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0)), 2, 1),  # a bow tie: self-intersecting
    ],
    ids=['order-0', 'cells-0', 'collinear', 'two-vertices', 'bow-tie'],
)
def test_a_bad_request_is_refused(vertices, order, cells):
    with pytest.raises(Refused):
        simplex_rule(np.asarray(vertices, dtype=float), order, _CPU, cells=cells)
