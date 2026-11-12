"""A Transolver learns the planted Poisson solution operator on scattered points and answers on other ones.

The planted operator: ``-lap u = f`` on the unit square, ``u = 0`` on its boundary, ``f`` a sum of the sine modes
``sin(k pi x) sin(l pi y)``, ``k, l in 1..3``, so ``u`` is exact: each mode divided by ``(k^2 + l^2) pi^2``. The
input is ``f`` at scattered points, the output ``u`` at the same points, each point weighted by its Voronoi cell
area (the ``measure``). Fit on 192 jittered-grid clouds of :data:`FIT_POINTS` points (one jittered point per cell
of an 11 x 11 grid, plus the corners; a different cloud per sample); judge 24 held-out sources on DIFFERENT,
larger, uniformly random clouds of :data:`QUERY_POINTS` points. Every fit is CPU float64 (pinned: CUDA gave
different numbers than the Linux forge).

The statistic is the mean over held-out samples of the per-sample relative L2 error ``||u_hat - u|| / ||u||``
over the sample's points (unitless). Measured 2026-11-12, this configuration (:data:`FIT`): runtime seed 0 0.0361,
seed 1 0.0324 (train 0.031 both); the pointwise-MLP control 0.601. What closed the gap, each measured:

- A slice token is a MEAN over its points: on a random cloud that is Monte-Carlo quadrature. Projecting ``f`` on
  the modes that way leaves 0.17 relative error at 512 random points (0.33 at 128); weighting by Voronoi area
  leaves 0.015 (0.11 at 128). Unweighted, the Transolver measured 0.16-0.20 held-out (seeds 0, 1; 1500-6000
  steps, widths 32-64, slices 16-64, ``reference=6``), right at that floor. With the area ``measure``: 0.074 /
  0.075 (1500 steps, cosine), the unweighted twin 0.183 / 0.161.
- The remaining 0.065-0.075 did not move with more random training clouds (576: 0.068 / 0.069), larger ones
  (256 points: 0.063 / 0.073), ``reference=8`` (0.107), Ada-Temp + Gumbel (0.071 / 0.067), a larger net (width 64,
  8 heads, 32 slices, 4 layers: 0.073) or 6000 steps (train 0.034, held-out 0.073), and was flat in the QUERY size
  (0.074 at 512, 0.075 at 2048): a learned-operator error, not resolution. Training inputs carried 128-point
  random-cloud quadrature noise (errors in the variables); jittered training clouds remove it: 1500 steps 0.062 /
  0.073 (under-trained, train equal), 4500 steps cosine 0.036 / 0.032.
- The existing TransformerOperator (softmax, width 32, heads 4, layers 2, 1500 steps; one shared random 128-point
  training cloud, as it takes one point set per call; judged per test cloud) measured 0.329 / 0.325 (seeds 0, 1).
  Not asserted: the setups differ.
"""

import numpy as np
import pytest
import torch
from scipy.spatial import cKDTree

from learn_lab.deep.fit import Schedule, minimise
from learn_lab.deep.networks import MLP
from learn_lab.deep.operator import Transolver
from learn_lab.deep.runtime import Runtime
from learn_lab.errors import Refused

CPU64 = Runtime(device='cpu')
MODES = 3
GRID = 11
FIT_POINTS, QUERY_POINTS = GRID * GRID + 4, 512
#: Mean held-out relative L2 (unitless). Measured 0.0361 / 0.0324 (seeds 0, 1); the pointwise control 0.601.
BAR = 0.05
FIT = Schedule(adam_steps=4500, learning_rate=3e-3, decay='cosine')
FEW = Schedule(adam_steps=3, learning_rate=1e-2)
#: The few-step fits see 8 sources; a margin (fraction of the value box) lets the held-out ones in.
WIDE = 1.0


def _cloud(count: int, seed: int) -> np.ndarray:
    """``count`` scattered points of the unit square, its four corners first (every cloud spans the same box)."""
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    return np.concatenate([corners, np.random.default_rng(seed).uniform(0.0, 1.0, (count - 4, 2))])


def _jittered(seed: int) -> np.ndarray:
    """One uniformly jittered point in each cell of a ``GRID x GRID`` grid of the unit square, the corners first."""
    g = np.arange(GRID)
    cells = np.stack(np.meshgrid(g, g), axis=-1).reshape(-1, 2)
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    return np.concatenate([corners, (cells + np.random.default_rng(seed).uniform(0.0, 1.0, cells.shape)) / GRID])


def _basis(points: np.ndarray) -> np.ndarray:
    """``(MODES^2, p)``: every sine mode at the points."""
    k = np.arange(1, MODES + 1)
    sx, sy = np.sin(np.pi * k[:, None] * points[:, 0]), np.sin(np.pi * k[:, None] * points[:, 1])
    return (sx[:, None, :] * sy[None, :, :]).reshape(MODES * MODES, -1)


def _pairs(count: int, points: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Sources ``f (n, 1, p)`` and exact solutions ``u (n, 1, p)``; points per sample ``(n, p, 2)`` or shared."""
    a = np.random.default_rng(seed).standard_normal((count, MODES * MODES))
    k = np.arange(1, MODES + 1)
    eigen = (np.pi**2 * (k[:, None] ** 2 + k[None, :] ** 2)).reshape(-1)
    if points.ndim == 2:
        b = _basis(points)
        return (a @ b)[:, None], ((a / eigen) @ b)[:, None]
    f = np.stack([ai @ _basis(p) for ai, p in zip(a, points, strict=True)])[:, None]
    u = np.stack([(ai / eigen) @ _basis(p) for ai, p in zip(a, points, strict=True)])[:, None]
    return f, u


def _relative(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.mean(np.linalg.norm((a - b).reshape(len(a), -1), axis=1) / np.linalg.norm(b.reshape(len(b), -1), axis=1))
    )


def _areas(points: np.ndarray) -> np.ndarray:
    """Each point's Voronoi cell area in the unit square (m^2 per unit square), counted on a 250 x 250 grid."""
    g = (np.arange(250) + 0.5) / 250
    grid = np.stack(np.meshgrid(g, g), axis=-1).reshape(-1, 2)
    nearest = cKDTree(points).query(grid)[1]
    return np.bincount(nearest, minlength=len(points)) / len(grid)


TRAIN_POINTS = np.stack([_jittered(100 + i) for i in range(192)])  # a different cloud per sample
F_TRAIN, U_TRAIN = _pairs(192, TRAIN_POINTS, 1)
TEST_POINTS = np.stack([_cloud(QUERY_POINTS, 900 + i) for i in range(24)])
F_TEST, U_TEST = _pairs(24, TEST_POINTS, 2)
A_TRAIN = np.stack([_areas(p) for p in TRAIN_POINTS])
A_TEST = np.stack([_areas(p) for p in TEST_POINTS])


def _transolver(**kw) -> Transolver:
    return Transolver(**{'width': 32, 'heads': 4, 'layers': 3, 'slices': 16, 'schedule': FIT, 'runtime': CPU64, **kw})


@pytest.fixture(scope='module')
def fitted():
    return _transolver(chunk=None).fit(F_TRAIN, TRAIN_POINTS, U_TRAIN, measure=A_TRAIN)


def _pointwise_control() -> float:
    """PLANTED CONTROL: an MLP on ``(x, y, f(x, y))`` alone -- no mixing across points -- trained as long."""
    torch.manual_seed(0)
    net = MLP([3, 64, 64, 1], activation='gelu').double()
    rows = np.concatenate([TRAIN_POINTS, F_TRAIN[:, 0, :, None]], axis=-1).reshape(-1, 3)
    scale = np.abs(U_TRAIN).max()
    xt, yt = torch.as_tensor(rows), torch.as_tensor(U_TRAIN.reshape(-1, 1) / scale)

    def loss():
        return ((net(xt) - yt) ** 2).mean()

    minimise(net.parameters(), loss, FIT)
    query = np.concatenate([TEST_POINTS, F_TEST[:, 0, :, None]], axis=-1).reshape(-1, 3)
    with torch.no_grad():
        guess = net(torch.as_tensor(query)).numpy().reshape(U_TEST.shape) * scale
    return _relative(guess, U_TEST)


def test_a_transolver_learns_the_poisson_operator_on_other_larger_clouds_and_a_pointwise_net_cannot(fitted):
    error = _relative(fitted.predict(F_TEST, TEST_POINTS, measure=A_TEST), U_TEST)
    assert error < BAR, error
    assert _pointwise_control() > BAR
    # PLANTED CONTROL: the same network told the wrong positions (shuffled) misses: position is read.
    shuffled = TEST_POINTS[:, np.random.default_rng(3).permutation(QUERY_POINTS)]
    assert _relative(fitted.predict(F_TEST, shuffled, measure=A_TEST), U_TEST) > BAR


def test_one_shared_cloud_answers_like_the_same_cloud_given_per_sample(fitted):
    shared, area = TEST_POINTS[0], A_TEST[0]
    f, _ = _pairs(3, shared, 2)
    one = fitted.predict(f, shared, measure=area)
    np.testing.assert_allclose(
        one, fitted.predict(f, np.stack([shared] * 3), measure=np.stack([area] * 3)), rtol=0, atol=1e-12
    )
    assert one.shape == (3, 1, QUERY_POINTS)


@pytest.mark.parametrize('options', [{}, {'adaptive_temperature': True}, {'reference': 4}])
def test_the_output_is_permutation_equivariant_in_the_points(options):
    f = _transolver(schedule=FEW, margin=WIDE, **options).fit(F_TRAIN[:8], TRAIN_POINTS[:8], U_TRAIN[:8])
    order = np.random.default_rng(4).permutation(QUERY_POINTS)
    shuffled = f.predict(F_TEST[:2][..., order], TEST_POINTS[:2][:, order])
    np.testing.assert_allclose(shuffled, f.predict(F_TEST[:2], TEST_POINTS[:2])[..., order], rtol=0, atol=1e-12)
    # PLANTED CONTROL: permuting the inputs but not the positions changes the answer.
    assert not np.allclose(f.predict(F_TEST[:2][..., order], TEST_POINTS[:2]), shuffled)


def _attention_records(f, points=None, query=None):
    """The slice weights, token-attention scores and slice tokens each Physics-Attention built, read by hooks."""
    seen = []
    hooks = [m.register_forward_hook(lambda m, _i, _o: seen.append(m.last)) for m in f.net.attentions()]
    try:
        f.predict(query[0], query[1], measure=query[2]) if query else f.predict(
            F_TEST[:2, :, : points.shape[1]], points
        )
    finally:
        for h in hooks:
            h.remove()
    return seen


def test_slice_weights_sum_to_one_per_point_and_the_attention_is_slices_by_slices_at_any_point_count():
    f = _transolver(schedule=FEW, margin=WIDE, slices=8).fit(F_TRAIN[:8], TRAIN_POINTS[:8], U_TRAIN[:8])
    for count in (QUERY_POINTS // 4, QUERY_POINTS):
        for weights, scores, _ in _attention_records(f, TEST_POINTS[:2, :count]):
            assert weights.shape == (2, 4, count, 8)  # (samples, heads, points, slices)
            np.testing.assert_allclose(weights.sum(-1).numpy(), 1.0, rtol=0, atol=1e-12)
            assert scores.shape == (2, 4, 8, 8)  # M x M, never N x N: linear in the points


def test_a_measured_slice_token_is_a_quadrature_the_same_at_n_and_4n_points():
    """The same source sampled on random clouds of 512 and 2048 points: the first layer's slice tokens agree to the
    quadrature error, and area weights shrink it. PLANTED CONTROL: the unweighted mean (a uniform measure) is
    Monte-Carlo quadrature and disagrees more; neither is a sum (a sum would grow 4x with the points).
    """
    f = _transolver(schedule=FEW, margin=WIDE).fit(F_TRAIN[:8], TRAIN_POINTS[:8], U_TRAIN[:8], measure=A_TRAIN[:8])
    clouds = [_cloud(n, 77) for n in (QUERY_POINTS, 4 * QUERY_POINTS)]
    sources = [_pairs(1, c, 5)[0] for c in clouds]

    def tokens(weighting):
        return [
            _attention_records(f, query=(s, c[None], weighting(c)[None]))[0][2]
            for s, c in zip(sources, clouds, strict=True)
        ]

    def gap(pair):
        return float((pair[0] - pair[1]).norm() / pair[1].norm())

    weighted, uniform = gap(tokens(_areas)), gap(tokens(lambda c: np.ones(len(c))))
    assert weighted < 0.5 * uniform, (weighted, uniform)
    assert uniform < 0.5  # a mean, not a sum


def test_gumbel_slices_are_random_in_training_and_deterministic_in_prediction():
    f = _transolver(schedule=FEW, margin=WIDE, gumbel=True).fit(F_TRAIN[:8], TRAIN_POINTS[:8], U_TRAIN[:8])
    q = (F_TEST[:2], TEST_POINTS[:2])
    assert np.array_equal(f.predict(*q), f.predict(*q))
    f.net.train()
    try:
        x, c = torch.as_tensor(f.inputs.forward(q[0])), torch.as_tensor(q[1])
        with torch.no_grad():
            assert not torch.equal(f.net(x, c), f.net(x, c))  # PLANTED CONTROL: training draws the noise
    finally:
        f.net.eval()


def test_a_stranger_is_refused():
    f = _transolver(schedule=FEW, margin=WIDE).fit(F_TRAIN[:8], TRAIN_POINTS[:8], U_TRAIN[:8])
    f.predict(F_TEST[:1], TEST_POINTS[:1])  # PLANTED CONTROL: in domain, it answers
    with pytest.raises(Refused, match='out_of_domain'):
        f.predict(F_TEST[:1] + 500.0, TEST_POINTS[:1])
    with pytest.raises(Refused, match='out_of_domain'):
        f.predict(F_TEST[:1], TEST_POINTS[:1] + 2.0)
    with pytest.raises(Refused, match='invalid'):
        f.predict(F_TEST[:1], TEST_POINTS[:1, :-1])
    with pytest.raises(Refused, match='point measure'):
        f.predict(F_TEST[:1], TEST_POINTS[:1], measure=A_TEST[:1])  # fitted without one
    g = _transolver(schedule=FEW, margin=WIDE).fit(F_TRAIN[:8], TRAIN_POINTS[:8], U_TRAIN[:8], measure=A_TRAIN[:8])
    g.predict(F_TEST[:1], TEST_POINTS[:1], measure=A_TEST[:1])  # PLANTED CONTROL: with its measure, it answers
    with pytest.raises(Refused, match='point measure'):
        g.predict(F_TEST[:1], TEST_POINTS[:1])
    with pytest.raises(Refused, match='point measure'):
        g.predict(F_TEST[:1], TEST_POINTS[:1], measure=-A_TEST[:1])
    with pytest.raises(Refused, match='heads'):
        Transolver(width=10, heads=4)
    with pytest.raises(Refused, match='slices'):
        Transolver(slices=0)
