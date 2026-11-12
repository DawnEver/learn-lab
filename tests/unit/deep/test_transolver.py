"""A Transolver learns the planted Poisson solution operator on scattered points and answers on other ones.

The planted operator: ``-lap u = f`` on the unit square, ``u = 0`` on its boundary, ``f`` a sum of the sine modes
``sin(k pi x) sin(l pi y)``, ``k, l in 1..3``, so ``u`` is exact: each mode divided by ``(k^2 + l^2) pi^2``. The
input is ``f`` at scattered points, the output ``u`` at the same points. Fit on point clouds of :data:`FIT_POINTS`
points; judge held-out sources on DIFFERENT, larger clouds of :data:`QUERY_POINTS` points. Every fit is CPU
float64 (pinned: CUDA gave different numbers than the Linux forge).

The statistic is the mean over held-out samples of the per-sample relative L2 error ``||u_hat - u|| / ||u||``
over the sample's points (unitless). Measured (seeds 0, 1, 2 of the runtime, :data:`FIT` below):
Transolver 0.0206 / 0.0196 / 0.0227 (worst 0.0227); the existing TransformerOperator (softmax, width 32, heads 4,
layers 2, the same schedule) on the same task 0.0285 / 0.0296 / 0.0261; the pointwise MLP control 0.66.
Transolver is not asserted to beat the TransformerOperator: the gap (~1.3x) is a few seeds wide, not robust.
"""

import numpy as np
import pytest
import torch

from learn_lab.deep.fit import Schedule, minimise
from learn_lab.deep.networks import MLP
from learn_lab.deep.operator import Transolver
from learn_lab.deep.runtime import Runtime
from learn_lab.errors import Refused

CPU64 = Runtime(device='cpu')
MODES = 3
FIT_POINTS, QUERY_POINTS = 128, 512
#: Mean held-out relative L2 (unitless). Measured worst over three seeds 0.0227; the pointwise control 0.66.
BAR = 0.05
FIT = Schedule(adam_steps=1500, learning_rate=3e-3)
FEW = Schedule(adam_steps=3, learning_rate=1e-2)


def _cloud(count: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.0, 1.0, (count, 2))


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


TRAIN_POINTS = np.stack([_cloud(FIT_POINTS, 100 + i) for i in range(192)])  # a different cloud per sample
F_TRAIN, U_TRAIN = _pairs(192, TRAIN_POINTS, 1)
TEST_POINTS = np.stack([_cloud(QUERY_POINTS, 900 + i) for i in range(24)])
F_TEST, U_TEST = _pairs(24, TEST_POINTS, 2)


def _transolver(**kw) -> Transolver:
    return Transolver(**{'width': 32, 'heads': 4, 'layers': 3, 'slices': 16, 'schedule': FIT, 'runtime': CPU64, **kw})


@pytest.fixture(scope='module')
def fitted():
    return _transolver().fit(F_TRAIN, TRAIN_POINTS, U_TRAIN)


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
    error = _relative(fitted.predict(F_TEST, TEST_POINTS), U_TEST)
    assert error < BAR, error
    assert _pointwise_control() > BAR
    # PLANTED CONTROL: the same network told the wrong positions (shuffled) misses: position is read.
    shuffled = TEST_POINTS[:, np.random.default_rng(3).permutation(QUERY_POINTS)]
    assert _relative(fitted.predict(F_TEST, shuffled), U_TEST) > BAR


def test_one_shared_cloud_answers_like_the_same_cloud_given_per_sample(fitted):
    shared = TEST_POINTS[0]
    one = fitted.predict(F_TEST[:3, :, :], np.broadcast_to(shared, (3, *shared.shape)))
    f, _ = _pairs(3, shared, 2)
    np.testing.assert_allclose(fitted.predict(f, shared), fitted.predict(f, np.stack([shared] * 3)), rtol=0, atol=1e-12)
    assert one.shape == (3, 1, QUERY_POINTS)


@pytest.mark.parametrize('options', [{}, {'adaptive_temperature': True}, {'reference': 4}])
def test_the_output_is_permutation_equivariant_in_the_points(options):
    f = _transolver(schedule=FEW, **options).fit(F_TRAIN[:8], TRAIN_POINTS[:8], U_TRAIN[:8])
    order = np.random.default_rng(4).permutation(QUERY_POINTS)
    shuffled = f.predict(F_TEST[:2][..., order], TEST_POINTS[:2][:, order])
    np.testing.assert_allclose(shuffled, f.predict(F_TEST[:2], TEST_POINTS[:2])[..., order], rtol=0, atol=1e-12)
    # PLANTED CONTROL: permuting the inputs but not the positions changes the answer.
    assert not np.allclose(f.predict(F_TEST[:2][..., order], TEST_POINTS[:2]), shuffled)


def _attention_records(f, points):
    """The slice weights and the token-attention scores each Physics-Attention built, read by forward hooks."""
    seen = []
    hooks = [m.register_forward_hook(lambda m, _i, _o: seen.append(m.last)) for m in f.net.attentions()]
    try:
        f.predict(F_TEST[:2, :, : points.shape[1]], points)
    finally:
        for h in hooks:
            h.remove()
    return seen


def test_slice_weights_sum_to_one_per_point_and_the_attention_is_slices_by_slices_at_any_point_count():
    f = _transolver(schedule=FEW, slices=8).fit(F_TRAIN[:8], TRAIN_POINTS[:8], U_TRAIN[:8])
    for count in (QUERY_POINTS // 4, QUERY_POINTS):
        for weights, scores in _attention_records(f, TEST_POINTS[:2, :count]):
            assert weights.shape == (2, 4, count, 8)  # (samples, heads, points, slices)
            np.testing.assert_allclose(weights.sum(-1).numpy(), 1.0, rtol=0, atol=1e-12)
            assert scores.shape == (2, 4, 8, 8)  # M x M, never N x N: linear in the points


def test_gumbel_slices_are_random_in_training_and_deterministic_in_prediction():
    f = _transolver(schedule=FEW, gumbel=True).fit(F_TRAIN[:8], TRAIN_POINTS[:8], U_TRAIN[:8])
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
    f = _transolver(schedule=FEW).fit(F_TRAIN[:8], TRAIN_POINTS[:8], U_TRAIN[:8])
    f.predict(F_TEST[:1], TEST_POINTS[:1])  # PLANTED CONTROL: in domain, it answers
    with pytest.raises(Refused, match='out_of_domain'):
        f.predict(F_TEST[:1] + 500.0, TEST_POINTS[:1])
    with pytest.raises(Refused, match='out_of_domain'):
        f.predict(F_TEST[:1], TEST_POINTS[:1] + 2.0)
    with pytest.raises(Refused, match='invalid'):
        f.predict(F_TEST[:1], TEST_POINTS[:1, :-1])
    with pytest.raises(Refused, match='heads'):
        Transolver(width=10, heads=4)
    with pytest.raises(Refused, match='slices'):
        Transolver(slices=0)
