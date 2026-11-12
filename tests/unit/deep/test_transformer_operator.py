"""A transformer operator learns the planted causal heat-kernel response on irregular points.

The planted operator: ``y(t) = int_0^t exp(-(t - s) / TAU) u(s) ds`` (unit: the input's unit times seconds),
discretised exactly as the trapezoid sum over the irregular sample times, so the target is the operator the
data defines. The answer is known; the bar is on held-out input functions. Every fit is CPU float64 (pinned:
CUDA, this box's default, gave different numbers than the Linux forge).
"""

import numpy as np
import pytest

from learn_lab.deep.fit import Schedule
from learn_lab.deep.operator import TransformerOperator
from learn_lab.deep.runtime import Runtime
from learn_lab.errors import Refused

CPU64 = Runtime(device='cpu')
TAU = 0.2  # s
POINTS = 24
#: Held-out RMS error, in the output's unit (input unit x s); the outputs' own RMS is ~0.15; measured 0.0077 (softmax), 0.0075 (linear).
BAR = 0.01
FEW = Schedule(adam_steps=3, learning_rate=1e-2)


def _times(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([[0.0, 1.0], rng.uniform(0.0, 1.0, POINTS - 2)]))


def _inputs(count: int, t: np.ndarray, seed: int) -> np.ndarray:
    a = np.random.default_rng(seed).standard_normal((count, 3))
    return (a @ np.stack([np.ones_like(t), np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)]))[:, None, :]


def _response(u: np.ndarray, t: np.ndarray) -> np.ndarray:
    """The trapezoid sum of the causal kernel over the irregular times: (n, 1, p)."""
    y = np.zeros_like(u)
    for j in range(1, t.size):
        f = np.exp(-(t[j] - t[: j + 1]) / TAU) * u[:, 0, : j + 1]
        y[:, 0, j] = np.trapezoid(f, t[: j + 1], axis=-1)
    return y


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


T = _times(0)
COORDS = T[:, None]
X_TRAIN, X_TEST = _inputs(256, T, 1), 0.8 * _inputs(32, T, 2)
Y_TRAIN, Y_TEST = _response(X_TRAIN, T), _response(X_TEST, T)


@pytest.fixture(scope='module', params=['softmax', 'linear'])
def fitted(request):
    model = TransformerOperator(
        width=16,
        heads=2,
        layers=2,
        kernel=request.param,
        causal=True,
        schedule=Schedule(adam_steps=600, learning_rate=3e-3),
        runtime=CPU64,
    )
    return model.fit(X_TRAIN, COORDS, Y_TRAIN)


def test_a_causal_transformer_learns_the_heat_kernel_response_and_controls_miss(fitted):
    error = _rms(fitted.predict(X_TEST, COORDS), Y_TEST)
    assert error < BAR, error
    # PLANTED CONTROL: the training mean, the best answer that ignores the input, misses the bar.
    assert _rms(np.broadcast_to(Y_TRAIN.mean(0), Y_TEST.shape), Y_TEST) > BAR
    # PLANTED CONTROL: the same network told the wrong coordinates (shuffled) misses: position is read.
    shuffled = COORDS[np.random.default_rng(3).permutation(POINTS)]
    assert _rms(fitted.predict(X_TEST, shuffled), Y_TEST) > BAR


@pytest.mark.parametrize('kernel', ['softmax', 'linear'])
def test_a_causal_output_never_reads_the_future(kernel):
    model = TransformerOperator(width=8, heads=2, layers=2, kernel=kernel, causal=True, schedule=FEW, runtime=CPU64)
    f = model.fit(X_TRAIN[:16], COORDS, Y_TRAIN[:16])
    k = POINTS // 2
    perturbed = X_TEST[:4].copy()
    perturbed[..., k + 1 :] *= -0.5
    before, after = f.predict(X_TEST[:4], COORDS), f.predict(perturbed, COORDS)
    assert np.array_equal(before[..., : k + 1], after[..., : k + 1])
    # PLANTED CONTROL: the perturbation is seen downstream, and an acausal operator sees it everywhere.
    assert not np.array_equal(before[..., k + 1 :], after[..., k + 1 :])
    acausal = TransformerOperator(width=8, heads=2, layers=2, kernel=kernel, schedule=FEW, runtime=CPU64)
    g = acausal.fit(X_TRAIN[:16], COORDS, Y_TRAIN[:16])
    assert not np.array_equal(g.predict(X_TEST[:4], COORDS)[..., :k], g.predict(perturbed, COORDS)[..., :k])


def test_the_causal_order_follows_the_first_coordinate_not_the_index():
    model = TransformerOperator(width=8, heads=2, layers=1, causal=True, schedule=FEW, runtime=CPU64)
    f = model.fit(X_TRAIN[:16], COORDS, Y_TRAIN[:16])
    order = np.random.default_rng(4).permutation(POINTS)
    shuffled = f.predict(X_TEST[:4, :, order], COORDS[order])
    np.testing.assert_allclose(shuffled, f.predict(X_TEST[:4], COORDS)[..., order], rtol=0, atol=1e-12)


def test_the_linear_kernel_equals_itself_chunked_and_a_query_takes_other_irregular_points():
    def fit(chunk):
        model = TransformerOperator(
            width=8, heads=2, layers=1, kernel='linear', chunk=chunk, schedule=FEW, runtime=CPU64
        )
        return model.fit(X_TRAIN[:16], COORDS, Y_TRAIN[:16])

    one, chunked = fit(None), fit(5)
    np.testing.assert_allclose(chunked.predict(X_TEST, COORDS), one.predict(X_TEST, COORDS), rtol=0, atol=1e-12)
    other = _times(9)[:17]  # fewer, different points inside the training box
    assert one.predict(_inputs(2, other, 5), other[:, None]).shape == (2, 1, 17)


def test_a_stranger_is_refused():
    f = TransformerOperator(width=8, heads=2, layers=1, schedule=FEW, runtime=CPU64).fit(
        X_TRAIN[:16], COORDS, Y_TRAIN[:16]
    )
    f.predict(X_TEST[:1], COORDS)  # PLANTED CONTROL: in domain, it answers
    with pytest.raises(Refused, match='out_of_domain'):
        f.predict(X_TEST[:1] + 50.0, COORDS)
    with pytest.raises(Refused, match='out_of_domain'):
        f.predict(X_TEST[:1], COORDS + 2.0)
    with pytest.raises(Refused, match='invalid'):
        f.predict(X_TEST[:1], COORDS[:-1])
    with pytest.raises(Refused, match='heads'):
        TransformerOperator(width=10, heads=4)
    with pytest.raises(Refused, match='kernel'):
        TransformerOperator(kernel='cosine')
