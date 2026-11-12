"""The graph neural operator learns a Green's function on one point set and applies it on another.

Planted operator: ``-u'' = f`` on ``[0, 1]`` with ``u(0) = u(1) = 0``, ``f = sum_k a_k sin(k pi x)`` -- so
``u = sum_k a_k sin(k pi x) / (k pi)^2`` exactly, the integral of the Green's function against ``f``. The fit
sees 48 scattered points; the held-out functions are evaluated on 97 OTHER scattered points (a finer set).
The bar is an RMS error in u's own unit. PLANTED CONTROL: a pointwise network of ``(f(x), x)`` -- the same
data, no message passing -- cannot reach it, because ``u(x)`` depends on ``f`` everywhere.
"""

import numpy as np
import pytest
import torch

from learn_lab.deep.fit import Schedule, minimise
from learn_lab.deep.networks import MLP
from learn_lab.deep.operator import GraphOperator
from learn_lab.deep.runtime import Runtime
from learn_lab.errors import Refused
from learn_lab.representation import intervals, mesh_graph

MODES = np.arange(1, 5)
#: The held-out RMS error bar, in u's unit (rms(u) over the held-out set is ~0.05).
BAR = 2.5e-3
CPU64 = Runtime(device='cpu', dtype='float64')


def _points(n: int, seed: int) -> np.ndarray:
    inner = np.sort(np.random.default_rng(seed).uniform(0, 1, n - 2))
    return np.concatenate([[0.0], inner, [1.0]])


def _pairs(points: np.ndarray, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    a = np.random.default_rng(seed).standard_normal((n, len(MODES)))
    basis = np.sin(np.pi * MODES[:, None] * points[None, :])
    f = a @ basis
    u = (a / (np.pi * MODES) ** 2) @ basis
    return f[:, None], u[:, None]


def _graph(points: np.ndarray):
    return mesh_graph(points[:, None], intervals(points), radius=1.5)


TRAIN, TEST = _points(48, 1), _points(97, 2)
F, U = _pairs(TRAIN, 128, 3)
F_TEST, U_TEST = _pairs(TEST, 32, 4)


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


@pytest.fixture(scope='module')
def fitted():
    model = GraphOperator(
        width=16,
        layers=2,
        kernel_width=32,
        schedule=Schedule(adam_steps=1500, learning_rate=3e-3),
        runtime=CPU64,
        margin=0.5,
    )
    return model.fit(F, _graph(TRAIN), U)


def test_the_test_points_are_a_different_finer_set() -> None:
    assert len(TEST) > len(TRAIN)
    assert len(np.intersect1d(TRAIN[1:-1], TEST[1:-1])) == 0


def test_the_operator_learned_on_one_point_set_holds_on_a_finer_one(fitted) -> None:
    error = _rms(fitted.predict(F_TEST, _graph(TEST)), U_TEST)
    assert error < BAR, error


def test_a_pointwise_network_without_message_passing_misses_the_bar() -> None:
    """PLANTED CONTROL: same inputs (f and x at each point), no neighbourhood."""
    rt = CPU64
    rt.seeded()
    rows = np.column_stack([F[:, 0].ravel(), np.tile(TRAIN, len(F))])
    net = MLP([2, 64, 64, 1], activation='gelu').to(rt.torch_device, rt.torch_dtype)
    xt, yt = rt.tensor(rows), rt.tensor(U[:, 0].ravel()[:, None])

    def loss() -> torch.Tensor:
        return ((net(xt) - yt) ** 2).mean()

    minimise(net.parameters(), loss, Schedule(adam_steps=1500, learning_rate=3e-3))
    test_rows = rt.tensor(np.column_stack([F_TEST[:, 0].ravel(), np.tile(TEST, len(F_TEST))]))
    with torch.no_grad():
        predicted = net(test_rows).numpy().reshape(U_TEST.shape)
    error = _rms(predicted, U_TEST)
    assert error > 3 * BAR, error


def test_a_point_outside_the_training_box_is_refused(fitted) -> None:
    far = np.array([0.0, 0.5, 3.0])
    with pytest.raises(Refused):
        fitted.predict(np.zeros((1, 1, 3)), _graph(far))


def test_a_query_whose_shape_misses_its_graph_is_refused(fitted) -> None:
    with pytest.raises(Refused, match='invalid'):
        fitted.predict(F_TEST[:, :, :10], _graph(TEST))
