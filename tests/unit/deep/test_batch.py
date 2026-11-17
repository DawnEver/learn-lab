"""``Schedule.batch``: Adam steps on drawn mini-batches, L-BFGS on the full objective, resumable, every family.

The loss closure is ``loss(index | None)``; these tests count the rows each call is handed. PLANTED CONTROLS:
``batch = N`` matches the full-batch fit while ``batch = N / 2`` does not (so the comparison can say no), and
each family's ``loss(permutation)`` equals its ``loss(None)`` -- a closure that ignored the index, or sliced the
wrong axis, would fail one or the other.
"""

import importlib

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from learn_lab.deep.fit import Checkpoint, Schedule, minimise, take
from learn_lab.deep.networks import MLP
from learn_lab.deep.physics import PhysicsInformedField
from learn_lab.deep.preconditioner import LearnedPreconditioner
from learn_lab.deep.registry import DEEP
from learn_lab.deep.runtime import Runtime
from learn_lab.errors import Refused
from learn_lab.point_graph import intervals, mesh_graph

from ._poisson import Poisson

RT = Runtime(device='cpu', dtype='float64')
N = 16
RNG = np.random.default_rng(0)


class Interrupted(Exception):
    pass


def _problem(rows: list[int | None], *, stop_after: int | None = None):
    """A seeded net and a closure that records how many rows each call sees (``None``: all N)."""
    RT.seeded()
    net = MLP([2, 8, 1]).to(RT.torch_device, RT.torch_dtype)
    x = torch.linspace(-1, 1, 2 * N, dtype=RT.torch_dtype).reshape(N, 2)
    y = torch.sin(3 * x[:, :1]) * x[:, 1:]

    def loss(index):
        if stop_after is not None and len(rows) >= stop_after:
            raise Interrupted
        xb, yb = take(index, x, y)
        rows.append(len(xb))
        return ((net(xb) - yb) ** 2).mean()

    return net, loss, x


def _fit(schedule, rows=None, **kwargs):
    rows = [] if rows is None else rows
    net, loss, x = _problem(rows, stop_after=kwargs.pop('stop_after', None))
    history = minimise(net.parameters(), loss, schedule, samples=N, seed=7, data=[x], **kwargs)
    return net, history, rows


def test_batch_none_is_the_plain_full_batch_adam_loop_bit_for_bit():
    net, history, rows = _fit(Schedule(adam_steps=20, learning_rate=1e-2))
    reference, loss, _ = _problem([])
    adam = torch.optim.Adam(reference.parameters(), lr=1e-2)
    expected = []
    for _ in range(20):
        adam.zero_grad(set_to_none=True)
        value = loss(None)
        value.backward()
        adam.step()
        expected.append(float(value.detach()))
    assert history == expected
    for a, b in zip(net.state_dict().values(), reference.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    assert rows == [N] * 20


def test_a_batched_adam_step_sees_batch_distinct_rows_drawn_afresh():
    seen = []
    RT.seeded()
    net = MLP([2, 4, 1]).to(RT.torch_device, RT.torch_dtype)
    x = torch.randn(N, 2, dtype=RT.torch_dtype)

    def loss(index):
        seen.append(index)
        return (net(take(index, x)[0]) ** 2).mean()

    minimise(net.parameters(), loss, Schedule(adam_steps=10, learning_rate=1e-2, batch=4), samples=N, seed=3)
    assert [len(i) for i in seen] == [4] * 10  # O(batch) rows per step, never N
    assert all(len(set(i.tolist())) == 4 and int(i.max()) < N for i in seen)
    assert len({tuple(i.tolist()) for i in seen}) > 1


def test_a_batch_at_least_n_is_every_sample_and_matches_the_full_batch_fit():
    _, full, _ = _fit(Schedule(adam_steps=15, learning_rate=1e-2))
    _, whole, rows = _fit(Schedule(adam_steps=15, learning_rate=1e-2, batch=N))
    assert rows == [N] * 15
    assert whole == pytest.approx(full, rel=1e-10, abs=1e-12)
    _, half, _ = _fit(Schedule(adam_steps=15, learning_rate=1e-2, batch=N // 2))  # PLANTED CONTROL
    assert half != pytest.approx(full, rel=1e-6, abs=1e-9)
    _, more, _ = _fit(Schedule(adam_steps=15, learning_rate=1e-2, batch=10 * N))
    assert more == pytest.approx(full, rel=1e-10, abs=1e-12)


def test_the_polish_sees_every_sample():
    _, history, rows = _fit(Schedule(adam_steps=5, learning_rate=1e-2, lbfgs_steps=3, batch=4))
    assert rows[:5] == [4] * 5
    assert rows[5:] and set(rows[5:]) == {N}
    assert len(history) == 8


def test_a_batched_fit_resumed_mid_adam_draws_the_same_batches(tmp_path):
    schedule = Schedule(adam_steps=12, learning_rate=1e-2, batch=5)
    whole, whole_history, _ = _fit(schedule)
    with pytest.raises(Interrupted):
        _fit(schedule, checkpoint=Checkpoint(tmp_path, every=5), stop_after=8)
    resumed, resumed_history, _ = _fit(schedule, checkpoint=Checkpoint(tmp_path, every=5))
    assert resumed_history == whole_history
    for a, b in zip(whole.state_dict().values(), resumed.state_dict().values(), strict=True):
        assert torch.equal(a, b)


def test_the_batch_is_part_of_the_checkpoint_identity(tmp_path):
    _fit(Schedule(adam_steps=4, learning_rate=1e-2, batch=4), checkpoint=Checkpoint(tmp_path, every=2))
    with pytest.raises(Refused, match='another config'):
        _fit(Schedule(adam_steps=4, learning_rate=1e-2, batch=8), checkpoint=Checkpoint(tmp_path, every=2))


@pytest.mark.parametrize('batch', [0, -1, 2.0, True])
def test_a_batch_is_a_positive_count(batch):
    with pytest.raises(Refused, match='batch'):
        Schedule(batch=batch)


def test_a_loss_without_a_sample_axis_refuses_a_batch():
    with pytest.raises(Refused, match='no sample axis'):
        minimise([torch.zeros(1, requires_grad=True)], lambda _i: torch.zeros(()), Schedule(batch=2))


# -- every family honours the batch or refuses it by name ------------------------------------------------------

_MODULES = ('constitutive', 'diffusion', 'dynamics', 'field', 'graph', 'hysteresis', 'operator', 'pdenet')
_MODULES += ('preconditioner', 'surrogate')


@pytest.fixture
def calls(monkeypatch):
    """Every family's ``minimise``, recording each fit's sample count and the rows of every closure call."""
    record = []

    def spy(parameters, loss, schedule, *, samples=None, **kwargs):
        sizes, parameters = [], list(parameters)
        if samples is not None:
            with torch.no_grad():  # the closure is a mean along its sample axis: order-free
                RT.seeded()
                permuted = loss(torch.randperm(samples, device=parameters[0].device))
                RT.seeded()
                full = loss(None)
        else:
            permuted = full = None

        def counted(index):
            sizes.append(None if index is None else len(index))
            return loss(index)

        history = minimise(parameters, counted, schedule, samples=samples, **kwargs)
        record.append({'samples': samples, 'sizes': sizes, 'schedule': schedule, 'pair': (permuted, full)})
        return history

    for name in _MODULES:
        monkeypatch.setattr(importlib.import_module(f'learn_lab.deep.{name}'), 'minimise', spy)
    return record


def _fields(n):
    x = RNG.standard_normal((n, 1, 8, 8))
    return x, 0.5 * x + 0.1


_SEQ = RNG.standard_normal((6, 1, 5))
_COORDS = np.sort(RNG.uniform(0, 1, 5))[:, None]
_LINE = np.linspace(0, 1, 5)
_T = np.linspace(0, 1, 5)
_PATHS = np.exp(-_T)[None, :, None] * RNG.uniform(0.5, 1.5, (6, 1, 1))
_NODES = RNG.standard_normal((6, 3, 2))
_HIST = np.sin(np.linspace(0, 6, 20))[None, :] * RNG.uniform(0.5, 1.0, (4, 1))
_SNAP = np.sin(np.linspace(0, 2 * np.pi, 16, endpoint=False))[None, :] * RNG.uniform(0.5, 1.5, (6, 1))
_X2 = RNG.uniform(-1, 1, (6, 2))

#: name -> (x, y, settings): the small CPU data of ``test_tuning`` (sample counts >= 4, so batch 2 is a subset).
FAMILIES = {
    'deep_ensemble': (_X2, _X2[:, :1] - _X2[:, 1:], {'width': 8, 'depth': 1, 'members': 1}),
    'unet': (*_fields(6), {'width': 4, 'levels': 1, 'margin': 10.0}),
    'deeponet': (_SEQ[:, 0], _SEQ[:, 0, :4], {'points': _LINE[:4, None], 'width': 8, 'depth': 1, 'basis': 4}),
    'fno': (*_fields(6), {'modes': 2, 'width': 4, 'layers': 1, 'margin': 10.0}),
    'transformer_operator': (_SEQ, -_SEQ, {'coords': _COORDS, 'heads': 2, 'head_width': 2, 'layers': 1}),
    'transolver': (_SEQ, -_SEQ, {'coords': _COORDS, 'width': 4, 'heads': 2, 'layers': 1, 'slices': 2}),
    'graph_operator': (
        _SEQ,
        -_SEQ,
        {'graph': mesh_graph(_LINE[:, None], intervals(_LINE), radius=2.0), 'width': 2, 'layers': 1, 'kernel_width': 4},
    ),
    'gnn': (_NODES, _NODES[..., :1], {'edges': np.array([[0, 1, 2], [1, 2, 0]]), 'hidden': 8, 'rounds': 1}),
    'neural_ode': (_PATHS, _PATHS, {'t': _T, 'width': 8, 'depth': 1, 'horizon': 2, 'margin': 10.0}),
    'monotone_law': (np.linspace(-1, 1, 12), np.tanh(np.linspace(-1, 1, 12)), {'hidden': 4, 'polish': 0}),
    'play_hysteresis': (_HIST, np.tanh(_HIST), {'plays': 2, 'hidden': 2, 'input_is': 'effort'}),
    'diffusion_field': (_X2, _fields(6)[0], {'width': 4, 'levels': 1, 'noise_steps': 10, 'count': 2}),
    'pde_net': (_SNAP, 0.99 * _SNAP, {'dt': 0.01, 'dx': 2 * np.pi / 16}),
}
#: Families whose loss draws fresh noise per call: their permuted and full losses are not comparable.
NOISY = {'diffusion_field'}


def test_every_registered_family_is_exercised():
    assert set(FAMILIES) == set(DEEP)


def _honoured(calls, name, batch=2):
    assert calls, f'{name} never reached minimise'
    for call in calls:
        adam = call['schedule'].adam_steps
        assert call['samples'] > batch, (name, call['samples'])
        assert call['sizes'][:adam] == [batch] * adam, name
        assert set(call['sizes'][adam:]) <= {None}, name
        permuted, full = call['pair']
        if name not in NOISY:
            assert float(permuted) == pytest.approx(float(full), rel=1e-5, abs=1e-6), name


@pytest.mark.parametrize('name', sorted(FAMILIES))
def test_every_registered_family_honours_the_batch(name, calls):
    x, y, settings = FAMILIES[name]
    model = DEEP[name].build(
        shapes=(x.shape[1:], y.shape[1:]), checkpoint=None, steps=3, learning_rate=1e-3, batch=2, **settings
    )
    model.fit(x.reshape(len(x), -1), y.reshape(len(y), -1))
    _honoured(calls, name)


def test_the_learned_preconditioner_batches_its_matrices(calls):
    def matrix(n, k):
        main, off = 2.0 + k * np.ones(n), -np.ones(n - 1)
        return sp.diags([off, main, off], [-1, 0, 1], format='csr')

    schedule = Schedule(adam_steps=3, learning_rate=1e-3, batch=2)
    LearnedPreconditioner(hidden=4, rounds=1, schedule=schedule, runtime=RT).fit([
        matrix(n, k) for n, k in ((4, 0.1), (5, 0.5), (6, 1.0), (4, 2.0))
    ])
    _honoured(calls, 'preconditioner')


def test_a_physics_informed_field_refuses_a_batch_by_name():
    with pytest.raises(Refused, match='physics-informed field cannot take batch=2'):
        PhysicsInformedField(hidden=(4,), schedule=Schedule(adam_steps=2, batch=2)).fit(Poisson())
