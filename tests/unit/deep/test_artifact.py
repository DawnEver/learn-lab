"""The artifact contract on the deep families: torch state reloads bit for bit and may move device.

Fits here run a few steps: a round trip must reproduce WHATEVER was fitted, trained or not. Same device, same
dtype -> exact equality (``np.array_equal``); CUDA -> CPU in float32 -> within :data:`CROSS_DEVICE` of the
output's own scale (different kernels sum in a different order; the bits may move, the answer may not).
"""

import hashlib
import json

import numpy as np
import pytest
import torch

from learn_lab import artifact
from learn_lab.deep.constitutive import MonotoneLaw
from learn_lab.deep.dynamics import NeuralODE
from learn_lab.deep.field import FieldModel
from learn_lab.deep.fit import Schedule
from learn_lab.deep.graph import GraphModel, Graphs
from learn_lab.deep.hysteresis import PlayHysteresis
from learn_lab.deep.networks import MLP
from learn_lab.deep.operator import FNO, DeepONet
from learn_lab.deep.physics import PhysicsInformedField
from learn_lab.deep.runtime import Runtime
from learn_lab.deep.surrogate import DeepEnsemble
from learn_lab.errors import Refused

from ._poisson import Poisson

FEW = Schedule(adam_steps=3, learning_rate=1e-2)
CPU64, CPU32 = Runtime(device='cpu'), Runtime(device='cpu', dtype='float32')
IDENTITY = 'sha256:planted-training-data'
#: Relative to the largest |output| (float32 eps ~1.2e-7, a few hundred ops deep), with an absolute floor in
#: the output's unit for an output near zero.
CROSS_DEVICE, FLOOR = 1e-5, 1e-6
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='this torch sees no CUDA device')

RNG = np.random.default_rng(5)
X = RNG.uniform(-1, 1, (32, 2))
Y = np.column_stack([np.sin(2 * X[:, 0]), X[:, 0] * X[:, 1]])
GRID = RNG.uniform(-1, 1, (6, 1, 8, 8))
SENSORS, POINTS = RNG.uniform(-1, 1, (6, 5)), np.linspace(0, 1, 7)[:, None]
T = np.linspace(0, 1, 12)
TRAJECTORIES = np.stack([np.column_stack([np.cos(T + p), np.sin(T + p)]) for p in (0.0, 0.7, 1.4)])
HISTORIES = [np.sin(np.linspace(0, 6, 30)) * a for a in (1.0, 0.6)]


def _graphs(seed: int) -> Graphs:
    rng = np.random.default_rng(seed)
    ring = np.array([[0, 1, 2, 3, 1, 2, 3, 0], [1, 2, 3, 0, 0, 1, 2, 3]])
    return Graphs.batch([(rng.uniform(-1, 1, (4, 2)), ring, rng.uniform(0, 1, (8, 1))) for _ in range(3)])


GRAPHS = _graphs(0)


def _ensemble(f):
    p = f.predict(X[:5], variance=True, gradient=True)
    return [p.mean, p.variance, p.gradient]


def _hysteresis(f):
    state = f.initial_state(2)
    return [f.respond(HISTORIES[1][:10]), f.value(state, np.array([0.1, -0.2])), f.derivative(state, [0.1, -0.2])]


#: family -> (declaration, its fit, its answers, an out-of-domain call). Every deep family.
CASES = {
    'deep_ensemble': (
        DeepEnsemble(hidden=(8,), members=2, schedule=FEW, runtime=CPU32),
        lambda m: m.fit(X, Y),
        _ensemble,
        lambda f: f.predict(np.array([[1.5, 0.0]])),
    ),
    'physics_informed_field': (
        PhysicsInformedField(hidden=(8, 8), schedule=FEW, runtime=CPU64),
        lambda m: m.fit(Poisson(points=16), identify=('k',), observations={'u': (POINTS, np.sin(np.pi * POINTS))}),
        lambda f: [f.evaluate(POINTS), np.array(list(f.parameters.values()))],
        lambda f: f.evaluate(np.array([[1.2]])),
    ),
    'monotone_law': (
        MonotoneLaw(hidden=4, schedule=FEW, runtime=CPU64),
        lambda m: m.fit(X[:, 0], np.tanh(2 * X[:, 0])),
        lambda f: [f.value(X[:5, 0]), f.derivative(X[:5, 0])],
        lambda f: f.value(np.array([3.0])),
    ),
    'play_hysteresis': (
        PlayHysteresis(input_is='displacement', plays=4, hidden=4, schedule=FEW, runtime=CPU64),
        lambda m: m.fit(HISTORIES, [np.tanh(2 * h) for h in HISTORIES]),
        _hysteresis,
        lambda f: f.respond(np.array([0.0, 2.0])),
    ),
    'field_model': (
        FieldModel(width=4, levels=1, schedule=FEW, runtime=CPU32),
        lambda m: m.fit(GRID, GRID**2),
        lambda f: [f.predict(GRID[:2])],
        lambda f: f.predict(GRID[:1] + 5.0),
    ),
    'deep_onet': (
        DeepONet(hidden=(8,), basis=4, schedule=FEW, runtime=CPU32),
        lambda m: m.fit(SENSORS, POINTS, SENSORS.sum(1, keepdims=True) * POINTS.T),
        lambda f: [f.predict(SENSORS[:2], POINTS[:3])],
        lambda f: f.predict(SENSORS[:1], np.array([[2.0]])),
    ),
    'fno': (
        FNO(modes=(2, 2), width=4, layers=2, schedule=FEW, runtime=CPU32),
        lambda m: m.fit(GRID, -GRID),
        lambda f: [f.predict(GRID[:2])],
        lambda f: f.predict(GRID[:1] + 5.0),
    ),
    'graph_model': (
        GraphModel(hidden=8, rounds=2, schedule=FEW, runtime=CPU32),
        lambda m: m.fit(GRAPHS, GRAPHS.nodes[:, :1] * 2),
        lambda f: [f.predict(GRAPHS)],
        lambda f: f.predict(Graphs(GRAPHS.nodes + 5.0, GRAPHS.edges, GRAPHS.edge_features, GRAPHS.offsets)),
    ),
    'neural_ode': (
        NeuralODE(hidden=(8,), horizon=3, schedule=FEW, runtime=CPU64),
        lambda m: m.fit(T, TRAJECTORIES),
        lambda f: [
            f.rate(TRAJECTORIES[0, :4]),
            f.integrate(np.array([[0.5, 0.5], [0.2, 0.6]]), np.array([0.0, 1e-3, 2e-3])),
        ],
        lambda f: f.rate(np.array([[3.0, 0.0]])),
    ),
}


def test_the_cases_cover_every_deep_family() -> None:
    deep = {name for name, (_, fitted) in artifact.FAMILIES.items() if fitted.startswith('learn_lab.deep.')}
    assert set(CASES) == deep


def test_the_comparison_sees_one_ulp() -> None:
    """PLANTED CONTROL: a float32 answer one unit in the last place away is a different answer."""
    a = np.float32([0.3, 1.7]).astype(float)
    assert np.array_equal(a, a.copy())
    assert not np.array_equal(a, np.float32([0.3, np.nextafter(np.float32(1.7), np.float32(2))]).astype(float))


@pytest.mark.parametrize('family', list(CASES))
def test_a_reloaded_deep_model_answers_bit_for_bit_and_refuses_the_same_stranger(family, tmp_path) -> None:
    declaration, fit, answers, stranger = CASES[family]
    fitted = fit(declaration)
    path = artifact.save(fitted, tmp_path / family, declaration=declaration, data_identity=IDENTITY)
    loaded = artifact.load(path)
    assert type(loaded) is type(fitted)
    before, after = answers(fitted), answers(loaded)
    assert all(np.array_equal(a, b) for a, b in zip(before, after, strict=True))
    for model in (fitted, loaded):  # the original refuses it too: the query is genuinely a stranger
        with pytest.raises(Refused):
            stranger(model)
    record = artifact.manifest(path)
    assert record['family'] == family
    assert record['runtime'] == {'device': 'cpu', 'dtype': declaration.runtime.dtype, 'seed': 0}
    assert set(record['payload']) == {artifact.ARRAYS, artifact.WEIGHTS}


def test_the_identified_parameter_survives_the_round_trip(tmp_path) -> None:
    declaration, fit, _, _ = CASES['physics_informed_field']
    fitted = fit(declaration)
    assert fitted.parameters['k'] != 1.0  # identified: it moved off its nominal value
    loaded = artifact.load(artifact.save(fitted, tmp_path / 'p', declaration=declaration, data_identity=IDENTITY))
    assert loaded.parameters == fitted.parameters


def test_loading_leaves_the_callers_torch_seed_where_it_was(tmp_path) -> None:
    declaration, fit, _, _ = CASES['deep_ensemble']
    path = artifact.save(fit(declaration), tmp_path / 'e', declaration=declaration, data_identity=IDENTITY)
    state = torch.get_rng_state()
    MLP([2, 8, 1])
    assert not torch.equal(state, torch.get_rng_state())  # PLANTED CONTROL: building a module draws
    state = torch.get_rng_state()
    artifact.load(path)
    assert torch.equal(state, torch.get_rng_state())


def test_one_changed_weight_byte_is_refused(tmp_path) -> None:
    declaration, fit, _, _ = CASES['monotone_law']
    path = artifact.save(fit(declaration), tmp_path / 'law', declaration=declaration, data_identity=IDENTITY)
    artifact.load(path)  # PLANTED CONTROL: untouched, it loads
    data = bytearray((path / artifact.WEIGHTS).read_bytes())
    data[-40] ^= 0x01
    (path / artifact.WEIGHTS).write_bytes(bytes(data))
    with pytest.raises(Refused, match='does not match its hash'):
        artifact.load(path)


@CUDA
def test_a_fit_saved_on_cuda_loads_on_cuda_exactly_and_on_the_cpu_within_float32(tmp_path) -> None:
    declaration = DeepEnsemble(
        hidden=(16, 16), members=2, schedule=FEW, runtime=Runtime(device='cuda', dtype='float32')
    )
    fitted = declaration.fit(X, Y)
    path = artifact.save(fitted, tmp_path / 'e', declaration=declaration, data_identity=IDENTITY)
    assert artifact.manifest(path)['runtime']['device'] == 'cuda'
    expected = fitted.predict(X, gradient=True)
    same = artifact.load(path).predict(X, gradient=True)
    assert np.array_equal(same.mean, expected.mean)
    assert np.array_equal(same.gradient, expected.gradient)
    moved = artifact.load(path, device='cpu')
    assert moved.runtime.device == 'cpu'
    assert {p.device.type for net in moved.nets for p in net.parameters()} == {'cpu'}
    on_cpu = moved.predict(X, gradient=True)
    for got, want in ((on_cpu.mean, expected.mean), (on_cpu.gradient, expected.gradient)):
        assert np.max(np.abs(got - want)) <= CROSS_DEVICE * np.max(np.abs(want)) + FLOOR
    # PLANTED CONTROL: the bar can say no -- a one-step-further fit is outside it.
    other = DeepEnsemble(hidden=(16, 16), members=2, schedule=Schedule(adam_steps=4, learning_rate=1e-2))
    nudged = other.fit(X, Y).predict(X).mean
    assert np.max(np.abs(nudged - expected.mean)) > CROSS_DEVICE * np.max(np.abs(expected.mean)) + FLOOR


def test_a_module_outside_the_table_is_never_built(tmp_path) -> None:
    """A re-sealed manifest naming torch's own Sequential: the hash passes, the module table refuses."""
    declaration, fit, _, _ = CASES['monotone_law']
    path = artifact.save(fit(declaration), tmp_path / 'law', declaration=declaration, data_identity=IDENTITY)
    record = json.loads((path / artifact.MANIFEST).read_text(encoding='utf-8'))
    record.pop('content_hash')
    assert record['state']['fields']['net']['module'] == 'MonotoneScalar'
    record['state']['fields']['net']['module'] = 'Sequential'
    canonical = json.dumps(record, sort_keys=True, separators=(',', ':')).encode()
    record['content_hash'] = 'sha256:' + hashlib.sha256(canonical).hexdigest()
    (path / artifact.MANIFEST).write_text(json.dumps(record), encoding='utf-8')
    with pytest.raises(Refused, match="module is one of .*got 'Sequential'"):
        artifact.load(path)


def test_a_cuda_placement_on_a_box_without_cuda_is_refused_by_the_runtime(tmp_path, monkeypatch) -> None:
    declaration, fit, _, _ = CASES['monotone_law']
    path = artifact.save(fit(declaration), tmp_path / 'law', declaration=declaration, data_identity=IDENTITY)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(Refused, match='sees no CUDA device'):
        artifact.load(path, device='cuda')
