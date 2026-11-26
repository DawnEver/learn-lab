"""Checkpoint/resume of the one training loop: a resumed seeded fit ends on the uninterrupted fit's weights.

The loss draws fresh noise every step, so a resume that dropped the RNG state (or the Adam moments, or the
step count) would land elsewhere: equality here is bit for bit, under :meth:`Runtime.seeded`'s kernels. The
planted controls are a payload changed after writing and a checkpoint written by another config.
"""

import json
import tomllib

import numpy as np
import pytest
import tomlkit
import torch

from learn_lab import artifact
from learn_lab.deep import fit
from learn_lab.deep.constitutive import MonotoneLaw
from learn_lab.deep.fit import Checkpoint, Schedule, minimise
from learn_lab.deep.networks import MLP
from learn_lab.deep.runtime import Runtime
from learn_lab.errors import Refused

pytestmark = pytest.mark.torch

RT = Runtime(device='cpu', seed=3)
SCHEDULE = Schedule(adam_steps=12, learning_rate=1e-2)
SHORT = Schedule(adam_steps=4, learning_rate=1e-2)
CONFIG = 'planted-config'


class Interrupted(Exception):
    pass


def _record(directory):
    return tomllib.loads((directory / 'checkpoint.toml').read_text(encoding='utf-8'))


def _reseal(directory, record):
    """Write *record* back with a content hash that matches it: the hostile edit a hash alone cannot refuse."""
    record = {k: v for k, v in record.items() if k != 'content_hash'}
    record['content_hash'] = artifact.sha256(artifact.canonical(record))
    (directory / 'checkpoint.toml').write_text(tomlkit.dumps(record), encoding='utf-8')


POLISHED = Schedule(adam_steps=4, learning_rate=1e-2, lbfgs_steps=6)


def _fit(checkpoint=None, *, stop_after=None, config=CONFIG, schedule=SCHEDULE, shift=0.0):
    RT.seeded()
    net = MLP([2, 8, 1]).to(RT.torch_device, RT.torch_dtype)
    x = torch.linspace(-1, 1, 16, dtype=RT.torch_dtype).reshape(8, 2) + shift
    calls = 0

    def loss(_index=None):
        nonlocal calls
        calls += 1
        if stop_after is not None and calls > stop_after:
            raise Interrupted
        noisy = x + 0.1 * torch.randn(x.shape, dtype=x.dtype)
        return (net(noisy) ** 2).mean()

    history = minimise(net.parameters(), loss, schedule, checkpoint=checkpoint, config=config, data=[x])
    return net, history


def test_a_resumed_fit_reaches_the_uninterrupted_weights(tmp_path):
    whole, whole_history = _fit()
    with pytest.raises(Interrupted):
        _fit(Checkpoint(tmp_path, every=5), stop_after=8)
    assert _record(tmp_path)['step'] == 5
    resumed, resumed_history = _fit(Checkpoint(tmp_path, every=5))
    for a, b in zip(whole.state_dict().values(), resumed.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    assert resumed_history == whole_history


def test_a_checkpoint_holds_weights_optimizer_step_and_rng(tmp_path):
    _fit(Checkpoint(tmp_path, every=4))
    record = _record(tmp_path)
    assert record['step'] == 12
    assert record['payload']['weights.pt'].startswith('sha256:')
    state = torch.load(tmp_path / 'weights.pt', weights_only=True)
    assert set(state) == {'parameters', 'optimizer', 'rng'}
    assert record['phase'] == 'adam'


def test_a_checkpoint_of_other_training_data_is_refused(tmp_path):
    # Planted control: identical shapes and dtype, different values -- only the data hash can tell them apart.
    _fit(Checkpoint(tmp_path, every=4), schedule=SHORT)
    with pytest.raises(Refused, match='another config'):
        _fit(Checkpoint(tmp_path, every=4), schedule=SHORT, shift=1e-3)


def test_a_checkpoint_without_training_data_is_refused(tmp_path):
    with pytest.raises(Refused, match='training data'):
        minimise(
            [torch.zeros(1, requires_grad=True)],
            lambda _index: torch.zeros(()),
            SHORT,
            checkpoint=Checkpoint(tmp_path, every=1),
        )


@pytest.mark.parametrize('stop_after', [3, 6, 11], ids=['adam', 'lbfgs-start', 'lbfgs-mid'])
def test_a_fit_interrupted_in_any_phase_resumes_to_the_uninterrupted_weights(tmp_path, stop_after):
    whole, whole_history = _fit(schedule=POLISHED)
    with pytest.raises(Interrupted):
        _fit(Checkpoint(tmp_path, every=1), stop_after=stop_after, schedule=POLISHED)
    resumed, resumed_history = _fit(Checkpoint(tmp_path, every=1), schedule=POLISHED)
    for a, b in zip(whole.state_dict().values(), resumed.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    assert resumed_history == whole_history
    assert len(whole_history) == POLISHED.adam_steps + POLISHED.lbfgs_steps


def test_a_polish_checkpoint_records_its_phase_and_the_lbfgs_state(tmp_path):
    with pytest.raises(Interrupted):
        _fit(Checkpoint(tmp_path, every=1), stop_after=12, schedule=POLISHED)
    record = _record(tmp_path)
    assert record['phase'] == 'lbfgs' and record['step'] > POLISHED.adam_steps
    state = torch.load(tmp_path / 'weights.pt', weights_only=True)
    assert 'old_dirs' in state['optimizer']['state'][0]


def test_a_checkpoint_whose_phase_disagrees_with_its_step_is_refused(tmp_path):
    _fit(Checkpoint(tmp_path, every=1), schedule=POLISHED)
    record = _record(tmp_path)
    record['phase'] = 'adam'
    _reseal(tmp_path, record)
    with pytest.raises(Refused, match='phase'):
        _fit(Checkpoint(tmp_path, every=1), schedule=POLISHED)


def test_a_corrupted_checkpoint_is_refused(tmp_path):
    _fit(Checkpoint(tmp_path, every=4), schedule=SHORT)
    data = bytearray((tmp_path / 'weights.pt').read_bytes())
    data[-10] ^= 0xFF
    (tmp_path / 'weights.pt').write_bytes(bytes(data))
    with pytest.raises(Refused, match='does not match its hash'):
        _fit(Checkpoint(tmp_path, every=4), schedule=SHORT)


def test_a_tampered_manifest_is_refused(tmp_path):
    _fit(Checkpoint(tmp_path, every=4), schedule=SHORT)
    path = tmp_path / 'checkpoint.toml'
    assert 'step = 4' in path.read_text(encoding='utf-8')
    path.write_text(path.read_text(encoding='utf-8').replace('step = 4', 'step = 3'), encoding='utf-8')
    with pytest.raises(Refused, match='content hash'):
        _fit(Checkpoint(tmp_path, every=4), schedule=SHORT)


def test_a_checkpoint_is_written_as_toml_and_no_json(tmp_path):
    """Manifests are TOML (family ruling 2026-11-25): the directory holds the record and the weights, nothing else."""
    _fit(Checkpoint(tmp_path, every=4), schedule=SHORT)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['checkpoint.toml', 'weights.pt']


def test_the_checkpoint_record_round_trips_every_declared_field(tmp_path):
    """Every field read back is what the fit wrote, re-derived here; the declared set is exactly what is on disk."""
    _, history = _fit(Checkpoint(tmp_path, every=4), schedule=SHORT)
    record = _record(tmp_path)
    assert set(record) == set(fit.FIELDS)
    rest = {k: v for k, v in record.items() if k != 'content_hash'}
    assert rest == {
        'format': fit.FORMAT,
        'version': fit.VERSION,
        'config': rest['config'],  # the identity digest: its own test refuses another config
        'step': SHORT.adam_steps,
        'phase': 'adam',
        'history': history,
        'payload': {'weights.pt': artifact.sha256((tmp_path / 'weights.pt').read_bytes())},
    }
    assert isinstance(rest['config'], str) and rest['config']
    assert record['content_hash'] == artifact.sha256(artifact.canonical(rest))


def test_an_undeclared_checkpoint_key_is_refused_by_name(tmp_path):
    """A resealed record with one extra key: the hash passes, the declaration refuses it -- absence never hides a typo.

    PLANTED CONTROL: the same record resealed unchanged resumes, so the refusal is the extra key's.
    """
    _fit(Checkpoint(tmp_path, every=4), schedule=SHORT)
    record = _record(tmp_path)
    _reseal(tmp_path, record)
    _fit(Checkpoint(tmp_path, every=4), schedule=SHORT)
    _reseal(tmp_path, {**record, 'stpe': 4})
    with pytest.raises(Refused, match="undeclared field 'stpe'"):
        _fit(Checkpoint(tmp_path, every=4), schedule=SHORT)


def test_a_json_checkpoint_is_refused_by_name_with_its_migration_not_read(tmp_path):
    """FORMATS-MIGRATE-ONCE: a ``checkpoint.json`` is never read, even one whose every hash still holds."""
    _fit(Checkpoint(tmp_path, every=4), schedule=SHORT)
    record = _record(tmp_path)
    (tmp_path / 'checkpoint.toml').unlink()
    (tmp_path / 'checkpoint.json').write_text(json.dumps(record, indent=1, sort_keys=True), encoding='utf-8')
    with pytest.raises(Refused, match=r'checkpoint\.json is a retired JSON checkpoint.*fresh checkpoint directory'):
        _fit(Checkpoint(tmp_path, every=4), schedule=SHORT)


@pytest.mark.parametrize(
    'other',
    [{'config': 'another-config'}, {'schedule': Schedule(adam_steps=4, learning_rate=2e-2)}],
    ids=['config', 'schedule'],
)
def test_a_checkpoint_from_another_config_is_refused(tmp_path, other):
    _fit(Checkpoint(tmp_path, every=4), schedule=SHORT)
    with pytest.raises(Refused, match='another config'):
        _fit(Checkpoint(tmp_path, every=4), **{'schedule': SHORT, **other})


def test_a_checkpoint_every_is_positive(tmp_path):
    with pytest.raises(Refused):
        Checkpoint(tmp_path, every=0)


def test_a_family_refuses_a_checkpoint_of_other_training_data(tmp_path):
    # The data identity reaches the families: a law refitted on other values of the same shape is refused.
    law = MonotoneLaw(hidden=4, schedule=Schedule(adam_steps=3, learning_rate=1e-2, lbfgs_steps=2))
    x = np.linspace(-1.0, 1.0, 9)
    law.fit(x, np.tanh(x), checkpoint=Checkpoint(tmp_path, every=1))
    with pytest.raises(Refused, match='another config'):
        law.fit(x, np.tanh(2 * x), checkpoint=Checkpoint(tmp_path, every=1))


COSINE = Schedule(adam_steps=12, learning_rate=1e-2, decay='cosine')


def test_a_cosine_schedule_anneals_adam_to_zero_and_a_constant_one_does_not(monkeypatch):
    # Spy on the class the loop builds: torch wraps `step` onto each optimizer class at its first instance, so
    # once any AdamW exists `AdamW.step` is its own attribute and a spy on `Adam.step` sees no call.
    rates = []
    adam_step = torch.optim.AdamW.step

    def spy(self, *args, **kwargs):
        rates.append(self.param_groups[0]['lr'])
        return adam_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, 'step', spy)
    net = MLP([2, 8, 1]).to(RT.torch_device, RT.torch_dtype)
    x = torch.ones(4, 2, dtype=RT.torch_dtype)

    def loss(_index=None):
        return (net(x) ** 2).mean()

    minimise(net.parameters(), loss, COSINE)
    expected = [1e-2 * (1 + np.cos(np.pi * k / 12)) / 2 for k in range(12)]
    np.testing.assert_allclose(rates, expected, rtol=0, atol=1e-15)  # 1/step: the half-cosine from lr to ~0
    rates.clear()
    minimise(net.parameters(), loss, SCHEDULE)  # PLANTED CONTROL: the constant schedule keeps its rate
    assert rates == [1e-2] * 12


def test_a_cosine_fit_resumes_to_the_uninterrupted_weights(tmp_path):
    whole, whole_history = _fit(schedule=COSINE)
    with pytest.raises(Interrupted):
        _fit(Checkpoint(tmp_path, every=5), stop_after=8, schedule=COSINE)
    resumed, resumed_history = _fit(Checkpoint(tmp_path, every=5), schedule=COSINE)
    for a, b in zip(whole.state_dict().values(), resumed.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    assert resumed_history == whole_history


def test_an_unknown_decay_is_refused():
    with pytest.raises(Refused, match='decay'):
        Schedule(decay='linear')
