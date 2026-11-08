"""Checkpoint/resume of the one training loop: a resumed seeded fit ends on the uninterrupted fit's weights.

The loss draws fresh noise every step, so a resume that dropped the RNG state (or the Adam moments, or the
step count) would land elsewhere: equality here is bit for bit, under :meth:`Runtime.seeded`'s kernels. The
planted controls are a payload changed after writing and a checkpoint written by another config.
"""

import json

import numpy as np
import pytest
import torch

from learn_lab.deep import fit
from learn_lab.deep.constitutive import MonotoneLaw
from learn_lab.deep.fit import Checkpoint, Schedule, minimise
from learn_lab.deep.networks import MLP
from learn_lab.deep.runtime import Runtime
from learn_lab.errors import Refused

RT = Runtime(device='cpu', seed=3)
SCHEDULE = Schedule(adam_steps=12, learning_rate=1e-2)
SHORT = Schedule(adam_steps=4, learning_rate=1e-2)
CONFIG = 'planted-config'


class Interrupted(Exception):
    pass


POLISHED = Schedule(adam_steps=4, learning_rate=1e-2, lbfgs_steps=6)


def _fit(checkpoint=None, *, stop_after=None, config=CONFIG, schedule=SCHEDULE, shift=0.0):
    RT.seeded()
    net = MLP([2, 8, 1]).to(RT.torch_device, RT.torch_dtype)
    x = torch.linspace(-1, 1, 16, dtype=RT.torch_dtype).reshape(8, 2) + shift
    calls = 0

    def loss():
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
    assert json.loads((tmp_path / 'checkpoint.json').read_text())['step'] == 5
    resumed, resumed_history = _fit(Checkpoint(tmp_path, every=5))
    for a, b in zip(whole.state_dict().values(), resumed.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    assert resumed_history == whole_history


def test_a_checkpoint_holds_weights_optimizer_step_and_rng(tmp_path):
    _fit(Checkpoint(tmp_path, every=4))
    record = json.loads((tmp_path / 'checkpoint.json').read_text())
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
            lambda: torch.zeros(()),
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
    record = json.loads((tmp_path / 'checkpoint.json').read_text())
    assert record['phase'] == 'lbfgs' and record['step'] > POLISHED.adam_steps
    state = torch.load(tmp_path / 'weights.pt', weights_only=True)
    assert 'old_dirs' in state['optimizer']['state'][0]


def test_a_checkpoint_whose_phase_disagrees_with_its_step_is_refused(tmp_path):
    _fit(Checkpoint(tmp_path, every=1), schedule=POLISHED)
    path = tmp_path / 'checkpoint.json'
    record = json.loads(path.read_text())
    record.pop('content_hash')
    record['phase'] = 'adam'
    record['content_hash'] = fit._sha256(fit._canonical(record))
    path.write_text(json.dumps(record))
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
    path = tmp_path / 'checkpoint.json'
    path.write_text(path.read_text().replace('"step": 4', '"step": 3'))
    with pytest.raises(Refused, match='content hash'):
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
