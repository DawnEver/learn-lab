"""Checkpoint/resume of the one training loop: a resumed seeded fit ends on the uninterrupted fit's weights.

The loss draws fresh noise every step, so a resume that dropped the RNG state (or the Adam moments, or the
step count) would land elsewhere: equality here is bit for bit, under :meth:`Runtime.seeded`'s kernels. The
planted controls are a payload changed after writing and a checkpoint written by another config.
"""

import json

import pytest
import torch

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


def _fit(checkpoint=None, *, stop_after=None, config=CONFIG, schedule=SCHEDULE):
    RT.seeded()
    net = MLP([2, 8, 1]).to(RT.torch_device, RT.torch_dtype)
    x = torch.linspace(-1, 1, 16, dtype=RT.torch_dtype).reshape(8, 2)
    calls = 0

    def loss():
        nonlocal calls
        calls += 1
        if stop_after is not None and calls > stop_after:
            raise Interrupted
        noisy = x + 0.1 * torch.randn(x.shape, dtype=x.dtype)
        return (net(noisy) ** 2).mean()

    history = minimise(net.parameters(), loss, schedule, checkpoint=checkpoint, config=config)
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
