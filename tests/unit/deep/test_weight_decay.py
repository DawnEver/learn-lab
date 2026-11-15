"""``Schedule.weight_decay``: decoupled (AdamW) decay on the Adam phase only, resumable, accepted by every family.

PLANTED CONTROLS: ``weight_decay = 0`` reproduces the plain ``torch.optim.Adam`` loop bit for bit, while a
positive decay changes the fit and shrinks the weight norm -- so the comparison can say no in both directions.
"""

import pytest
import torch

from learn_lab.deep.fit import Checkpoint, Schedule
from learn_lab.deep.registry import DEEP
from learn_lab.errors import Refused

from .test_batch import FAMILIES, RT, Interrupted, N, _fit, _problem, calls  # noqa: F401 -- fixture


def _norm(net) -> float:
    return float(sum((p.detach() ** 2).sum() for p in net.parameters()))


def test_zero_decay_is_the_plain_adam_loop_bit_for_bit():
    net, history, _ = _fit(Schedule(adam_steps=20, learning_rate=1e-2, decay='cosine', weight_decay=0.0))
    reference, loss, _ = _problem([])
    adam = torch.optim.Adam(reference.parameters(), lr=1e-2)
    schedule = Schedule(adam_steps=20, learning_rate=1e-2, decay='cosine')
    expected = []
    for step in range(20):
        for group in adam.param_groups:
            group['lr'] = schedule.rate(step)
        adam.zero_grad(set_to_none=True)
        value = loss(None)
        value.backward()
        adam.step()
        expected.append(float(value.detach()))
    assert history == expected
    for a, b in zip(net.state_dict().values(), reference.state_dict().values(), strict=True):
        assert torch.equal(a, b)


def test_a_positive_decay_shrinks_the_weights():
    plain, plain_history, _ = _fit(Schedule(adam_steps=200, learning_rate=1e-2))
    decayed, decayed_history, _ = _fit(Schedule(adam_steps=200, learning_rate=1e-2, weight_decay=1.0))
    assert decayed_history != plain_history
    assert _norm(decayed) < 0.9 * _norm(plain)


def test_the_polish_takes_no_decay():
    """Past the Adam phase the decay is inert: an L-BFGS-only fit is the same whatever the decay."""
    _, plain, _ = _fit(Schedule(adam_steps=0, lbfgs_steps=3))
    _, decayed, _ = _fit(Schedule(adam_steps=0, lbfgs_steps=3, weight_decay=1.0))
    assert decayed == plain


def test_a_decayed_fit_resumed_mid_adam_is_the_uninterrupted_fit(tmp_path):
    schedule = Schedule(adam_steps=12, learning_rate=1e-2, weight_decay=0.1, batch=5)
    whole, whole_history, _ = _fit(schedule)
    with pytest.raises(Interrupted):
        _fit(schedule, checkpoint=Checkpoint(tmp_path, every=5), stop_after=8)
    resumed, resumed_history, _ = _fit(schedule, checkpoint=Checkpoint(tmp_path, every=5))
    assert resumed_history == whole_history
    for a, b in zip(whole.state_dict().values(), resumed.state_dict().values(), strict=True):
        assert torch.equal(a, b)


def test_the_decay_is_part_of_the_checkpoint_identity(tmp_path):
    _fit(Schedule(adam_steps=4, learning_rate=1e-2, weight_decay=0.1), checkpoint=Checkpoint(tmp_path, every=2))
    with pytest.raises(Refused, match='another config'):
        _fit(Schedule(adam_steps=4, learning_rate=1e-2, weight_decay=0.2), checkpoint=Checkpoint(tmp_path, every=2))


@pytest.mark.parametrize('weight_decay', [-1e-3, float('inf'), float('nan'), True, '0.1', None])
def test_a_weight_decay_is_a_finite_non_negative_number(weight_decay):
    with pytest.raises(Refused, match='weight decay'):
        Schedule(weight_decay=weight_decay)


@pytest.mark.parametrize('name', sorted(FAMILIES))
def test_every_registered_family_takes_the_decay(name, calls):  # noqa: F811
    x, y, settings = FAMILIES[name]
    model = DEEP[name].build(
        shapes=(x.shape[1:], y.shape[1:]), checkpoint=None, steps=2, learning_rate=1e-3, weight_decay=0.01, **settings
    )
    model.fit(x.reshape(len(x), -1), y.reshape(len(y), -1))
    assert calls and all(c['schedule'].weight_decay == 0.01 for c in calls), name
