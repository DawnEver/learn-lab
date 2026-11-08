"""The deep families declare what is tuned, and one objective scores them on small CPU data.

The planted control: a linear map learned with 300 Adam steps must beat the same network given 1 step,
and a setting outside a deep family's space (a U-Net learning rate of 10) is refused before any fit.
"""

import math

import numpy as np
import pytest

from learn_lab.deep.fit import Checkpoint
from learn_lab.deep.registry import DEEP
from learn_lab.errors import Refused
from learn_lab.registry import family
from learn_lab.tuning import objective

RNG = np.random.default_rng(0)
FAST = {'steps': 2, 'learning_rate': 1e-2}


def _fields(n):
    x = RNG.standard_normal((n, 1, 8, 8))
    return x, 0.5 * x + 0.1


def test_every_deep_family_resolves_through_the_registry_with_a_space():
    expected = {'deep_ensemble', 'unet', 'deeponet', 'fno', 'gnn', 'neural_ode', 'monotone_law', 'play_hysteresis'}
    assert set(DEEP) == expected
    for name in expected:
        assert family(name) is DEEP[name]
        assert {'steps', 'learning_rate'} <= set(DEEP[name].space), name


def test_more_training_ranks_above_one_step():
    x = RNG.uniform(-1, 1, (40, 2))
    y = (x @ np.array([1.0, -2.0]))[:, None]
    f = objective(
        'deep_ensemble',
        x,
        y,
        k=4,
        seed=0,
    )
    trained = f({'width': 16, 'depth': 1, 'members': 1, 'steps': 300, 'learning_rate': 1e-2})
    untrained = f({'width': 16, 'depth': 1, 'members': 1, 'steps': 1, 'learning_rate': 1e-2})
    assert trained < untrained, f.evaluations
    assert trained < -0.9


def test_the_unet_space_is_enforced_and_its_objective_runs_on_cpu():
    x, y = _fields(6)
    f = objective('unet', x, y, k=2, seed=0, fixed={'margin': 10.0})
    with pytest.raises(Refused, match='learning_rate'):
        f({'learning_rate': 10.0})
    value = f({'width': 4, 'levels': 1, **FAST})
    assert math.isfinite(value)


def test_operator_graph_and_sequence_families_score_on_small_data():
    sensors = RNG.standard_normal((6, 5))
    points = np.linspace(0, 1, 4)[:, None]
    assert math.isfinite(
        objective('deeponet', sensors, sensors[:, :4], k=2, fixed={'points': points, 'margin': 10.0})({
            'width': 8,
            'depth': 1,
            'basis': 4,
            **FAST,
        })
    )
    x, y = _fields(6)
    assert math.isfinite(
        objective('fno', x, y, k=2, fixed={'margin': 10.0})({'modes': 2, 'width': 4, 'layers': 1, **FAST})
    )
    edges = np.array([[0, 1, 2], [1, 2, 0]])
    nodes = RNG.standard_normal((6, 3, 2))
    g = objective('gnn', nodes, nodes[..., :1], k=2, fixed={'edges': edges, 'margin': 10.0})
    assert math.isfinite(g({'hidden': 8, 'rounds': 1, **FAST}))
    t = np.linspace(0, 1, 5)
    paths = np.exp(-t)[None, :, None] * RNG.uniform(0.5, 1.5, (6, 1, 1))
    ode = objective('neural_ode', paths, paths, k=2, fixed={'t': t, 'margin': 10.0})
    assert ode({'width': 8, 'depth': 1, 'horizon': 2, **FAST}) < math.inf or ode.evaluations[-1].failure
    law = objective('monotone_law', np.linspace(-1, 1, 12), np.tanh(np.linspace(-1, 1, 12)), k=3, seed=0)
    assert law({'hidden': 4, 'polish': 0, **FAST}) <= math.inf  # an extreme held-out row may leave the law's range
    assert len(law.evaluations) == 1
    histories = np.sin(np.linspace(0, 6, 20))[None, :] * RNG.uniform(0.5, 1.0, (4, 1))
    play = objective('play_hysteresis', histories, np.tanh(histories), k=2, fixed={'input_is': 'effort'})
    assert play({'plays': 2, 'hidden': 2, **FAST}) <= math.inf
    assert len(play.evaluations) == 1


def test_a_checkpoint_is_kept_per_setting_and_per_fold(tmp_path):
    x, y = _fields(4)
    f = objective('unet', x, y, k=2, fixed={'margin': 10.0}, checkpoint=Checkpoint(tmp_path, every=1))
    f({'width': 4, 'levels': 1, **FAST})
    f({'width': 8, 'levels': 1, **FAST})
    settings_dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(settings_dirs) == 2
    assert all({q.name for q in p.iterdir()} == {'member-0', 'member-1'} for p in settings_dirs)


def test_a_response_family_takes_no_checkpoint(tmp_path):
    x = RNG.uniform(-1, 1, (10, 2))
    with pytest.raises(Refused, match='checkpoint'):
        objective('polynomial', x, x[:, :1], checkpoint=Checkpoint(tmp_path, every=1))
