"""A learned per-coordinate input scale: the net can steepen itself along a coordinate where the field turns sharply."""

import numpy as np
import pytest
import torch

from learn_lab import artifact
from learn_lab.deep.fit import Schedule
from learn_lab.deep.networks import MLP
from learn_lab.deep.physics import PhysicsInformedField
from learn_lab.deep.runtime import Runtime

from ._poisson import Poisson

SHORT = Schedule(adam_steps=60, learning_rate=1e-2, lbfgs_steps=2)
CPU64 = Runtime(device='cpu', dtype='float64')


def test_an_input_scaled_mlp_starts_as_the_plain_one_and_records_the_option():
    torch.manual_seed(0)
    plain = MLP([2, 8, 1]).double()
    scaled = MLP([2, 8, 1], input_scale=True).double()
    for mine, theirs in zip(list(scaled)[1:], list(plain), strict=True):
        mine.load_state_dict(theirs.state_dict())
    x = torch.randn(5, 2, dtype=torch.float64)
    assert torch.equal(scaled(x), plain(x))  # the scale starts at one
    assert scaled.arguments == {'widths': (2, 8, 1), 'activation': 'tanh', 'input_scale': True}
    assert next(n for n, _ in scaled.named_parameters()) == '0.scale'
    # Planted control: the default has no scale, so a fit cannot learn one.
    assert all('scale' not in n for n, _ in plain.named_parameters())
    assert plain.arguments['input_scale'] is False


def test_a_physics_fit_learns_the_scale_and_the_artifact_keeps_it(tmp_path):
    model = PhysicsInformedField(hidden=(8, 8), schedule=SHORT, runtime=CPU64, input_scale=True)
    fitted = model.fit(Poisson(points=32))
    scale = fitted.net[0].scale.detach().numpy()
    assert np.abs(scale - 1.0).max() > 1e-4  # dimensionless: the scale trained
    path = artifact.save(fitted, tmp_path / 'f', declaration=model, data_identity='sha256:planted')
    again = artifact.load(path)
    x = np.linspace(0.0, 1.0, 7)[:, None]
    np.testing.assert_array_equal(again.evaluate(x), fitted.evaluate(x))
    # Planted control: without the option the net carries no scale.
    plain = PhysicsInformedField(hidden=(8, 8), schedule=SHORT, runtime=CPU64).fit(Poisson(points=32))
    assert not hasattr(plain.net[0], 'scale')


def test_the_default_is_unscaled():
    assert PhysicsInformedField().input_scale is False
    with pytest.raises(TypeError):
        MLP([1, 1], True)  # keyword only
