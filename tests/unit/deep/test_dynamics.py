"""A neural ODE learns a planted damped oscillator from trajectories and predicts one it never saw."""

import numpy as np
import pytest
import torch

from learn_lab.deep.dynamics import NeuralODE, rk4
from learn_lab.deep.fit import Schedule
from learn_lab.deep.runtime import Runtime
from learn_lab.errors import Refused

OMEGA, ZETA = 2.0, 0.1
BAR = 0.01  # max |x error| over the held-out trajectory, unit of x (its amplitude starts at ~0.6)
T = np.linspace(0.0, 5.0, 51)


def _exact(x0, v0, t):
    """The closed form of x'' + 2 zeta omega x' + omega^2 x = 0."""
    wd, a = OMEGA * np.sqrt(1 - ZETA**2), ZETA * OMEGA
    b = (v0 + a * x0) / wd
    e = np.exp(-a * t)
    x = e * (x0 * np.cos(wd * t) + b * np.sin(wd * t))
    v = e * ((-a * x0 + wd * b) * np.cos(wd * t) + (-a * b - wd * x0) * np.sin(wd * t))
    return np.stack([x, v], axis=-1)


def _trajectories(count, seed):
    starts = np.random.default_rng(seed).uniform(-1, 1, (count, 2)) * [1.0, 2.0]
    return np.stack([_exact(x0, v0, T) for x0, v0 in starts])


def test_rk4_is_fourth_order_on_exponential_decay():
    errors = []
    for steps in (10, 20):
        t = torch.linspace(0, 1, steps + 1, dtype=torch.float64)
        x = rk4(lambda s: -s, torch.ones(1, 1, dtype=torch.float64), t)
        errors.append(abs(float(x[0, -1, 0]) - np.exp(-1)))
    assert errors[0] / errors[1] == pytest.approx(16, rel=0.1)  # dimensionless convergence ratio


def _error(fitted):
    truth = _exact(0.6, -0.3, T)
    try:
        return float(np.max(np.abs(fitted.integrate(truth[0][None], T)[0, :, 0] - truth[:, 0])))
    except Refused:  # the trajectory left the envelope: no answer is a failure too
        return np.inf


def test_a_neural_ode_predicts_a_held_out_trajectory():
    model = NeuralODE(
        horizon=4, schedule=Schedule(adam_steps=1000, learning_rate=3e-3), runtime=Runtime(dtype='float32')
    )
    fitted = model.fit(T, _trajectories(16, 0))
    assert _error(fitted) < BAR
    # PLANTED CONTROL: one training step cannot pass the bar.
    assert _error(NeuralODE(schedule=Schedule(adam_steps=1)).fit(T, _trajectories(16, 0))) > BAR
    with pytest.raises(Refused, match='out_of_domain'):
        fitted.integrate(np.array([[5.0, 0.0]]), T)
