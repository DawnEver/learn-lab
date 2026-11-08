"""A learned play-bank hysteresis identifies a planted Prandtl-Ishlinskii law and predicts a reversal curve."""

import itertools

import numpy as np
import pytest
import torch

from learn_lab.deep.hysteresis import PlayHysteresis, dissipation, play
from learn_lab.errors import Refused

R_TRUE = np.linspace(0.0, 0.9, 60)  # input unit
W_TRUE = np.exp(-(((R_TRUE - 0.3) / 0.2) ** 2))
W_TRUE *= 0.7 / W_TRUE.sum()
BAR = 0.02  # max |y error| over a held-out history, fraction of the major loop's output span; measured 5.3e-3, 6.2e-3


def _truth(x):
    """A planted generalized Prandtl-Ishlinskii operator from the demagnetised state: 60 plays, saturating map."""
    p, out = np.zeros_like(R_TRUE), []
    for xi in x:
        p = np.clip(p, xi - R_TRUE, xi + R_TRUE)
        s = 0.3 * xi + W_TRUE @ p
        out.append(1.2 * np.tanh(2.0 * s) + 0.2 * s)
    return np.array(out)


def _path(*turns, step=0.05):
    """A piecewise-linear input through the turning points, sampled at most *step* apart."""
    pieces = [np.linspace(a, b, max(2, int(np.ceil(abs(b - a) / step)) + 1))[1:] for a, b in itertools.pairwise(turns)]
    return np.concatenate([[turns[0]], *pieces])


TRAIN = [
    _path(0.0, 1.0, -1.0, 1.0),  # initial curve and the major loop
    _path(0.0, 0.8, -0.6, 0.4, -0.2, 0.1),  # symmetric minor loops, shrinking
    _path(0.0, 1.0, 0.2, 0.6, 0.2, 0.6, -1.0, -0.3, -0.7, -0.3),  # biased minor loops on both branches
]
HELD_OUT = [
    _path(0.0, 1.0, -0.4, 1.0),  # a first-order reversal curve
    _path(0.0, -1.0, 0.5, -0.1, 0.3, -0.9),  # nested reversals never trained on
]


@pytest.fixture(scope='module')
def model():
    return PlayHysteresis().fit(TRAIN, [_truth(x) for x in TRAIN])  # measured 5.5 s on the local GPU


def test_a_held_out_history_and_a_reversal_curve_are_predicted_within_the_bar(model):
    span = np.ptp(_truth(TRAIN[0]))
    for x in HELD_OUT:
        error = np.max(np.abs(model.respond(x) - _truth(x))) / span
        assert error < BAR


def test_the_response_is_rate_independent_and_monotone_on_each_branch(model):
    coarse, fine = _path(0.0, 1.0, -0.4, 1.0, step=0.1), _path(0.0, 1.0, -0.4, 1.0, step=0.01)
    y_fine = model.respond(fine)
    shared = np.isin(np.round(fine, 9), np.round(coarse, 9))
    np.testing.assert_allclose(model.respond(coarse), y_fine[shared], rtol=0, atol=1e-12)  # unit of y
    dy = np.diff(y_fine) * np.sign(np.diff(fine))
    assert np.all(dy > 0)  # every branch ascends with its input


def test_every_closed_cycle_dissipates_and_the_planted_active_models_do_not(model):
    rng = np.random.default_rng(1)
    for _ in range(5):
        turns = rng.uniform(-1, 1, 6)
        cycle = _path(*turns, turns[0])
        assert dissipation(model.step, model.initial_state(), cycle) > 0  # x-unit * y-unit
    cycle = _path(-0.8, 0.8, -0.8)
    # PLANTED CONTROLS: a negative threshold leads its input; a decreasing map inverts the loop.
    thresholds = torch.tensor([-0.2, 0.1], dtype=torch.float64)

    def leading(state, x):
        state = play(torch.as_tensor(state), torch.as_tensor(x), thresholds)
        return state.numpy(), state.sum(-1).numpy()

    def inverted(state, x):
        state, y = model.step(state, x)
        return state, -y

    assert dissipation(leading, np.zeros((1, 2)), cycle) < 0
    assert dissipation(inverted, model.initial_state(), cycle) < 0


def test_the_step_hands_a_newton_solver_its_value_and_tangent(model):
    """The hybrid seam: a classical stepper inverts the model by Newton, carrying the play state itself."""
    x_true = HELD_OUT[1]
    y_target = model.respond(x_true)
    state, x, recovered = model.initial_state(), np.zeros(1), []
    lo, hi = model.envelope.lower[0], model.envelope.upper[0]
    for target in y_target:
        a, b = lo, hi  # bracket: the step is strictly increasing in x
        for _ in range(60):
            r = model.value(state, x)[0] - target
            if abs(r) < 1e-12:
                break
            a, b = (a, x[0]) if r > 0 else (x[0], b)
            x = x - r / model.derivative(state, x)
            if not a < x[0] < b:
                x = np.array([(a + b) / 2])
        state, _ = model.step(state, x)
        recovered.append(x[0])
    np.testing.assert_allclose(recovered, x_true, rtol=0, atol=1e-9)  # input unit
    probe, state = np.array([0.31]), model.step(model.initial_state(), np.array([0.9]))[0]
    fd = (model.value(state, probe + 1e-6) - model.value(state, probe - 1e-6)) / 2e-6
    np.testing.assert_allclose(model.derivative(state, probe), fd, rtol=1e-6)  # y-unit per x-unit


def test_an_input_beyond_the_trained_range_is_refused(model):
    with pytest.raises(Refused, match='out_of_domain'):
        model.step(model.initial_state(), np.array([1.5]))
    assert np.all(model.thresholds >= 0)
