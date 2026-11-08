"""A learned play-bank hysteresis identifies a planted Prandtl-Ishlinskii law in BOTH causalities -- driven by the
effort (B(H)) and by the displacement (H(B)) -- predicts reversal curves, and dissipates in the declared orientation.
"""

import itertools

import numpy as np
import pytest
import torch

from learn_lab.deep.hysteresis import PlayHysteresis, dissipation, play
from learn_lab.errors import Refused

R_TRUE = np.linspace(0.0, 0.9, 60)  # input unit
W_TRUE = np.exp(-(((R_TRUE - 0.3) / 0.2) ** 2))
W_TRUE *= 0.7 / W_TRUE.sum()
LOSS_BAR = (
    0.05  # |learned - planted| loop energy, fraction of the planted; measured 0.11 % (effort), 0.69 % (displacement)
)
# measured 5.3e-3, 6.2e-3 (effort), 4.6e-3, 5.9e-3 (displacement)
BAR = 0.02  # max |y error| over a held-out history, fraction of the major loop's output span


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


def _truth_step(state, x):
    """The planted law as a ``(state, x) -> (state, y)`` step, effort-driven."""
    p = np.clip(state, x[0] - R_TRUE, x[0] + R_TRUE)
    s = 0.3 * x[0] + W_TRUE @ p
    return p, np.array([1.2 * np.tanh(2.0 * s) + 0.2 * s])


def _data(input_is, histories):
    """``(inputs, outputs)``: H -> B for the effort orientation, the same loops read B -> H for the displacement."""
    outputs = [_truth(h) for h in histories]
    return (histories, outputs) if input_is == 'effort' else (outputs, list(histories))


@pytest.fixture(scope='module', params=['effort', 'displacement'])
def case(request):
    inputs, outputs = _data(request.param, TRAIN)
    return PlayHysteresis(input_is=request.param).fit(inputs, outputs), request.param  # ~60 s each, CPU


def test_a_held_out_history_and_a_reversal_curve_are_predicted_within_the_bar(case):
    model, input_is = case
    span = np.ptp(_data(input_is, TRAIN[:1])[1][0])
    inputs, outputs = _data(input_is, HELD_OUT)
    for x, y in zip(inputs, outputs, strict=True):
        assert np.max(np.abs(model.respond(x) - y)) / span < BAR


def test_the_response_is_rate_independent_and_monotone_on_each_branch(case):
    model, _ = case
    top = 0.95 * model.envelope.upper[0]
    turns = (0.0, top, -0.4 * top, top)
    coarse, fine = _path(*turns, step=0.1 * top), _path(*turns, step=0.01 * top)
    y_fine = model.respond(fine)
    shared = np.isin(np.round(fine, 9), np.round(coarse, 9))
    np.testing.assert_allclose(model.respond(coarse), y_fine[shared], rtol=0, atol=1e-12)  # unit of y
    dy = np.diff(y_fine) * np.sign(np.diff(fine))
    assert np.all(dy > 0)  # every branch ascends with its input


def test_every_closed_cycle_dissipates_in_the_declared_orientation(case):
    model, input_is = case
    rng = np.random.default_rng(1)
    top = 0.95 * model.envelope.upper[0]
    for _ in range(5):
        turns = rng.uniform(-top, top, 6)
        cycle = _path(*turns, turns[0], step=0.05 * top)
        assert dissipation(model.step, model.initial_state(), cycle, input_is=input_is) > 0  # x-unit * y-unit


def test_the_identified_loop_dissipates_the_planted_energy(case):
    """The major loop's oint H dB is one number whichever variable drives it: each orientation's fit, measured in
    its own declared orientation, reproduces the planted law's loss.
    """
    model, input_is = case
    h = _path(-1.0, 1.0, -1.0)
    planted = dissipation(_truth_step, np.zeros_like(R_TRUE), h, input_is='effort')
    cycle = h if input_is == 'effort' else _truth(np.concatenate([h, h]))[h.size :]
    cycle[-1] = cycle[0]  # the second pass is closed by return-point memory; pin the float
    learned = dissipation(model.step, model.initial_state(), cycle, input_is=input_is)
    assert planted > 0.1  # x-unit * y-unit
    assert learned == pytest.approx(planted, rel=LOSS_BAR, abs=1e-6)


def test_the_effort_construction_driven_by_the_displacement_is_active_and_is_caught():
    """PLANTED CONTROL, the measured defect: a lagging bank fitted B -> H puts the rising branch below the falling
    one, so oint H dB < 0. Measured in its declared (effort) orientation it looks passive -- the declaration lies;
    measured as what it is, displacement-driven, :func:`dissipation` says no.
    """
    b, h = _data('displacement', TRAIN)
    wrong = PlayHysteresis(input_is='effort').fit(b, h)
    top = 0.95 * wrong.envelope.upper[0]
    cycle = _path(-top, top, -top, step=0.05 * top)
    assert dissipation(wrong.step, wrong.initial_state(), cycle, input_is='effort') > 0
    assert dissipation(wrong.step, wrong.initial_state(), cycle, input_is='displacement') < 0  # measured -9.8e-3


def test_the_planted_active_elements_do_not_dissipate(case):
    model, input_is = case
    cycle = _path(-0.5, 0.5, -0.5)
    # PLANTED CONTROLS: a negative threshold leads its input (active when effort-driven); a decreasing map
    # inverts the loop in either orientation.
    thresholds = torch.tensor([-0.2, 0.1], dtype=torch.float64)

    def leading(state, x):
        state = play(torch.as_tensor(state), torch.as_tensor(x), thresholds)
        return state.numpy(), state.sum(-1).numpy()

    def inverted(state, x):
        state, y = model.step(state, x)
        return state, -y

    assert dissipation(leading, np.zeros((1, 2)), cycle, input_is='effort') < 0
    assert dissipation(inverted, model.initial_state(), cycle * model.envelope.upper[0], input_is=input_is) < 0


def test_a_causality_is_declared_never_guessed():
    with pytest.raises(TypeError):
        PlayHysteresis()
    with pytest.raises(Refused, match='conjugate'):
        PlayHysteresis(input_is='flux').fit(TRAIN, [_truth(x) for x in TRAIN])
    with pytest.raises(Refused, match='conjugate'):
        dissipation(_truth_step, np.zeros_like(R_TRUE), _path(0.0, 0.5, 0.0), input_is='flow')


def test_the_step_hands_a_newton_solver_its_value_and_tangent(case):
    """The hybrid seam: a classical stepper inverts the model by Newton, carrying the play state itself."""
    model, input_is = case
    x_true = _data(input_is, HELD_OUT)[0][1]
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
    top = model.envelope.upper[0]
    probe, state = np.array([0.31 * top]), model.step(model.initial_state(), np.array([0.9 * top]))[0]
    fd = (model.value(state, probe + 1e-6) - model.value(state, probe - 1e-6)) / 2e-6
    np.testing.assert_allclose(model.derivative(state, probe), fd, rtol=1e-6)  # y-unit per x-unit


def test_an_input_beyond_the_trained_range_is_refused(case):
    model, _ = case
    with pytest.raises(Refused, match='out_of_domain'):
        model.step(model.initial_state(), np.array([1.5 * model.envelope.upper[0]]))
    assert np.all(model.thresholds >= 0)
