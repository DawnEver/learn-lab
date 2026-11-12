"""A learned rate-independent hysteresis: a bank of play operators, a monotone map, the state explicit.

The model is domain-free, so it DECLARES its causality -- which of two work-conjugate variables is its input --
and that declaration decides both the construction and which integral is the cycle's loss. In bond-graph terms
an *effort* (H, force, voltage) and a *displacement* (B, position, charge -- the time integral of a flow) are
conjugate: the work into the element is ``oint effort d(displacement)``.

- ``input_is='effort'``: ``y = F(a x + sum_k w_k p_k(x))`` -- a generalized Prandtl-Ishlinskii operator, plays
  LAGGING their input. The loss is ``oint x dy``; each lagging play gives ``oint x dp = r oint |dp| >= 0``.
- ``input_is='displacement'``: ``u = G(x)``, ``y = a u + sum_k w_k (u - p_k(u))`` -- ``u - p_k(u)`` is the
  Prandtl STOP operator, which LEADS: ``+r`` on a long rise, ``-r`` on a long fall. The loss is ``oint y dx``;
  each stop gives ``oint e du = r oint |dp| >= 0``. This is the exact INVERSE class of the effort form: on a
  branch, ``v = a x + sum w p_r(x)`` has slope ``a + sum_{moving} w``; inverting it, ``x`` has slope
  ``1 / (a + sum_{moving} w)`` in ``v``, steepest just after a reversal and flattening as plays engage --
  which is a stop bank in ``v`` (all stops elastic after a reversal, each saturating at its own threshold),
  with thresholds ``a r_i + sum_{j<i} w_j (r_i - r_j)`` and non-negative weights. And the OUTER map ``F`` of
  one is the INNER map ``G = F^-1`` of the other. So the displacement-driven inverse of any effort-driven law
  in the class is itself in the class, never approximated across it.

That inverse closure holds for ONE outer map. A Preisach play model -- ``y = a x + sum_k f_k(p_k)``, each play
with its own saturating shape, the class a steel's identified play model is -- is outside it: on a branch its
slope ``a + sum_{moving} f_k'(p_k)`` depends on where each moving play sits, not on one aggregate, so no stop bank
with a single ``G`` inverts it and more stops plateau rather than converge. Two options close that:

- ``shaped``: each play (effort) or stop (displacement) passes through its own odd monotone shape
  ``w z + v tanh(s z)`` -- the effort form then contains the Preisach play model;
- ``inverted`` (displacement only): the model IS an effort bank, fitted on the histories read the other way, and
  each step solves the effort step for the input displacement (bisection, then one Newton step that carries the
  implicit tangent). Its loop is the effort bank's loop, so passivity carries over exactly.

Using the effort construction with the displacement as input -- a lagging bank fed B and asked for H -- puts
the ascending branch BELOW the descending one, so ``oint H dB < 0``: an active material. :func:`dissipation`
measures the declared integral, so it says no to exactly that.

``r_k >= 0``, ``w_k >= 0`` and ``a > 0`` through softplus, ``F``/``G`` a
:class:`~learn_lab.deep.networks.MonotoneScalar`. What holds BY CONSTRUCTION, in both orientations:

- rate independence: a play moves only by how far its input went, never by how fast, so inserting samples on a
  monotone stretch of the input changes no output;
- monotone branches: ``dy/dx >= a F' > 0`` (effort), ``dy/dx = G' (a + sum_{held} w) >= a G' > 0``
  (displacement) -- the tangent a Newton solver needs never vanishes;
- passivity: the operator has return-point memory, so any closed cycle splits into simple loops; in each the
  ascending branch lies below (effort) / above (displacement) the descending one, and a monotone ``F``/``G``
  keeps that order: the declared loss is ``>= 0`` (:func:`dissipation`).

The state is the play outputs, carried by the caller: :meth:`FittedHysteresis.step` is a pure
``(state, x) -> (state, y)``, so a classical time-stepper owns time and hands back the converged state.
:meth:`~FittedHysteresis.value` and :meth:`~FittedHysteresis.derivative` are the same step as the two
callables a Newton solve on ``x`` takes (the HYBRID seam). Training is backpropagation through the play
recursion over whole input histories, each starting from the demagnetised state (every play at 0). The model
is valid on the input range it was trained on; ``odd`` makes ``y(-x) = -y(x)`` exact from the demagnetised
state, so that range is symmetric.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal

import numpy as np
import torch
from torch import nn

from learn_lab.deep.fit import Checkpoint, Schedule, minimise
from learn_lab.deep.networks import MonotoneScalar
from learn_lab.deep.runtime import Runtime, numpy
from learn_lab.errors import refuse, require_finite
from learn_lab.validity import BoxEnvelope

__all__ = ['CAUSALITIES', 'Causality', 'FittedHysteresis', 'PlayHysteresis', 'dissipation', 'play']

Step = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
Causality = Literal['effort', 'displacement']
#: The input a hysteresis declares: which of the two work-conjugate variables drives it.
CAUSALITIES: Final = ('effort', 'displacement')


def _causality(input_is: str) -> str:
    if input_is not in CAUSALITIES:
        refuse(f'a hysteresis input is one of {CAUSALITIES} (the conjugate variable driving it), got {input_is!r}')
    return input_is


def play(state: torch.Tensor, x: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    """One step of a bank of plays: ``state (n, K)``, ``x (n,)``, ``thresholds (K,)`` -> the new ``state``.

    A play follows its input at distance ``r`` once the input pulls further than ``r`` away, else holds. The
    threshold test is written out rather than as a clamp, so a NEGATIVE threshold is not silently a clamp
    with crossed bounds: it makes the play LEAD its input, an active element in the effort orientation that
    :func:`dissipation` catches.
    """
    x = x.unsqueeze(-1)
    gap = x - state
    return torch.where(gap > thresholds, x - thresholds, torch.where(gap < -thresholds, x + thresholds, state))


def dissipation(step: Step, state: np.ndarray, cycle: np.ndarray, *, input_is: Causality) -> float:
    """The energy one closed input *cycle* dissipates; >= 0 for a passive model of the declared causality.

    ``oint effort d(displacement)``: ``oint x dy`` when the input is the effort, ``oint y dx`` when it is the
    displacement (``oint x dy = -oint y dx`` on a closed cycle, so measuring the wrong one flips the verdict).
    The cycle is run twice from *state* and measured on the second pass, once the first has carried the state
    onto the cycle. Unit: input unit times output unit (A/m * T = J/m^3 for H and B, either way round).
    """
    _causality(input_is)
    x = require_finite(np.asarray(cycle, float).ravel(), 'the cycle')
    if x.size < 3 or x[0] != x[-1]:
        refuse(f'a dissipation cycle is closed (first input == last) with >= 3 points, got {x.size} points')
    ys = []
    for _ in range(2):
        ys = []
        for xi in x:
            state, y = step(state, np.array([xi]))
            ys.append(float(np.asarray(y).ravel()[0]))
    y = np.array(ys)
    if input_is == 'effort':
        return float(np.sum(0.5 * (x[1:] + x[:-1]) * np.diff(y)))
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


class PlayBank(nn.Module):
    """The operator on scaled input/output: ``F(a x + w . p(x))`` (effort), ``a u + w . (u - p(u))`` with
    ``u = G(x)`` (displacement), or -- *inverted* -- the displacement-driven exact inverse of an effort bank.
    """

    def __init__(
        self, plays: int, hidden: int, *, odd: bool, input_is: Causality, shaped: bool = False, inverted: bool = False
    ) -> None:
        super().__init__()
        self.arguments = {
            'plays': plays,
            'hidden': hidden,
            'odd': odd,
            'input_is': _causality(input_is),
            'shaped': shaped,
            'inverted': inverted,
        }
        if inverted and input_is != 'displacement':
            refuse('an inverted play bank is the displacement-driven inverse of an effort bank: declare displacement')
        self.input_is = input_is
        self.inverted = inverted
        #: The construction the parameters define: the effort form drives its plays by the effort itself.
        self.form: Causality = 'effort' if inverted else input_is
        start = torch.linspace(0.02, 1.0, plays)  # thresholds spread over the scaled input amplitude
        self.raw_thresholds = nn.Parameter(torch.log(torch.expm1(start)))
        self.raw_weights = nn.Parameter(torch.full((plays,), float(np.log(np.expm1(1.0 / plays)))))
        self.raw_slope = nn.Parameter(torch.tensor(-2.0))
        if shaped:  # each element's own odd monotone shape: w z + v tanh(s z / r), starting near linear
            self.raw_shape_weights = nn.Parameter(torch.full((plays,), -3.0))
            self.raw_shape_rates = nn.Parameter(torch.zeros(plays))
        self.shaped = shaped
        self.monotone = MonotoneScalar(hidden, odd=odd)

    @property
    def thresholds(self) -> torch.Tensor:
        return nn.functional.softplus(self.raw_thresholds)

    def played(self, x: torch.Tensor) -> torch.Tensor:
        """What the plays are driven by: the input itself (effort), else ``G(x)`` (displacement)."""
        return x if self.form == 'effort' else self.monotone(x)

    def shape(self, z: torch.Tensor) -> torch.Tensor:
        """Each element's contribution ``(..., K)``: ``w z`` (linear), ``w z + v tanh(s z)`` (shaped).

        Odd and strictly increasing in ``z`` (``w, v, s >= 0``), so the order of a loop's branches -- and with it
        passivity -- survives. ``z`` is a play output (effort) or a stop output ``u - p in [-r, r]``
        (displacement), where ``s`` is per unit ``r`` so the knee lies inside that span.
        """
        sp = nn.functional.softplus
        out = sp(self.raw_weights) * z
        if self.shaped:
            rate = sp(self.raw_shape_rates)
            if self.form == 'displacement':
                rate = rate / (self.thresholds + 1e-3)
            out = out + sp(self.raw_shape_weights) * torch.tanh(rate * z)
        return out

    def output_of(self, u: torch.Tensor, plays: torch.Tensor) -> torch.Tensor:
        """The output for played inputs ``u (...)`` and their play outputs ``plays (..., K)``."""
        a = nn.functional.softplus(self.raw_slope)
        if self.form == 'effort':
            return self.monotone(a * u + self.shape(plays).sum(-1))
        return a * u + self.shape(u.unsqueeze(-1) - plays).sum(-1)

    def forward(self, state: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.inverted:
            return self._inverse(state, x)
        u = self.played(x)
        state = play(state, u, self.thresholds)
        return state, self.output_of(u, state)

    def _effort_step(self, state: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        new = play(state, h, self.thresholds)
        return new, self.output_of(h, new)

    def _inverse(self, state: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The effort ``h`` whose step from *state* reaches the displacement *b*, and the plays it leaves.

        At fixed state the effort step is strictly increasing in ``h`` (slope ``>= a F' > 0``), so bisection on a
        bracket grown until it holds *b* converges to the unique root; one Newton step on top, through autograd,
        carries the implicit derivative ``dh/db = 1 / (db/dh)`` and polishes the value.
        """
        with torch.no_grad():
            target, frozen = b.detach(), state.detach()
            lo, hi = torch.full_like(target, -1.0), torch.full_like(target, 1.0)
            for _ in range(60):  # grow: the root may lie past the trained effort span
                below, above = self._effort_step(frozen, lo)[1] > target, self._effort_step(frozen, hi)[1] < target
                if not (below.any() or above.any()):
                    break
                lo, hi = torch.where(below, 2 * lo, lo), torch.where(above, 2 * hi, hi)
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                high = self._effort_step(frozen, mid)[1] > target
                lo, hi = torch.where(high, lo, mid), torch.where(high, mid, hi)
            root = 0.5 * (lo + hi)
        with torch.enable_grad():
            trial = root.clone().requires_grad_(True)
            reached = self._effort_step(frozen, trial)[1]
            (slope,) = torch.autograd.grad(reached.sum(), trial)
        h = root - (self._effort_step(frozen, root)[1] - b) / slope.detach()
        return play(state, h.detach(), self.thresholds), h

    def histories(self, x: torch.Tensor) -> torch.Tensor:
        """The play outputs ``(m, T, K)`` along input histories ``x (m, T)`` from the demagnetised state.

        The play recursion unrolled exactly: every play output is either 0 (never moved) or
        ``x[tau] - sigma r`` with ``tau`` the last step it moved and ``sigma`` its direction. Which branch and
        which ``tau`` is decided without autograd (numpy, at the current thresholds); the value is then one
        differentiable gather. Its gradient is the gradient of the stepwise recursion -- every ``where`` there
        selects the same branch -- at a fraction of the cost: no graph through T sequential steps.
        """
        r = self.thresholds
        xn, rn = numpy(x), numpy(r)
        m, length = xn.shape
        state = np.zeros((m, rn.size))
        anchor = np.zeros((m, rn.size), dtype=np.int64)
        sign = np.zeros((m, rn.size))
        anchors, signs = np.empty((m, length, rn.size), np.int64), np.empty((m, length, rn.size))
        for t in range(length):
            gap = xn[:, t, None] - state
            rising, falling = gap > rn, gap < -rn
            moved = rising | falling
            sign = np.where(rising, 1.0, np.where(falling, -1.0, sign))
            anchor = np.where(moved, t, anchor)
            state = np.where(moved, xn[:, t, None] - sign * rn, state)
            anchors[:, t], signs[:, t] = anchor, sign
        anchored = torch.gather(x, 1, torch.as_tensor(anchors.reshape(m, -1), device=x.device)).reshape(anchors.shape)
        return (
            anchored * (torch.as_tensor(signs, dtype=x.dtype, device=x.device) != 0)
            - torch.as_tensor(signs, dtype=x.dtype, device=x.device) * r
        )


@dataclass(frozen=True, slots=True)
class PlayHysteresis:
    """*input_is* declares the causality -- the conjugate variable that drives the model. No default: the
    orientation decides which loop is passive, so it is never guessed.
    """

    input_is: Causality = field(kw_only=True)
    plays: int = 16
    hidden: int = 16
    odd: bool = True
    shaped: bool = False
    inverted: bool = False
    schedule: Schedule = field(default_factory=lambda: Schedule(adam_steps=1500, learning_rate=1e-2))
    runtime: Runtime = field(default_factory=Runtime)

    def fit(
        self, inputs: Sequence[np.ndarray], outputs: Sequence[np.ndarray], *, checkpoint: Checkpoint | None = None
    ) -> FittedHysteresis:
        """Input histories and the outputs measured along them, each from the demagnetised state."""
        xs = [require_finite(np.asarray(x, float).ravel(), 'an input history') for x in inputs]
        ys = [require_finite(np.asarray(y, float).ravel(), 'an output history') for y in outputs]
        if not xs or len(xs) != len(ys) or any(x.shape != y.shape or x.size < 2 for x, y in zip(xs, ys, strict=True)):
            refuse(f'hysteresis is fitted on matched histories (each >= 2 samples), got {len(xs)} and {len(ys)}')
        _causality(self.input_is)
        if self.plays < 1 or self.hidden < 1:
            refuse(f'a play bank has >= 1 play and >= 1 hidden unit, got {self.plays}, {self.hidden}')
        rt = self.runtime
        rt.seeded()
        # Scale without shifting: a shift would break the exact oddness of the operator.
        seen = np.concatenate(xs)
        x_scale = float(np.max(np.abs(seen))) or 1.0
        y_scale = float(np.max(np.abs(np.concatenate(ys)))) or 1.0
        # Pad by holding the last input: a play does not move, so the padding is masked out of the loss only.
        length = max(x.size for x in xs)
        x_pad = np.stack([np.pad(x, (0, length - x.size), mode='edge') for x in xs]) / x_scale
        y_pad = np.stack([np.pad(y, (0, length - y.size), mode='edge') for y in ys]) / y_scale
        mask = rt.tensor(np.arange(length)[None, :] < np.array([x.size for x in xs])[:, None])
        x_t, y_t = rt.tensor(x_pad), rt.tensor(y_pad)
        bank = PlayBank(
            self.plays, self.hidden, odd=self.odd, input_is=self.input_is, shaped=self.shaped, inverted=self.inverted
        ).to(rt.torch_device, rt.torch_dtype)
        # An inverted bank is an effort bank: fitted on the histories read the other way, its residual in the
        # displacement -- the variable the declared input is.
        driven, measured = (y_t, x_t) if self.inverted else (x_t, y_t)

        def loss() -> torch.Tensor:
            u = bank.played(driven)
            return (((bank.output_of(u, bank.histories(u)) - measured) ** 2) * mask).sum() / mask.sum()

        history = minimise(
            bank.parameters(), loss, self.schedule, checkpoint=checkpoint, config=repr(self), data=[x_t, y_t, mask]
        )
        span = float(np.max(np.abs(seen)))
        envelope = BoxEnvelope(np.array([-span]), np.array([span])) if self.odd else BoxEnvelope.of(seen[:, None])
        return FittedHysteresis(
            bank=bank.eval(), runtime=rt, x_scale=x_scale, y_scale=y_scale, envelope=envelope, history=history
        )


@dataclass(frozen=True, slots=True, eq=False)
class FittedHysteresis:
    bank: PlayBank
    runtime: Runtime
    x_scale: float
    y_scale: float
    envelope: BoxEnvelope
    history: list[float]

    @property
    def input_is(self) -> Causality:
        """The declared causality -- the orientation :func:`dissipation` measures this model in."""
        return self.bank.input_is

    @property
    def thresholds(self) -> np.ndarray:
        """``r_k``, >= 0 by construction: in the effort unit (effort; inverted), in the scaled ``G(x)`` (stop form)."""
        scale = {'effort': self.x_scale, 'inverted': self.y_scale}.get(
            'inverted' if self.bank.inverted else self.input_is, 1.0
        )
        return numpy(self.bank.thresholds) * scale

    def initial_state(self, points: int = 1) -> np.ndarray:
        """The demagnetised state of *points* independent material points: ``(points, plays)``, every play at 0."""
        return np.zeros((points, self.bank.raw_thresholds.numel()))

    def _forward(
        self, state: np.ndarray, x: np.ndarray, *, grad: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = np.asarray(x, float).ravel()
        state = require_finite(np.asarray(state, float), 'the play state')
        if state.shape != (x.size, self.bank.raw_thresholds.numel()):
            refuse(
                f'a play state is (points, plays) = ({x.size}, {self.bank.raw_thresholds.numel()}), got {state.shape}'
            )
        self.envelope.require(x[:, None], 'a hysteresis input')
        rt = self.runtime
        xt = rt.tensor(x / self.x_scale, grad=grad)
        new, y = self.bank(rt.tensor(state), xt)
        return xt, new, y

    def step(self, state: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(state (n, plays), x (n,)) -> (state, y (n,))``: one input increment of *n* material points."""
        with torch.no_grad():
            _, new, y = self._forward(state, x, grad=False)
        return numpy(new), numpy(y) * self.y_scale

    def value(self, state: np.ndarray, x: np.ndarray) -> np.ndarray:
        """``y`` after stepping from *state* to *x* -- the residual a Newton solve on ``x`` evaluates."""
        return self.step(state, x)[1]

    def derivative(self, state: np.ndarray, x: np.ndarray) -> np.ndarray:
        """``dy/dx`` of that step at fixed *state*: the branch tangent, ``> 0`` everywhere by construction."""
        xt, _, y = self._forward(state, x, grad=True)
        (slope,) = torch.autograd.grad(y.sum(), xt)
        return numpy(slope) * self.y_scale / self.x_scale

    def respond(self, x: np.ndarray) -> np.ndarray:
        """``y (T,)`` along the input history ``x (T,)`` from the demagnetised state."""
        state, out = self.initial_state(), []
        for xi in np.asarray(x, float).ravel():
            state, y = self.step(state, np.array([xi]))
            out.append(y[0])
        return np.array(out)
