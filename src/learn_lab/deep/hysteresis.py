"""A learned rate-independent hysteresis: a bank of play operators, a monotone output map, the state explicit.

``s = a x + sum_k w_k play_k(x)``, ``y = F(s)`` -- a generalized Prandtl-Ishlinskii operator -- with thresholds
``r_k >= 0``, weights ``w_k >= 0`` and slope ``a > 0`` through softplus, and ``F`` a
:class:`~learn_lab.deep.networks.MonotoneScalar`. What holds BY CONSTRUCTION:

- rate independence: a play moves only by how far its input went, never by how fast, so inserting samples on a
  monotone stretch of the input changes no output;
- monotone branches: on a rising (falling) input every play and ``s`` rise (fall), and ``F`` is increasing, so
  ``dy/dx >= a F' > 0`` -- the tangent a Newton solver needs never vanishes;
- passivity: each play lags its input by its threshold, so on a closed cycle ``oint x dp = r oint |dp| >= 0``;
  the operator has return-point memory, so any closed cycle splits into simple loops whose ascending branch
  lies below the descending one, and an increasing ``F`` keeps that order: ``oint x dy >= 0``
  (:func:`dissipation`).

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

import numpy as np
import torch
from torch import nn

from learn_lab.deep.fit import Schedule, minimise
from learn_lab.deep.networks import MonotoneScalar
from learn_lab.deep.runtime import Runtime, numpy
from learn_lab.errors import refuse, require_finite
from learn_lab.validity import BoxEnvelope

__all__ = ['FittedHysteresis', 'PlayHysteresis', 'dissipation', 'play']

Step = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def play(state: torch.Tensor, x: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    """One step of a bank of plays: ``state (n, K)``, ``x (n,)``, ``thresholds (K,)`` -> the new ``state``.

    A play follows its input at distance ``r`` once the input pulls further than ``r`` away, else holds. The
    threshold test is written out rather than as a clamp, so a NEGATIVE threshold is not silently a clamp
    with crossed bounds: it makes the play LEAD its input, an active element :func:`dissipation` catches.
    """
    x = x.unsqueeze(-1)
    gap = x - state
    return torch.where(gap > thresholds, x - thresholds, torch.where(gap < -thresholds, x + thresholds, state))


def dissipation(step: Step, state: np.ndarray, cycle: np.ndarray) -> float:
    """``oint x dy`` over the closed input *cycle* -- the energy one cycle dissipates; >= 0 for a passive model.

    The cycle is run twice from *state* and measured on the second pass, once the first has carried the state
    onto the cycle. Unit: input unit times output unit (A/m * T = J/m^3 for H and B).
    """
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
    return float(np.sum(0.5 * (x[1:] + x[:-1]) * np.diff(y)))


class PlayBank(nn.Module):
    """The operator on scaled input/output: plays, then ``F(a x + w . p)``."""

    def __init__(self, plays: int, hidden: int, *, odd: bool) -> None:
        super().__init__()
        self.arguments = {'plays': plays, 'hidden': hidden, 'odd': odd}
        start = torch.linspace(0.02, 1.0, plays)  # thresholds spread over the scaled input amplitude
        self.raw_thresholds = nn.Parameter(torch.log(torch.expm1(start)))
        self.raw_weights = nn.Parameter(torch.full((plays,), float(np.log(np.expm1(1.0 / plays)))))
        self.raw_slope = nn.Parameter(torch.tensor(-2.0))
        self.output = MonotoneScalar(hidden, odd=odd)

    @property
    def thresholds(self) -> torch.Tensor:
        return nn.functional.softplus(self.raw_thresholds)

    def output_of(self, x: torch.Tensor, plays: torch.Tensor) -> torch.Tensor:
        """``F(a x + w . p)`` for inputs ``x (...)`` and their play outputs ``plays (..., K)``."""
        sp = nn.functional.softplus
        return self.output(sp(self.raw_slope) * x + (sp(self.raw_weights) * plays).sum(-1))

    def forward(self, state: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = play(state, x, self.thresholds)
        return state, self.output_of(x, state)

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
    plays: int = 16
    hidden: int = 16
    odd: bool = True
    schedule: Schedule = field(default_factory=lambda: Schedule(adam_steps=1500, learning_rate=1e-2))
    runtime: Runtime = field(default_factory=Runtime)

    def fit(self, inputs: Sequence[np.ndarray], outputs: Sequence[np.ndarray]) -> FittedHysteresis:
        """Input histories and the outputs measured along them, each from the demagnetised state."""
        xs = [require_finite(np.asarray(x, float).ravel(), 'an input history') for x in inputs]
        ys = [require_finite(np.asarray(y, float).ravel(), 'an output history') for y in outputs]
        if not xs or len(xs) != len(ys) or any(x.shape != y.shape or x.size < 2 for x, y in zip(xs, ys, strict=True)):
            refuse(f'hysteresis is fitted on matched histories (each >= 2 samples), got {len(xs)} and {len(ys)}')
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
        bank = PlayBank(self.plays, self.hidden, odd=self.odd).to(rt.torch_device, rt.torch_dtype)

        def loss() -> torch.Tensor:
            return (((bank.output_of(x_t, bank.histories(x_t)) - y_t) ** 2) * mask).sum() / mask.sum()

        history = minimise(bank.parameters(), loss, self.schedule)
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
    def thresholds(self) -> np.ndarray:
        """``r_k`` in the input unit; >= 0 by construction."""
        return numpy(self.bank.thresholds) * self.x_scale

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
